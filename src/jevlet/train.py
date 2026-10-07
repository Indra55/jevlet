"""Small supervised training loop and post-hoc calibration."""

import math
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import HEADS, device_for, metadata, seed_all, write_json
from .data import collate, packed_items, prepare_data, requests_for
from .metrics import fit_temperatures
from .model import Jevlet, training_loss


def loader(items, model, cfg, training=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(items, batch_size=cfg["train" if training else "eval"]["batch_size"],
                      shuffle=training, generator=generator,
                      collate_fn=partial(collate, pad_id=model.tokenizer.pad_token_id))


def on_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.inference_mode()
def collect_records(model, requests, cfg, items=None):
    model.eval()
    items = items if items is not None else packed_items(model.tokenizer, requests, model.max_length)
    records = []
    device = next(model.parameters()).device
    for batch in loader(items, model, cfg):
        batch = on_device(batch, device)
        outputs = {k: v.cpu() for k, v in model(batch).items()}
        for b, item in enumerate(batch["items"]):
            request = item["request"]
            for q, question in enumerate(request["questions"]):
                head = question["type"]
                logits = outputs[head][b, q]
                if head == "choice":
                    logits = logits[:len(question["options"])]
                records.append({**question, "source_id": request.get("source_id"),
                                "logits": logits.tolist(),
                                "context_tokens_dropped": item["context_tokens_dropped"]})
    return records


def train(cfg, directory, bundle=None, head=None):
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"], cfg["cpu_threads"])
    bundle = bundle or prepare_data(cfg, directory)
    write_json(directory / "data.json", bundle)
    model = Jevlet.pretrained(cfg).to(device_for(cfg["device"]))
    device = next(model.parameters()).device
    # Separate task baselines receive the same triple-derived supervision, unpacked by head.
    validation = requests_for(bundle, "validation", cfg, head=head)
    val_items = packed_items(model.tokenizer, validation, model.max_length)
    initial = requests_for(bundle, "train", cfg, training=head is None, head=head)
    steps_per_epoch = math.ceil(math.ceil(len(initial) / cfg["train"]["batch_size"]) /
                                cfg["train"]["accumulation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["train"]["warmup"]), total_steps)
    amp = device.type == "cuda" and cfg["train"]["mixed_precision"]
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    history, best, stale = [], math.inf, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg["train"]["epochs"]):
        requests = requests_for(bundle, "train", cfg, epoch, training=head is None, head=head)
        items = packed_items(model.tokenizer, requests, model.max_length)
        batches = loader(items, model, cfg, training=True, epoch=epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(batches):
            batch = on_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                loss = training_loss(model, model(batch), batch, cfg["train"]["brier_weight"])
            # The final partial accumulation group uses its actual batch count.
            group_start = step // cfg["train"]["accumulation"] * cfg["train"]["accumulation"]
            divisor = min(cfg["train"]["accumulation"], len(batches) - group_start)
            scaler.scale(loss / divisor).backward()
            total_loss += loss.item()
            if (step + 1) % cfg["train"]["accumulation"] == 0 or step + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            losses = []
            for batch in loader(val_items, model, cfg):
                batch = on_device(batch, device)
                losses.append(training_loss(model, model(batch), batch, cfg["train"]["brier_weight"]).item())
        val_loss = sum(losses) / len(losses)
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(batches), "validation_loss": val_loss})
        print(f"epoch {epoch + 1}: train={history[-1]['train_loss']:.4f} validation={val_loss:.4f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            model.save(directory)
        else:
            stale += 1
            if stale >= cfg["train"]["patience"]:
                break
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del optimizer, scheduler, model
    model = Jevlet.load(directory, str(device))
    calibration = collect_records(model, requests_for(bundle, "calibration", cfg, head=head), cfg)
    temperatures, details = fit_temperatures(calibration)
    model.temperatures.copy_(torch.tensor([temperatures[h] for h in HEADS], device=device))
    model.save(directory)
    write_json(directory / "calibration.json", {"temperatures": temperatures, "fit": details})
    write_json(directory / "training.json", {**metadata(cfg, bundle["sources"]), "head": head,
               "history": history, "training_seconds": elapsed, "peak_gpu_bytes": peak,
               "model_revision": model.source_revision})
    return model, bundle
