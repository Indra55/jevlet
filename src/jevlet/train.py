"""Small supervised training loop and post-hoc calibration."""

import math
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import MULTILABEL_HEADS, device_for, metadata, seed_all, write_json
from .data import collate, packed_items, prepare_data, requests_for
from .metrics import fit_temperatures, fit_multilabel_threshold, score_records
from .model import Jevlet, training_loss


def loader(items, model, cfg, training=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(items, batch_size=cfg["train" if training else "eval"]["batch_size"],
                      shuffle=training, generator=generator,
                      collate_fn=partial(collate, pad_id=model.tokenizer.pad_token_id))


def on_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.inference_mode()
def collect_records(model, requests, cfg, items=None, description=None):
    model.eval()
    if description:
        print(f"{description}: packing {len(items) if items is not None else len(requests):,} requests", flush=True)
    items = items if items is not None else packed_items(model.tokenizer, requests, model.max_length)
    records = []
    device = next(model.parameters()).device
    batches = loader(items, model, cfg)
    if description:
        batches = tqdm(batches, desc=description, unit="batch", dynamic_ncols=True, mininterval=1)
    for batch in batches:
        batch = on_device(batch, device)
        outputs = {k: v.cpu() for k, v in model(batch).items()}
        for b, item in enumerate(batch["items"]):
            request = item["request"]
            for q, question in enumerate(request["questions"]):
                head = question["type"]
                logits = outputs[head][b, q]
                if head in ("choice",) + MULTILABEL_HEADS:
                    logits = logits[:len(question["options"])]
                records.append({**question, **({"threshold": getattr(model, head + "_threshold")} if head in MULTILABEL_HEADS else {}),
                                "source_id": request.get("source_id"),
                                "validation_view": request.get("validation_view", 0),
                                "score_mode": model.score_mode,
                                "logits": logits.tolist(),
                                "context_tokens_dropped": item["context_tokens_dropped"]})
    return records


def train(cfg, directory, bundle=None, head=None):
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"], cfg["cpu_threads"])
    print("Preparing source data...", flush=True)
    bundle = bundle or prepare_data(cfg, directory)
    write_json(directory / "data.json", bundle)
    device = device_for(cfg["device"])
    print(f"Loading {cfg['model']} on {device}...", flush=True)
    model = (Jevlet.load(cfg["initial_checkpoint"], str(device)) if cfg.get("initial_checkpoint")
             else Jevlet.pretrained(cfg).to(device))
    if cfg.get("multilabel"):
        model.enable_multilabel()
    if cfg.get("banking_multilabel"):
        model.enable_banking_multilabel()
    model.temperatures.fill_(1.)
    device = next(model.parameters()).device
    # Separate task baselines receive the same triple-derived supervision, unpacked by head.
    validation = requests_for(bundle, "validation", cfg, head=head)
    print(f"Packing {len(validation):,} validation requests...", flush=True)
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
        print(f"Packing epoch {epoch + 1}/{cfg['train']['epochs']}: {len(requests):,} training requests...", flush=True)
        items = packed_items(model.tokenizer, requests, model.max_length)
        batches = loader(items, model, cfg, training=True, epoch=epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        epoch_started = time.perf_counter()
        progress = tqdm(batches, desc=f"Train {epoch + 1}/{cfg['train']['epochs']}",
                        unit="batch", dynamic_ncols=True, mininterval=1)
        for step, batch in enumerate(progress):
            batch = on_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                loss = training_loss(model, model(batch), batch, cfg["train"]["brier_weight"])
            # The final partial accumulation group uses its actual batch count.
            group_start = step // cfg["train"]["accumulation"] * cfg["train"]["accumulation"]
            divisor = min(cfg["train"]["accumulation"], len(batches) - group_start)
            scaler.scale(loss / divisor).backward()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", mean=f"{total_loss / (step + 1):.4f}", refresh=False)
            if (step + 1) % cfg["train"]["accumulation"] == 0 or step + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == len(batches):
                elapsed_epoch = time.perf_counter() - epoch_started
                remaining_steps = (cfg["train"]["epochs"] - epoch - 1) * len(batches) + len(batches) - step - 1
                write_json(directory / "progress.json", {
                    "phase": "training", "epoch": epoch + 1, "max_epochs": cfg["train"]["epochs"],
                    "batch": step + 1, "batches_per_epoch": len(batches),
                    "mean_training_loss": total_loss / (step + 1),
                    "elapsed_seconds": time.perf_counter() - started,
                    "estimated_remaining_training_seconds": remaining_steps * elapsed_epoch / (step + 1),
                    "estimate_note": "Based on current epoch speed; excludes validation, calibration and evaluation; early stopping may shorten the run.",
                })
        model.eval()
        write_json(directory / "progress.json", {"phase": "validation", "epoch": epoch + 1,
                   "max_epochs": cfg["train"]["epochs"], "elapsed_seconds": time.perf_counter() - started})
        with torch.inference_mode():
            losses = []
            validation_progress = tqdm(loader(val_items, model, cfg), desc=f"Validate {epoch + 1}",
                                       unit="batch", dynamic_ncols=True, mininterval=1)
            for batch in validation_progress:
                batch = on_device(batch, device)
                losses.append(training_loss(model, model(batch), batch, cfg["train"]["brier_weight"]).item())
                validation_progress.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)
        val_loss = sum(losses) / len(losses)
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        entry = {"epoch": epoch + 1, "train_loss": total_loss / len(batches), "validation_loss": val_loss}
        selection_loss = val_loss
        if cfg.get("checkpoint_metric") == "sentiment_accuracy":
            metrics, _ = score_records(collect_records(model, validation, cfg, val_items))
            entry["validation_metrics"] = metrics
            selection_loss = -metrics["score"]["accuracy"]
        history.append(entry)
        print(f"epoch {epoch + 1}: train={history[-1]['train_loss']:.4f} validation={val_loss:.4f}", flush=True)
        write_json(directory / "progress.json", {"history": history, "device": str(device),
                   "elapsed_seconds": time.perf_counter() - started,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        if selection_loss < best:
            best, stale = selection_loss, 0
            print("Saving best checkpoint...", flush=True)
            model.save(directory)
        else:
            stale += 1
            if stale >= cfg["train"]["patience"]:
                break
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del optimizer, scheduler, model
    model = Jevlet.load(directory, str(device))
    write_json(directory / "progress.json", {"phase": "calibration", "elapsed_seconds": elapsed})
    calibration = collect_records(model, requests_for(bundle, "calibration", cfg, head=head), cfg,
                                  description="Calibration")
    print("Fitting temperatures on the calibration split...", flush=True)
    temperatures, details = fit_temperatures(calibration, str(device))
    model.temperatures.copy_(torch.tensor([temperatures[h] for h in model.heads], device=device))
    threshold_fit = {}
    if any(head in model.heads for head in MULTILABEL_HEADS):
        validation_records = collect_records(model, validation, cfg, items=val_items,
                                             description="Threshold validation")
        for head in MULTILABEL_HEADS:
            if head in model.heads:
                threshold, threshold_fit[head] = fit_multilabel_threshold(validation_records, temperatures[head], head)
                setattr(model, head + "_threshold", threshold)
    model.save(directory)
    write_json(directory / "calibration.json", {"temperatures": temperatures, "fit": details,
               "multilabel_threshold": getattr(model, "multilabel_threshold", None),
               "banking_multilabel_threshold": getattr(model, "banking_multilabel_threshold", None),
               "threshold_fit": threshold_fit.get("multilabel"), "banking_threshold_fit": threshold_fit.get("banking_multilabel")})
    write_json(directory / "training.json", {**metadata(cfg, bundle["sources"]), "head": head,
               "history": history, "training_seconds": elapsed, "peak_gpu_bytes": peak,
               "model_revision": model.source_revision})
    write_json(directory / "progress.json", {"phase": "trained", "epochs": len(history),
               "training_seconds": elapsed})
    return model, bundle
