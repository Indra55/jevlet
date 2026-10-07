"""Held-out task metrics and matched-workload latency benchmarks."""

import json
import platform
import time
from itertools import cycle, islice

import numpy as np
import torch

from . import ROOT
from .config import HEADS, metadata, seed_all, write_json
from .data import collate, packed_items, requests_for
from .metrics import make_figures, risk_coverage, score_records
from .model import Jevlet
from .train import collect_records, on_device


def unpack_items(tokenizer, items, max_length):
    """Keep exactly the context tokens that survived the original packed request."""
    separate = []
    for item in items:
        ids = item["input_ids"]
        shared_context = ids[ids.index(tokenizer.sep_token_id) + 1:-1]
        request = item["request"]
        for question in request["questions"]:
            single = packed_items(tokenizer, [{**request, "questions": [question]}], max_length)[0]
            prefix = single["input_ids"][:single["input_ids"].index(tokenizer.sep_token_id) + 1]
            single["input_ids"] = prefix + shared_context + [tokenizer.sep_token_id]
            single["context_tokens_dropped"] = item["context_tokens_dropped"]
            separate.append(single)
    return separate


def forward_batches(model, items, mode):
    if mode == "packed":
        groups = [items]
    else:
        separate = unpack_items(model.tokenizer, items, model.max_length)
        # Question-major grouping gives three passes, each with B contexts.
        groups = [separate] if mode == "batched_questions" else [separate[i::3] for i in range(3)]
    device = next(model.parameters()).device
    return [on_device(collate(group, model.tokenizer.pad_token_id), device) for group in groups]


@torch.inference_mode()
def benchmark(model, requests, cfg, devices=None):
    requests = [r for r in requests if len(r["questions"]) == 3]
    if not requests:
        return {"status": "no_three_question_requests"}
    original = str(next(model.parameters()).device)
    devices = devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    output = {"hardware": {"cpu": platform.processor() or platform.machine(),
              "cpu_threads": torch.get_num_threads(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "precision": "float32", "questions_per_request": 3,
              "cuda": {"status": "not_available"} if not torch.cuda.is_available() else {}, "measurements": []}
    try:
        for device_name in devices:
            model.to(device_name).eval()
            device = next(model.parameters()).device

            def synchronize():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            for size in (1, 32):
                selected = list(islice(cycle(requests[:cfg["eval"]["latency_requests"]]), size))
                for mode in ("packed", "sequential", "batched_questions"):
                    items = packed_items(model.tokenizer, selected, model.max_length)
                    batches = forward_batches(model, items, mode)

                    def run_forward():
                        for batch in batches:
                            model.probabilities(model(batch), batch["omask"])

                    for _ in range(cfg["eval"]["latency_warmup"]):
                        run_forward()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    forwards, complete = [], []
                    for _ in range(cfg["eval"]["latency_repeats"]):
                        synchronize()
                        started = time.perf_counter()
                        run_forward()
                        synchronize()
                        forwards.append((time.perf_counter() - started) * 1000)
                        synchronize()
                        started = time.perf_counter()
                        fresh = packed_items(model.tokenizer, selected, model.max_length)
                        for batch in forward_batches(model, fresh, mode):
                            model.probabilities(model(batch), batch["omask"])
                        synchronize()
                        complete.append((time.perf_counter() - started) * 1000)
                    median = float(np.median(forwards))
                    output["measurements"].append({"device": str(device), "mode": mode, "batch_size": size,
                        "forward_calls": len(batches), "encoder_rows": sum(len(b["input_ids"]) for b in batches),
                        "sequence_lengths": [b["input_ids"].shape[1] for b in batches],
                        "p50_forward_ms_per_request": median / size,
                        "p95_forward_ms_per_request": float(np.percentile(forwards, 95)) / size,
                        "p50_pack_and_forward_ms_per_request": float(np.median(complete)) / size,
                        "requests_per_second": 1000 * size / median,
                        "precision": "float32", "scope": "packing and forward; excludes loading and JSON",
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "repeats": len(forwards), "warmup": cfg["eval"]["latency_warmup"]})
    finally:
        model.to(original)
    return output


def write_evaluation(records, temperatures, cfg, run, name, data_sources, latency=None):
    raw, before = score_records(records, bins=cfg["eval"]["ece_bins"])
    scaled, after = score_records(records, temperatures, cfg["eval"]["ece_bins"])
    result = {"run": run, "model": name, "seed": cfg["seed"], "smoke": cfg["is_smoke"],
              **metadata(cfg, data_sources), "raw": raw, "calibrated": scaled,
              "temperatures": temperatures, "latency": latency,
              "risk_coverage": {h: dict(zip(("coverage", "risk"), risk_coverage(after, h)))
                                for h in scaled}}
    stem = f"{run}_{name}"
    write_json(ROOT / "results" / f"{stem}.json", result)
    write_json(ROOT / "results/predictions" / f"{stem}.json", {"raw": before, "calibrated": after})
    make_figures(before, after, ROOT / "results/figs" / stem, cfg["eval"]["ece_bins"])
    return result


def evaluate(cfg, directory, run, model=None, bundle=None, name="jevlet", with_latency=True):
    seed_all(cfg["seed"], cfg["cpu_threads"])
    model = model or Jevlet.load(directory, cfg["device"])
    bundle = bundle or json.loads((directory / "data.json").read_text())
    requests = requests_for(bundle, "test", cfg)
    records = collect_records(model, requests, cfg)
    temperatures = dict(zip(HEADS, model.temperatures.cpu().tolist()))
    latency = benchmark(model, requests, cfg) if with_latency else None
    result = write_evaluation(records, temperatures, cfg, run, name, bundle["sources"], latency)
    print(json.dumps(result["calibrated"], indent=2), flush=True)
    return result
