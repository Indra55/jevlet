"""Resumable CUDA-only fitting/inference for a controlled classifier comparison.

CPU tokenization, sparse feature extraction and descriptive metrics are allowed;
all learned classifier forward/backward passes and temperature fitting use CUDA.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import ROOT
from .config import HEADS, load_config, metadata, seed_all, write_json
from .data import requests_for
from .eval import benchmark, evaluate, write_evaluation
from .metrics import fit_temperatures, score_records
from .model import Jevlet
from .train import collect_records, train

NAMES = ["jevlet_ordinal", "jevlet_softmax", "tfidf_logistic", "tfidf_svm",
         "distilbert_classifier", "bert_classifier", "roberta_classifier"]
ENCODERS = {"distilbert_classifier": "distilbert/distilbert-base-uncased",
            "bert_classifier": "google-bert/bert-base-uncased",
            "roberta_classifier": "FacebookAI/roberta-base"}
BASE = ROOT / "checkpoints/gpu_comparison"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; CPU fallback is forbidden")


def monitor(stop, path):
    with path.open("a", buffering=1) as handle:
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                handle.write(json.dumps({"time": time.time(), "gpu": result.stdout.strip(),
                                         "exit_code": result.returncode}) + "\n")
            except Exception as exc:
                handle.write(json.dumps({"time": time.time(), "error": str(exc)}) + "\n")
            stop.wait(10)


class ContextClassifier(nn.Module):
    """Closed-label classifiers: CLS features, 77 intent, five sentiment, 77 predicates."""
    def __init__(self, encoder=None, features=None):
        super().__init__()
        self.encoder = encoder
        size = features if encoder is None else encoder.config.hidden_size
        self.dropout = nn.Identity() if encoder is None else nn.Dropout(0.1)
        self.intent = nn.Linear(size, 77)
        self.sentiment = nn.Linear(size, 5)
        self.binary = nn.Linear(size, 77)

    def forward(self, batch):
        assert next(self.parameters()).is_cuda
        hidden = batch if self.encoder is None else self.encoder(**batch).last_hidden_state[:, 0]
        hidden = self.dropout(hidden)
        return self.intent(hidden).float(), self.sentiment(hidden).float(), self.binary(hidden).float()


def tensors_from_questions(requests):
    indices, labels, targets, kinds = [], [], [], []
    for i, r in enumerate(requests):
        for q in r["questions"]:
            indices.append(i)
            labels.append(q.get("option_labels", [q.get("query_label", 0)]))
            targets.append(q["target"])
            kinds.append(q["type"])
    return indices, labels, targets, kinds


def loss_for(outputs, requests, svm=False):
    # Vectorized within each head, preserving exactly the sampled supervision.
    indices, labels, targets, kinds = tensors_from_questions(requests)
    values = []
    for head in HEADS:
        selected = [j for j, k in enumerate(kinds) if k == head]
        if not selected:
            continue
        b = torch.tensor([indices[j] for j in selected], device="cuda")
        y = torch.tensor([targets[j] for j in selected], device="cuda")
        if head == "choice":
            z = torch.full((len(selected), 10), -1e4, device="cuda")
            mask = torch.zeros_like(z, dtype=torch.bool)
            for i, j in enumerate(selected):
                ids = torch.tensor(labels[j], device="cuda")
                z[i, :len(ids)] = outputs[0][b[i], ids]
                mask[i, :len(ids)] = True
        elif head == "score":
            z = outputs[1][b]
            mask = torch.ones_like(z, dtype=torch.bool)
        else:
            queried = torch.tensor([labels[j][0] for j in selected], device="cuda")
            z = outputs[2][b, queried]
        if svm:
            if head == "yesno":
                loss = F.relu(1 - (2 * y - 1) * z).square().mean()
            else:
                signs = 2 * F.one_hot(y, z.shape[-1]) - 1
                terms = F.relu(1 - signs * z).square() * mask
                loss = (terms.sum(-1) / mask.sum(-1)).mean()
        elif head == "yesno":
            p = z.sigmoid()
            loss = F.binary_cross_entropy_with_logits(z, y.float()) + 0.5 * (2 * (p - y).square()).mean()
        else:
            p = z.softmax(-1)
            loss = F.cross_entropy(z, y) + 0.5 * (p - F.one_hot(y, z.shape[-1])).square().sum(-1).mean()
        values.append(loss)
    return torch.stack(values).mean()


class Features:
    def __init__(self, bundle, name, directory):
        self.linear = name.startswith("tfidf")
        self.lookup = {}
        self.bundle = bundle
        if self.linear:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
                                             dtype=np.float32)
            self.vectorizer.fit([r["text"] for r in bundle["splits"]["train"]])
            joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
            for split, rows in bundle["splits"].items():
                matrix = self.vectorizer.transform([r["text"] for r in rows])
                for i, r in enumerate(rows):
                    self.lookup[r["id"]] = (matrix, i)
            self.size = len(self.vectorizer.vocabulary_)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(ENCODERS[name])
            self.tokenizer.save_pretrained(directory / "tokenizer")
            for rows in bundle["splits"].values():
                tokens = self.tokenizer([r["text"] for r in rows], truncation=True, max_length=256)
                for i, r in enumerate(rows):
                    self.lookup[r["id"]] = {k: v[i] for k, v in tokens.items()}

    def batch(self, requests):
        if self.linear:
            # Only the current small minibatch is densified; matrix operations stay on CUDA.
            a = np.vstack([self.lookup[r["source_id"]][0][self.lookup[r["source_id"]][1]].toarray()
                           for r in requests])
            return torch.from_numpy(a).to("cuda")
        values = [self.lookup[r["source_id"]] for r in requests]
        return self.tokenizer.pad(values, return_tensors="pt").to("cuda")


@torch.inference_mode()
def records_for(model, features, requests, batch_size=32):
    model.eval()
    records = []
    for start in range(0, len(requests), batch_size):
        group = requests[start:start + batch_size]
        a, s, b = [v.cpu() for v in model(features.batch(group))]
        for i, r in enumerate(group):
            for q in r["questions"]:
                h = q["type"]
                logits = a[i, q["option_labels"]] if h == "choice" else s[i] if h == "score" else b[i, q["query_label"]]
                records.append({**q, "source_id": r["source_id"], "logits": logits.tolist(),
                                "score_mode": "categorical"})
    return records


def conventional(cfg, bundle, name, directory):
    seed_all(cfg["seed"], cfg["cpu_threads"])
    features = Features(bundle, name, directory)
    encoder = None if features.linear else AutoModel.from_pretrained(ENCODERS[name])
    model = ContextClassifier(encoder, features.size if features.linear else None).cuda()
    revision = None if features.linear else getattr(encoder.config, "_commit_hash", None)
    parameter_count = sum(p.numel() for p in model.parameters())
    linear = features.linear
    local = copy.deepcopy(cfg)
    if not linear:
        local["model"], local["model_revision"] = ENCODERS[name], revision
    if linear:
        local["model"], local["model_revision"] = name, None
        local["train"].update(epochs=40, batch_size=256, accumulation=1, lr=0.01, patience=5,
                              weight_decay=0.0001, mixed_precision=False)
    if (directory / "training.json").exists():
        model.load_state_dict(torch.load(directory / "classifier.pt", map_location="cuda", weights_only=True))
        return model, features, json.loads((directory / "training.json").read_text())
    optimizer = torch.optim.AdamW(model.parameters(), lr=local["train"]["lr"],
                                 weight_decay=local["train"]["weight_decay"])
    steps = math.ceil(math.ceil(len(bundle["splits"]["train"]) / local["train"]["batch_size"]) /
                      local["train"]["accumulation"]) * local["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * local["train"]["warmup"]), steps)
    validation = requests_for(bundle, "validation", cfg)
    best, stale, history = math.inf, 0, []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(local["train"]["epochs"]):
        requests = requests_for(bundle, "train", cfg, epoch, training=True)
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(len(requests))
        batches = [order[i:i + local["train"]["batch_size"]] for i in range(0, len(order), local["train"]["batch_size"])]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for step, ids in enumerate(batches):
            group = [requests[int(i)] for i in ids]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not linear):
                loss = loss_for(model(features.batch(group)), group, name == "tfidf_svm")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {name}")
            accumulation = local["train"]["accumulation"]
            divisor = min(accumulation, len(batches) - step // accumulation * accumulation)
            (loss / divisor).backward()
            total += loss.item()
            if (step + 1) % accumulation == 0 or step + 1 == len(batches):
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        val = []
        with torch.inference_mode():
            for i in range(0, len(validation), local["eval"]["batch_size"]):
                group = validation[i:i + local["eval"]["batch_size"]]
                val.append(loss_for(model(features.batch(group)), group, name == "tfidf_svm").item())
        val_loss = float(np.mean(val))
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total / len(batches), "validation_loss": val_loss})
        write_json(directory / "progress.json", {"history": history, "elapsed_seconds": time.perf_counter() - started,
                                                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        print(f"{name} epoch {epoch+1}: train={total/len(batches):.4f} validation={val_loss:.4f} "
              f"peak VRAM={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), directory / "classifier.pt")
            if encoder is not None:
                encoder.config.save_pretrained(directory / "encoder")
        else:
            stale += 1
            if stale >= local["train"]["patience"]:
                break
    info = {**metadata(local, bundle["sources"]), "history": history,
            "training_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "model_revision": revision, "parameters": parameter_count, "device": "cuda",
            "architecture": "context CLS with three task-specific classifiers" if not linear else "TF-IDF linear CUDA classifiers",
            "loss": "squared one-vs-rest hinge" if name == "tfidf_svm" else "CE/BCE plus 0.5 Brier"}
    write_json(directory / "training.json", info)
    del optimizer, scheduler
    model.load_state_dict(torch.load(directory / "classifier.pt", map_location="cuda", weights_only=True))
    return model, features, info


@torch.inference_mode()
def context_latency(model, features, requests, temperatures):
    bank = [r for r in requests if len(r["questions"]) == 3]
    output = []
    # Same fixed banking requests as Jevlet's benchmark, float32 for neural models.
    for size in (1, 32):
        selected = bank[:size]
        prepared = features.batch(selected)
        option_ids = torch.zeros((size, 10), dtype=torch.long, device="cuda")
        option_mask = torch.zeros((size, 10), dtype=torch.bool, device="cuda")
        query_ids = torch.zeros((size, 2), dtype=torch.long, device="cuda")
        for i, request in enumerate(selected):
            choice = next(q for q in request["questions"] if q["type"] == "choice")
            ids = choice["option_labels"]
            option_ids[i, :len(ids)] = torch.tensor(ids, device="cuda")
            option_mask[i, :len(ids)] = True
            query_ids[i] = torch.tensor([q["query_label"] for q in request["questions"]
                                         if q["type"] == "yesno"], device="cuda")
        def forward(batch):
            a, s, b = model(batch)
            # Score all three banking questions and form calibrated probabilities on CUDA.
            (a.gather(1, option_ids) / temperatures["choice"]).masked_fill(~option_mask, -torch.inf).softmax(-1)
            (b.gather(1, query_ids) / temperatures["yesno"]).sigmoid()
        forwards, complete = [], []
        for _ in range(5):
            forward(prepared)
        for _ in range(20):
            torch.cuda.synchronize()
            started = time.perf_counter()
            forward(prepared)
            torch.cuda.synchronize()
            forwards.append((time.perf_counter() - started) * 1000 / size)
            started = time.perf_counter()
            # Fresh preprocessing, rather than token/feature cache lookups, for complete time.
            if features.linear:
                fresh = torch.from_numpy(features.vectorizer.transform([r["text"] for r in selected]).toarray()).cuda()
            else:
                fresh = features.tokenizer([r["text"] for r in selected], truncation=True, max_length=256,
                                          padding=True, return_tensors="pt").to("cuda")
            forward(fresh)
            torch.cuda.synchronize()
            complete.append((time.perf_counter() - started) * 1000 / size)
        output.append({"device": "cuda:0", "batch_size": size, "mode": "context_classifier",
                       "p50_forward_ms_per_request": float(np.median(forwards)),
                       "p95_forward_ms_per_request": float(np.percentile(forwards, 95)),
                       "p50_pack_and_forward_ms_per_request": float(np.median(complete)),
                       "precision": "float32", "warmup": 5, "repeats": 20})
    return {"measurements": output, "scope": "GPU forward/probabilities and fresh preprocessing; excludes loading/JSON"}


def print_report(summary, names, history=False):
    """Display recorded metrics/history without loading models or running CUDA."""
    rows = [(name, summary["models"][name]) for name in names if name in summary["models"]]
    refined_path = ROOT / "results/jevlet_refinement.json"
    if "jevlet_softmax" in names and refined_path.exists():
        refined = json.loads(refined_path.read_text())
        if "test" in refined:
            checkpoint = Path(refined["selected_checkpoint"])
            info_path = checkpoint / "training.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            rows.append(("jevlet_refined", {"calibrated": refined["test"]["calibrated"],
                         "checkpoint": str(checkpoint), "epochs_completed": len(info.get("history", []))}))
    if not rows:
        print("No saved results for the requested models.", flush=True)
        return
    print("\nSAVED TEST RESULTS (calibrated; accuracy in percent)", flush=True)
    print(f'{"Model":<25} {"Choice %":>9} {"Sent. %":>9} {"Sent. F1":>9} {"Binary %":>9}', flush=True)
    for name, result in rows:
        metrics = result["calibrated"]
        print(f'{name:<25} {100 * metrics["choice"]["accuracy"]:>9.2f} '
              f'{100 * metrics["score"]["accuracy"]:>9.2f} '
              f'{metrics["score"]["macro_f1"]:>9.4f} '
              f'{100 * metrics["yesno"]["accuracy"]:>9.2f}', flush=True)
    if any(name == "jevlet_refined" for name, _ in rows):
        print("Refined Jevlet comes from the separate refinement run.", flush=True)
    if not history:
        return
    print("\nRECORDED TRAINING HISTORY (previous runs; no training is started by this display)", flush=True)
    for name, result in rows:
        info_path = Path(result.get("checkpoint", str(BASE / name))) / "training.json"
        if not info_path.exists():
            print(f"\n{name}: training history file unavailable: {info_path}", flush=True)
            continue
        info = json.loads(info_path.read_text())
        print(f'\n{name}: {len(info["history"])} epochs, '
              f'{info["training_seconds"]:.1f} seconds, '
              f'peak allocated GPU memory {info["peak_gpu_bytes"] / 2**30:.2f} GiB', flush=True)
        for epoch in info["history"]:
            print(f'  epoch {epoch["epoch"]}: train_loss={epoch["train_loss"]:.4f} '
                  f'validation_loss={epoch["validation_loss"]:.4f}', flush=True)


def run(names, history=False):
    require_cuda()
    BASE.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["device"], cfg["seed"] = "cuda", 42
    cfg["train"]["epochs"] = 6
    # Freeze source partitions and the common pretrained DistilBERT revision.
    bundle = json.loads((ROOT / "checkpoints/full/data.json").read_text())
    cfg["model_revision"] = json.loads((ROOT / "checkpoints/full/training.json").read_text())["model_revision"]
    protocol = {"models": NAMES, "seed": 42, "config": cfg, "source_manifest_sha256":
                hashlib.sha256((ROOT / "checkpoints/full/data.json").read_bytes()).hexdigest(),
                "device": "cuda", "hardware": torch.cuda.get_device_name(),
                "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "selection": "validation loss; Jevlet variant selected by uncalibrated validation sentiment accuracy",
                "limitations": "single seed; existing test set previously inspected; model-specific context truncation",
                "started": time.time()}
    write_json(BASE / "protocol.json", protocol)
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, BASE / "vram.jsonl"), daemon=True)
    thread.start()
    requests = requests_for(bundle, "test", cfg)
    summary_path = ROOT / "results/gpu_comparison_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol": protocol, "models": {}}
    try:
        for name in names:
            if name in summary["models"]:
                print(f"REUSING completed {name}", flush=True)
                continue
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"START {name}: free VRAM {free/2**30:.2f}/{total/2**30:.2f} GiB", flush=True)
            write_json(BASE / "status.json", {"state": "running", "model": name, "completed": list(summary["models"]), "time": time.time()})
            directory = BASE / name
            directory.mkdir(parents=True, exist_ok=True)
            local = copy.deepcopy(cfg)
            if name.startswith("jevlet"):
                local["score_mode"] = "categorical" if name.endswith("softmax") else "ordinal"
                if (directory / "training.json").exists():
                    model = Jevlet.load(directory, "cuda")
                else:
                    model, _ = train(local, directory, bundle)
                info = json.loads((directory / "training.json").read_text())
                validation, _ = score_records(collect_records(model, requests_for(bundle, "validation", cfg), local))
                latency = benchmark(model, requests, cfg, devices=["cuda"])
                result = evaluate(local, directory, "gpu_compare", model, bundle, name, with_latency=False)
                result["latency"] = latency
                result["parameters"] = sum(p.numel() for p in model.parameters())
                write_json(ROOT / f"results/gpu_compare_{name}.json", result)
            else:
                if name in ENCODERS:
                    local["model"], local["model_revision"] = ENCODERS[name], "main"
                model, features, info = conventional(local, bundle, name, directory)
                local = copy.deepcopy(info["config"])
                val_records = records_for(model, features, requests_for(bundle, "validation", cfg))
                validation, _ = score_records(val_records)
                calibration = records_for(model, features, requests_for(bundle, "calibration", cfg))
                temperatures, details = fit_temperatures(calibration, "cuda")
                write_json(directory / "calibration.json", {"temperatures": temperatures, "fit": details})
                test = records_for(model, features, requests)
                latency = context_latency(model, features, requests, temperatures)
                result = write_evaluation(test, temperatures, local, "gpu_compare", name, bundle["sources"], latency)
                result["parameters"] = info["parameters"]
                write_json(ROOT / f"results/gpu_compare_{name}.json", result)
                del features
            summary["models"][name] = {"validation": validation, "raw": result["raw"],
                "calibrated": result["calibrated"], "temperatures": result["temperatures"], "latency": result["latency"],
                "training_seconds": info["training_seconds"], "peak_gpu_bytes": info["peak_gpu_bytes"],
                "parameters": result["parameters"], "epochs_completed": len(info["history"]),
                "model_revision": info["model_revision"], "checkpoint": str(directory)}
            write_json(summary_path, summary)
            print(f"DONE {name}: {json.dumps(result['calibrated'])}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        variants = [n for n in ("jevlet_ordinal", "jevlet_softmax") if n in summary["models"]]
        if len(variants) == 2:
            summary["selected_jevlet"] = max(variants, key=lambda n: summary["models"][n]["validation"]["score"]["accuracy"])
        summary["completed"] = all(n in summary["models"] for n in NAMES)
        write_json(summary_path, summary)
        write_json(BASE / "status.json", {"state": "completed" if summary["completed"] else "partial",
                   "completed": list(summary["models"]), "time": time.time()})
    except Exception as exc:
        write_json(BASE / "status.json", {"state": "failed", "model": name, "error": repr(exc), "time": time.time()})
        raise
    finally:
        stop.set()
        thread.join(timeout=12)
    print_report(summary, names, history)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    p.add_argument("--show-results", action="store_true",
                   help="Only display saved results; do not train, evaluate, or require CUDA")
    p.add_argument("--history", action="store_true", help="Also display recorded per-epoch training losses")
    args = p.parse_args()
    if args.show_results:
        summary_path = ROOT / "results/gpu_comparison_summary.json"
        if not summary_path.exists():
            p.error(f"No saved comparison results: {summary_path}")
        print_report(json.loads(summary_path.read_text()), args.models, args.history)
    else:
        run(args.models, args.history)
