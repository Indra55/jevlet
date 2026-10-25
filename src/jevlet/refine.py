"""Validation-only selection of question-augmented CUDA refinement candidates."""
import copy
import gc
import json
import threading
import time

import torch

from . import ROOT
from .config import HEADS, load_config, write_json
from .data import requests_for
from .eval import benchmark, write_evaluation
from .gpu_compare import monitor, require_cuda
from .metrics import score_records
from .model import Jevlet
from .train import collect_records, train

BASE = ROOT / "checkpoints/jevlet_refinement"
INITIAL = ROOT / "checkpoints/gpu_comparison/jevlet_softmax"


def run():
    require_cuda()
    BASE.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((INITIAL / "data.json").read_text())
    cfg = load_config(ROOT / "config.yaml")
    cfg.update(device="cuda", request_seed=42, augment_questions=True,
               initial_checkpoint=str(INITIAL), checkpoint_metric="sentiment_accuracy")
    cfg["eval"].update(validation_views=5)
    cfg["train"].update(epochs=3, patience=2)
    source_sets = [{r['id'] for r in bundle['splits'][s]}
                   for s in ('train', 'validation', 'calibration', 'test')]
    assert all(not a.intersection(b) for i, a in enumerate(source_sets) for b in source_sets[i+1:])
    plan = [(42, 5e-6), (43, 1e-5), (44, 2e-5)]
    result_path = ROOT / "results/jevlet_refinement.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"candidates": {}}
    result.update(protocol={"validation_views": 5, "unique_validation_contexts": 1549,
        "validation_requests": 7745, "validation_decisions": 17735,
        "selection": "expanded-validation sentiment accuracy, with banking accuracy guardrails",
        "guardrail": "choice and binary accuracy no more than 1 percentage point below initial checkpoint",
        "initial_checkpoint": str(INITIAL), "plan": plan, "test_request_seed": 42,
        "limitations": "shared initial checkpoint; three refinement settings, not independent from-scratch runs; correlated validation views; previously inspected test"})
    stop = threading.Event()
    worker = threading.Thread(target=monitor, args=(stop, BASE / "vram.jsonl"), daemon=True)
    worker.start()
    try:
        validation = requests_for(bundle, "validation", cfg)
        if "initial_validation" not in result:
            initial = Jevlet.load(INITIAL, "cuda")
            result["initial_validation"], _ = score_records(collect_records(initial, validation, cfg))
            del initial
            gc.collect(); torch.cuda.empty_cache()
            write_json(result_path, result)
        for seed, lr in plan:
            name = f"seed{seed}_lr{lr:g}"
            if name in result["candidates"]:
                continue
            local = copy.deepcopy(cfg)
            local["seed"] = seed
            local["train"]["lr"] = lr
            directory = BASE / name
            write_json(BASE / "status.json", {"state": "training", "candidate": name, "time": time.time()})
            if (directory / "training.json").exists():
                model = Jevlet.load(directory, "cuda")
            else:
                model, _ = train(local, directory, bundle)
            metrics, _ = score_records(collect_records(model, validation, local))
            info = json.loads((directory / "training.json").read_text())
            result["candidates"][name] = {"validation": metrics, "checkpoint": str(directory),
                "training_seconds": info["training_seconds"], "peak_gpu_bytes": info["peak_gpu_bytes"],
                "epochs_completed": len(info["history"]), "seed": seed, "learning_rate": lr}
            write_json(result_path, result)
            print(f"CANDIDATE {name}: {json.dumps(metrics)}", flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        initial = result["initial_validation"]
        eligible = {name: item for name, item in result["candidates"].items()
                    if all(item["validation"][h]["accuracy"] >= initial[h]["accuracy"] - .01
                           for h in ("choice", "yesno"))}
        # Retain the initial checkpoint if refinement fails to improve validation.
        options = {"initial": {"validation": initial, "checkpoint": str(INITIAL)}, **eligible}
        winner = max(options, key=lambda n: options[n]["validation"]["score"]["accuracy"])
        result.update(selected=winner, selected_checkpoint=options[winner]["checkpoint"],
                      eligible_candidates=list(eligible))
        write_json(result_path, result)
        if not result.get("completed"):
            cfg["seed"] = 42 if winner == "initial" else result["candidates"][winner]["seed"]
            model = Jevlet.load(result["selected_checkpoint"], "cuda")
            # This is the only test evaluation in the search; all candidate ranking uses validation.
            requests = requests_for(bundle, "test", cfg)
            records = collect_records(model, requests, cfg)
            temperatures = dict(zip(HEADS, model.temperatures.cpu().tolist()))
            latency = benchmark(model, requests, cfg, devices=["cuda"])
            evaluation = write_evaluation(records, temperatures, cfg, "refinement", "jevlet",
                                          bundle["sources"], latency)
            result.update(test=evaluation, completed=True)
            write_json(result_path, result)
        write_json(BASE / "status.json", {"state": "completed", "selected": winner, "time": time.time()})
        print(f"SELECTED {winner}: {json.dumps(result['test']['calibrated'])}", flush=True)
    except BaseException as exc:
        write_json(BASE / "status.json", {"state": "failed", "error": repr(exc), "time": time.time()})
        raise
    finally:
        stop.set(); worker.join(timeout=15)


if __name__ == "__main__":
    run()
