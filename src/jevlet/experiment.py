"""Explicit, reusable baseline and ablation stages."""

import copy
import json

from .baselines import flan, separate, tfidf
from .config import HEADS, run_dir, seed_all, write_json
from .data import packed_items, prepare_data, requests_for
from .eval import evaluate, unpack_items, write_evaluation
from .model import Jevlet
from .report import generate_report
from .train import collect_records, loader, on_device, train


def single_records(model, requests, cfg):
    import torch
    items = unpack_items(model.tokenizer, packed_items(model.tokenizer, requests, model.max_length), model.max_length)
    records = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader(items, model, cfg):
            batch = on_device(batch, device)
            outputs = {k: v.cpu() for k, v in model(batch).items()}
            for i, item in enumerate(batch["items"]):
                question = item["request"]["questions"][0]
                logits = outputs[question["type"]][i, 0]
                if question["type"] == "choice":
                    logits = logits[:len(question["options"])]
                records.append({**question, "source_id": item["request"]["source_id"],
                                "logits": logits.tolist(), "context_tokens_dropped": item["context_tokens_dropped"]})
    return records


def run_experiments(cfg, base_run, stage, seeds, only=None):
    base_directory = run_dir(base_run)
    base_directory.mkdir(parents=True, exist_ok=True)
    bundle = prepare_data(cfg, base_directory)
    for seed in seeds:
        current = copy.deepcopy(cfg)
        current["seed"] = seed
        run = base_run if seed == cfg["seed"] else f"{base_run}-seed{seed}"
        directory = run_dir(run)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "data.json", bundle)
        seed_all(seed, cfg["cpu_threads"])
        if (directory / "training.json").exists():
            saved = json.loads((directory / "training.json").read_text())
            current = saved["config"]
            current["device"] = cfg["device"]
            model = Jevlet.load(directory, current["device"])
        else:
            model, _ = train(current, directory, bundle)
        # All experiments use the exact pretrained encoder revision of the main run.
        current["model_revision"] = model.source_revision or current["model_revision"]
        evaluate(current, directory, run, model, bundle)
        if stage == "baselines":
            if only in (None, "tfidf"):
                tfidf(current, directory, bundle, run)
            if only in (None, "separate"):
                separate(current, directory, bundle, run)
            if only in (None, "flan"):
                flan(current, directory, bundle, run, model)
        else:
            temperatures = dict(zip(HEADS, model.temperatures.cpu().tolist()))
            requests = requests_for(bundle, "test", current)
            write_evaluation(single_records(model, requests, current), temperatures, current, run,
                             "single_question", bundle["sources"])
            for k in (2, 5, 10):
                # Smoke uses a 128-token budget; larger candidate sets get the full budget.
                original_length = model.max_length
                model.max_length = max(original_length, 256) if k > 3 else original_length
                records = collect_records(model, requests_for(bundle, "test", current, fixed_k=k), current)
                option_cfg = copy.deepcopy(current)
                option_cfg["max_length"] = model.max_length
                option_cfg["evaluation_options"] = k
                write_evaluation(records, temperatures, option_cfg, run, f"options{k}", bundle["sources"])
                model.max_length = original_length
            no_brier_cfg = copy.deepcopy(current)
            no_brier_cfg["train"]["brier_weight"] = 0.0
            target = directory / "no_brier"
            if (target / "training.json").exists():
                no_brier = Jevlet.load(target, current["device"])
            else:
                no_brier, _ = train(no_brier_cfg, target, bundle)
            evaluate(no_brier_cfg, target, run, no_brier, bundle, "no_brier", with_latency=False)
    generate_report()
