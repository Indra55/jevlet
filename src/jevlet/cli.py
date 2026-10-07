"""Command-line workflows for local training, research, and prediction."""

import argparse
import json
import time
from pathlib import Path

from . import ROOT
from .config import load_config, run_dir, write_json


def parser():
    command = argparse.ArgumentParser(description="Jevlet: local typed decisions")
    command.add_argument("--config", default="config.yaml")
    sub = command.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    sub.add_parser("report")
    for name in ("train", "eval", "baselines", "ablate"):
        child = sub.add_parser(name)
        child.add_argument("--run", default="full")
        child.add_argument("--device", default=None)
        child.add_argument("--smoke", action="store_true")
        if name in ("baselines", "ablate"):
            child.add_argument("--seeds", nargs="+", type=int, default=None)
        if name == "baselines":
            child.add_argument("--only", choices=("tfidf", "separate", "flan"), default=None)
    child = sub.add_parser("predict")
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--request", required=True, type=Path, help="JSON file with text and questions")
    child.add_argument("--device", default="cpu")
    child.add_argument("--raw", action="store_true")
    return command


def main():
    args = parser().parse_args()
    if args.command == "report":
        from .report import generate_report
        generate_report()
        return
    if args.command == "predict":
        from .model import Jevlet
        from .config import seed_all
        prediction_cfg = load_config(args.config)
        seed_all(prediction_cfg["seed"], prediction_cfg["cpu_threads"])
        request = json.loads(args.request.read_text())
        model = Jevlet.load(args.checkpoint, args.device)
        print(json.dumps(model.predict(request["text"], request["questions"], not args.raw), indent=2))
        return
    smoke = args.command == "smoke" or args.smoke
    cfg = load_config(args.config, smoke=smoke)
    if getattr(args, "device", None):
        cfg["device"] = args.device
    if args.command == "smoke":
        from .eval import evaluate
        from .report import generate_report
        from .train import train
        cfg["device"] = "cpu"
        started = time.perf_counter()
        directory = run_dir("smoke")
        model, bundle = train(cfg, directory)
        evaluate(cfg, directory, "smoke", model, bundle)
        from .model import Jevlet
        reloaded = Jevlet.load(directory)
        request = {"text": "My card payment was charged twice.", "questions": [
            {"id": "intent", "type": "choice", "text": "What is the customer's intent?",
             "options": ["card arrival", "transaction charged twice"]},
            {"id": "duplicate", "type": "yesno", "text": "Is this about transaction charged twice?"},
            {"id": "arrival", "type": "yesno", "text": "Is this about card arrival?"}]}
        prediction = reloaded.predict(**request)
        write_json(ROOT / "results/smoke_demo.json", prediction)
        generate_report()
        seconds = time.perf_counter() - started
        write_json(ROOT / "results/smoke_status.json", {"passed": True, "seconds_including_asset_loading": seconds,
                   "device": "cpu", "note": "Pipeline check only; tiny training is not a quality benchmark."})
        print(f"SMOKE PASSED in {seconds:.1f}s (includes asset loading).", flush=True)
    elif args.command == "train":
        from .train import train
        train(cfg, run_dir(args.run))
    elif args.command == "eval":
        from .eval import evaluate
        from .report import generate_report
        directory = run_dir(args.run)
        saved_cfg = json.loads((directory / "training.json").read_text())["config"]
        saved_cfg["device"] = cfg["device"]
        evaluate(saved_cfg, directory, args.run)
        generate_report()
    else:
        from .experiment import run_experiments
        run_experiments(cfg, args.run, args.command, args.seeds or [cfg["seed"]], getattr(args, "only", None))


if __name__ == "__main__":
    main()
