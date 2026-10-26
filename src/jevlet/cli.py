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
    child = sub.add_parser("multilabel", help="Train and evaluate a candidate-based multi-label extension")
    child.add_argument("--checkpoint", type=Path, default=None)
    child.add_argument("--run", default="multilabel-emotions")
    child.add_argument("--device", default="cpu")
    source = child.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Reviewed Banking77 multi-label JSONL with explicit splits")
    source.add_argument("--go-emotions", action="store_true", help="Use real human-annotated GoEmotions data")
    source.add_argument("--nlupp", action="store_true", help="Add NLU++ banking multi-label prediction to an emotion checkpoint")
    child.add_argument("--nlupp-archive", type=Path, default=None, help="Optional cached official source archive")
    child.add_argument("--epochs", type=int, default=3)
    child.add_argument("--train-samples", type=int, default=4096)
    child.add_argument("--eval-samples", type=int, default=384)
    child.add_argument("--legacy-per-domain", type=int, default=64)
    child = sub.add_parser("serve", help="Launch the local case-study dashboard")
    child.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/multilabel-banking")
    child.add_argument("--device", default="cpu")
    child.add_argument("--host", default="127.0.0.1")
    child.add_argument("--port", type=int, default=8000)
    child = sub.add_parser("watch", help="Monitor an existing run without restarting it")
    child.add_argument("--run", default="full")
    child.add_argument("--interval", type=float, default=2)
    for name in ("run", "train", "eval", "baselines", "ablate"):
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
    if args.command == "multilabel":
        from .multilabel import prepare_bundle, baseline
        from .train import train
        from .eval import evaluate
        if args.nlupp and args.run == "multilabel-emotions":
            args.run = "multilabel-banking"
        directory = run_dir(args.run)
        if directory.exists():
            raise SystemExit("Use a new run name; multi-label training does not overwrite existing runs")
        if min(args.epochs, args.train_samples, args.eval_samples, args.legacy_per_domain) < 1:
            raise SystemExit("Epoch and sample counts must be positive")
        checkpoint = (args.checkpoint or ROOT / "checkpoints" / ("multilabel-emotions" if args.nlupp else "cpu-full")).resolve()
        cfg = json.loads((checkpoint / "training.json").read_text())["config"]
        cfg.update(device=args.device, initial_checkpoint=str(checkpoint), multilabel=True,
                   is_smoke=False)
        cfg["train"].update(epochs=args.epochs, patience=args.epochs)
        base = json.loads((checkpoint / "data.json").read_text())
        counts = {"train": args.train_samples, "validation": args.eval_samples,
                  "calibration": args.eval_samples, "test": args.eval_samples * 2}
        head = "banking_multilabel" if args.nlupp else "multilabel"
        if args.nlupp:
            from .banking_multilabel import prepare_banking_bundle
            bundle = prepare_banking_bundle(base, cfg, args.nlupp_archive, counts, args.legacy_per_domain)
            cfg["banking_multilabel"] = True
        else:
            bundle = prepare_bundle(base, cfg, args.data, args.go_emotions, counts, args.legacy_per_domain)
        cfg[head + "_data"] = bundle["sources"][head]
        status = {"status": "running", "phase": "training", "device": args.device,
                  "data": bundle["sources"][head]}
        write_json(directory / "run_status.json", status)
        started = time.perf_counter()
        try:
            model, bundle = train(cfg, directory, bundle)
            status["phase"] = "evaluation"
            write_json(directory / "run_status.json", status)
            result = evaluate(cfg, directory, args.run, model, bundle, with_latency=False)
            comparison = baseline(bundle, cfg, directory, head)
            write_json(directory / (head + "_results.json"), {
                "data": bundle["sources"][head], "model": result["calibrated"][head],
                "baseline": comparison["metrics"], "threshold": getattr(model, head + "_threshold"),
                "scope": "supplied candidates; empty selection does not establish out-of-domain detection"})
            status.update(status="complete", phase="complete", seconds=time.perf_counter() - started)
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=str(error), seconds=time.perf_counter() - started)
            raise
        finally:
            write_json(directory / "run_status.json", status)
        print(f"Multi-label run complete: {directory}", flush=True)
        return
    if args.command == "serve":
        try:
            import uvicorn
            from .server import create_app
        except ImportError as error:
            raise SystemExit('Dashboard dependencies missing. Install with: uv pip install -e ".[demo]"') from error
        cfg = load_config(args.config)
        uvicorn.run(create_app(args.checkpoint, args.device, cfg["cpu_threads"]),
                    host=args.host, port=args.port)
        return
    if args.command == "watch":
        from .progress import watch
        try:
            watch(run_dir(args.run), args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring; the training process is unaffected.", flush=True)
        return
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
    elif args.command == "run":
        from .eval import evaluate
        from .report import generate_report
        from .train import train
        directory = run_dir(args.run)
        started = time.perf_counter()
        status = {"status": "running", "phase": "training", "device": cfg["device"]}
        write_json(directory / "run_status.json", status)
        try:
            model, bundle = train(cfg, directory)
            status["phase"] = "evaluation"
            write_json(directory / "run_status.json", status)
            evaluate(cfg, directory, args.run, model, bundle)
            generate_report()
            status.update(status="complete", phase="complete", seconds=time.perf_counter() - started)
            write_json(directory / "run_status.json", status)
        except Exception as error:
            status.update(status="failed", error=str(error), seconds=time.perf_counter() - started)
            write_json(directory / "run_status.json", status)
            raise
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
