"""Read existing progress files without interfering with the training process."""

import json
import time

from tqdm.auto import tqdm


def read_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def watch(directory, interval=2):
    if interval < 0.25:
        raise ValueError("Watch interval must be at least 0.25 seconds")
    print(f"Watching {directory}. Ctrl+C stops this monitor only.", flush=True)
    progress_path = directory / "progress.json"
    status_path = directory / "run_status.json"
    with tqdm(total=None, desc="Waiting for progress", bar_format="{desc}{postfix}",
              dynamic_ncols=True) as bar:
        while True:
            progress = read_state(progress_path)
            status = read_state(status_path)
            phase = progress.get("phase", "waiting")
            terminal_status = status.get("status")
            if terminal_status in ("complete", "failed", "interrupted"):
                bar.set_description_str(f"Run {terminal_status}", refresh=False)
                bar.set_postfix_str(status.get("error", status.get("reason", "")), refresh=False)
                bar.refresh()
                return
            if status.get("phase") == "evaluation":
                phase = "evaluation"
            if phase == "training":
                batches = progress["batches_per_epoch"]
                epoch = progress["epoch"]
                epochs = progress["max_epochs"]
                bar.total = batches
                bar.n = progress["batch"]
                bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}"
                bar.set_description_str(f"Epoch {epoch}/{epochs}", refresh=False)
                since_update = max(0, time.time() - progress_path.stat().st_mtime)
                bar.set_postfix_str(
                    f"loss={progress['mean_training_loss']:.4f}, "
                    f"total ETA={duration(progress['estimated_remaining_training_seconds'])}, "
                    f"updated={int(since_update)}s ago", refresh=False)
            else:
                bar.bar_format = "{desc}{postfix}"
                bar.set_description_str(phase.capitalize(), refresh=False)
                bar.set_postfix_str("", refresh=False)
            bar.refresh()
            time.sleep(interval)
