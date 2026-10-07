"""Proper scoring rules, held-out temperature fitting, and calibration plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import HEADS
from .model import ordinal_probs


def ece(confidence, correct, bins=15):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("ECE bins must be a positive integer")
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    if not len(confidence) or len(confidence) != len(correct):
        raise ValueError("ECE requires equally sized, nonempty arrays")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be finite and between zero and one")
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(sum(np.mean(indices == b) * abs(confidence[indices == b].mean() -
                     correct[indices == b].mean()) for b in range(bins) if (indices == b).any()))


def distribution(head, logits, temperature=1.0):
    z = logits / temperature
    if head == "choice":
        return z.softmax(-1)
    if head == "score":
        return ordinal_probs(z)
    yes = z.sigmoid()
    return torch.stack((1 - yes, yes), dim=-1)


def fit_temperatures(records):
    temperatures, details = {}, {}
    for head in HEADS:
        selected = [r for r in records if r["type"] == head]
        if not selected:
            temperatures[head] = 1.0
            details[head] = {"status": "no_samples", "samples": 0}
            continue
        targets = torch.tensor([r["target"] for r in selected], dtype=torch.long)
        if head == "choice":
            kmax = max(len(r["logits"]) for r in selected)
            z = torch.full((len(selected), kmax), -1e4)
            for i, r in enumerate(selected):
                z[i, :len(r["logits"])] = torch.tensor(r["logits"])
        else:
            z = torch.tensor([r["logits"] for r in selected], dtype=torch.float32)
        log_t = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")

        def objective():
            # Bound the search for numerical stability; record any boundary solution.
            p = distribution(head, z, log_t.clamp(-4, 4).exp())
            return -p[torch.arange(len(selected)), targets].clamp_min(1e-12).log().mean()

        before = float(objective().detach())

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        after = float(objective().detach())
        fitted = float(log_t.detach().clamp(-4, 4).exp())
        if not np.isfinite(fitted) or not np.isfinite(after):
            raise RuntimeError(f"Temperature fitting failed for {head}")
        if after > before + 1e-6:
            fitted, after = 1.0, before
        temperatures[head] = fitted
        details[head] = {"status": "fitted", "samples": len(selected), "nll_before": before,
                         "nll_after": after, "at_search_boundary": abs(float(log_t.detach())) >= 4}
    return temperatures, details


def score_records(records, temperatures=None, bins=15, intent_count=77):
    temperatures = temperatures or {head: 1.0 for head in HEADS}
    metrics, predictions = {}, []
    for head in HEADS:
        rows = [r for r in records if r["type"] == head]
        if not rows:
            continue
        probs = [distribution(head, torch.tensor(r["logits"]), temperatures[head]).numpy() for r in rows]
        predicted = np.array([p.argmax() for p in probs])
        truth = np.array([r["target"] for r in rows])
        confidence = np.array([p.max() for p in probs])
        correct = predicted == truth
        result = {"samples": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
                  "ece": ece(confidence, correct, bins),
                  "nll": float(np.mean([-np.log(max(float(p[y]), 1e-12)) for p, y in zip(probs, truth)])),
                  "brier": float(np.mean([np.sum((p - np.eye(len(p))[y]) ** 2) for p, y in zip(probs, truth)]))}
        if head == "choice":
            gold = [r["option_labels"][int(y)] for r, y in zip(rows, truth)]
            pred_labels = [r["option_labels"][int(y)] for r, y in zip(rows, predicted)]
            result["macro_f1"] = float(f1_score(gold, pred_labels, labels=list(range(intent_count)),
                                               average="macro", zero_division=0))
        if head == "score":
            result["mae"] = float(np.abs(predicted - truth).mean())
            expected = np.array([np.dot(np.arange(5), p) for p in probs])
            result["expected_score_mae"] = float(np.abs(expected - truth).mean())
        if head == "yesno":
            result["auroc"] = float(roc_auc_score(truth, [p[1] for p in probs])) if len(set(truth)) == 2 else None
        metrics[head] = result
        for r, p, pred, conf in zip(rows, probs, predicted, confidence):
            predictions.append({**r, "probabilities": p.tolist(), "predicted": int(pred),
                                "confidence": float(conf)})
    return metrics, predictions


def risk_coverage(predictions, head):
    rows = [r for r in predictions if r["type"] == head]
    rows.sort(key=lambda r: -r["confidence"])
    if not rows:
        return [], []
    # Include tied confidence values together: each point corresponds to a threshold.
    errors = np.array([abs(r["predicted"] - r["target"]) / 4 if head == "score"
                       else float(r["predicted"] != r["target"]) for r in rows])
    ends = [i for i in range(len(rows)) if i == len(rows) - 1 or
            rows[i]["confidence"] != rows[i + 1]["confidence"]]
    return [(i + 1) / len(rows) for i in ends], [float(errors[:i + 1].mean()) for i in ends]


def make_figures(before, after, directory, bins=15):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for head in HEADS:
        if not any(r["type"] == head for r in after):
            continue
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        for title, predictions in (("raw", before), ("temperature scaled", after)):
            rows = [r for r in predictions if r["type"] == head]
            conf = np.array([r["confidence"] for r in rows])
            correct = np.array([r["target"] == r["predicted"] for r in rows])
            indices = np.minimum((conf * bins).astype(int), bins - 1)
            occupied = [b for b in range(bins) if (indices == b).any()]
            ax.plot([conf[indices == b].mean() for b in occupied],
                    [correct[indices == b].mean() for b in occupied], "o-", label=title)
        ax.set(xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1), title=head)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"{head}_reliability.pdf")
        fig.savefig(directory / f"{head}_reliability.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for title, predictions in (("raw", before), ("temperature scaled", after)):
            coverage, risk = risk_coverage(predictions, head)
            ax.plot(coverage, risk, label=title)
        ax.set(xlabel="Coverage", ylabel="Normalized MAE" if head == "score" else "Error rate", title=head)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"{head}_risk_coverage.pdf")
        fig.savefig(directory / f"{head}_risk_coverage.png", dpi=150)
        plt.close(fig)
