"""Proper scoring rules, held-out temperature fitting, and calibration plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from .config import HEADS, MULTILABEL_HEADS
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


def distribution(head, logits, temperature=1.0, score_mode="ordinal"):
    z = logits / temperature
    if head in MULTILABEL_HEADS:
        return z.sigmoid()
    if head == "choice" or (head == "score" and score_mode == "categorical"):
        return z.softmax(-1)
    if head == "score":
        return ordinal_probs(z)
    yes = z.sigmoid()
    return torch.stack((1 - yes, yes), dim=-1)


def fit_temperatures(records, device="cpu"):
    temperatures, details = {}, {}
    for head in HEADS + MULTILABEL_HEADS:
        selected = [r for r in records if r["type"] == head]
        if not selected:
            if head in MULTILABEL_HEADS:
                continue
            temperatures[head] = 1.0
            details[head] = {"status": "no_samples", "samples": 0}
            continue
        if head in MULTILABEL_HEADS:
            flat = [(z, y) for r in selected for z, y in zip(r["logits"], r["target"]) if y >= 0]
            z = torch.tensor([v[0] for v in flat], dtype=torch.float32)
            targets = torch.tensor([v[1] for v in flat], dtype=torch.float32)
        else:
            targets = torch.tensor([r["target"] for r in selected], dtype=torch.long)
        if head == "choice":
            kmax = max(len(r["logits"]) for r in selected)
            z = torch.full((len(selected), kmax), -1e4)
            for i, r in enumerate(selected):
                z[i, :len(r["logits"])] = torch.tensor(r["logits"])
        elif head not in MULTILABEL_HEADS:
            z = torch.tensor([r["logits"] for r in selected], dtype=torch.float32)
        z, targets = z.to(device), targets.to(device)
        score_mode = selected[0].get("score_mode", "ordinal")
        log_t = torch.zeros((), device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")

        def objective():
            # Bound the search for numerical stability; record any boundary solution.
            if head in MULTILABEL_HEADS:
                return torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.clamp(-4, 4).exp(), targets)
            p = distribution(head, z, log_t.clamp(-4, 4).exp(), score_mode)
            return -p[torch.arange(len(selected), device=device), targets].clamp_min(1e-12).log().mean()

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


def fit_multilabel_threshold(records, temperature, head="multilabel"):
    rows = [r for r in records if r["type"] == head]
    truth = [y for r in rows for y in r["target"] if y >= 0]
    probs = [p for r in rows for p, y in zip(distribution("multilabel", torch.tensor(r["logits"]),
                                                               temperature).tolist(), r["target"]) if y >= 0]
    if not truth or not any(truth):
        raise ValueError("Threshold tuning requires observed positive labels in validation data")
    choices = [(float(f1_score(truth, np.asarray(probs) >= t, zero_division=0)), -abs(t - 0.5), t)
               for t in np.linspace(0.05, 0.95, 91)]
    f1, _, threshold = max(choices)
    return float(threshold), {"split": "validation", "objective": "candidate_micro_f1", "f1": f1,
                              "samples": len(rows), "grid": "0.05 to 0.95 in steps of 0.01"}


def multilabel_metrics(rows, temperature=1.0, bins=15):
    truth, probabilities, predicted, exact, predictions, labels = [], [], [], [], [], {}
    covered = total = 0
    for row in rows:
        p = distribution("multilabel", torch.tensor(row["logits"]), temperature).numpy()
        y = np.asarray(row["target"])
        chosen = p >= row.get("threshold", 0.5)
        observed = y >= 0
        truth.extend(y[observed].tolist())
        probabilities.extend(p[observed].tolist())
        predicted.extend(chosen[observed].astype(int).tolist())
        if observed.all():
            exact.append(bool(np.array_equal(chosen, y)))
        for label, target, probability, decision in zip(row["option_labels"], y, p, chosen):
            if target >= 0:
                labels.setdefault(label, []).append((int(target), float(probability), int(decision)))
        if "gold_labels" in row:
            total += len(row["gold_labels"])
            covered += len(set(row["gold_labels"]) & set(row["option_labels"]))
        predictions.append({**row, "probabilities": p.tolist(), "predicted": chosen.astype(int).tolist()})
    y, p, decision = np.asarray(truth), np.asarray(probabilities), np.asarray(predicted)
    per_label = {}
    for label, values in sorted(labels.items()):
        target, _, output = np.asarray(values).T
        per_label[str(label)] = {"support": int(target.sum()), "observations": len(values),
            "precision": float(precision_score(target, output, zero_division=0)),
            "recall": float(recall_score(target, output, zero_division=0)),
            "f1": float(f1_score(target, output, zero_division=0))}
    result = {"samples": len(rows), "observed_label_decisions": len(y), "scope": "supplied_candidates",
              "micro_f1": float(f1_score(y, decision, zero_division=0)),
              "macro_f1": float(np.mean([v["f1"] for v in per_label.values()])),
              "precision": float(precision_score(y, decision, zero_division=0)),
              "recall": float(recall_score(y, decision, zero_division=0)),
              "exact_set_accuracy": float(np.mean(exact)) if exact else None,
              "complete_annotation_samples": len(exact),
              "hamming_loss": float(np.mean(y != decision)), "brier": float(np.mean((p - y) ** 2)),
              "nll": float(np.mean(-y * np.log(p.clip(1e-12, 1)) - (1-y) * np.log((1-p).clip(1e-12, 1)))),
              "ece": ece(p, y, bins), "per_label": per_label,
              "candidate_recall": covered / total if total else None}
    return result, predictions


def score_records(records, temperatures=None, bins=15, intent_count=77):
    temperatures = temperatures or {head: 1.0 for head in HEADS}
    metrics, predictions = {}, []
    for head in HEADS:
        rows = [r for r in records if r["type"] == head]
        if not rows:
            continue
        probs = [distribution(head, torch.tensor(r["logits"]), temperatures[head],
                              r.get("score_mode", "ordinal")).numpy() for r in rows]
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
            result["macro_f1"] = float(f1_score(truth, predicted, labels=list(range(5)),
                                               average="macro", zero_division=0))
            result["mae"] = float(np.abs(predicted - truth).mean())
            expected = np.array([np.dot(np.arange(5), p) for p in probs])
            result["expected_score_mae"] = float(np.abs(expected - truth).mean())
        if head == "yesno":
            result["auroc"] = float(roc_auc_score(truth, [p[1] for p in probs])) if len(set(truth)) == 2 else None
        metrics[head] = result
        for r, p, pred, conf in zip(rows, probs, predicted, confidence):
            predictions.append({**r, "probabilities": p.tolist(), "predicted": int(pred),
                                "confidence": float(conf)})
    for head in MULTILABEL_HEADS:
        multi = [r for r in records if r["type"] == head]
        if multi:
            metrics[head], multi_predictions = multilabel_metrics(multi, temperatures.get(head, 1.0), bins)
            predictions.extend(multi_predictions)
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
