"""Candidate-based multi-label annotations, GoEmotions data, and baseline."""

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .config import write_json
from .data import normalized, pretty_label

QUESTION = "Which of these intents apply to the customer's message?"
SPLITS = ("train", "validation", "calibration", "test")


def sample_multilabel_request(row, names, cfg, epoch=0, fixed_k=None):
    rng = random.Random(hashlib.sha256(f"{cfg.get('request_seed', cfg['seed'])}:{epoch}:{row['id']}".encode()).digest())
    gold = set(row["labels"])
    observed = set(row.get("observed_labels", range(len(names))))
    positives = sorted(gold)
    mode = row.get("candidate_mode", "all")
    if mode == "none":
        positives = []
    elif mode == "partial":
        positives = positives[:1]
    if len(positives) > cfg["options_max"]:
        raise ValueError("Positive labels exceed the configured candidate limit")
    k = fixed_k or rng.randint(max(cfg["options_min"], len(positives)), cfg["options_max"])
    if len(positives) > k:
        raise ValueError("Positive labels exceed the candidate limit; increase it or use explicit shortlist evaluation")
    negatives = [i for i in range(len(names)) if i not in gold]
    candidates = positives + rng.sample(negatives, min(k - len(positives), len(negatives)))
    if len(candidates) < 2:
        raise ValueError("Multi-label data requires at least two available candidates")
    rng.shuffle(candidates)
    target = [int(i in gold) if i in observed else -1 for i in candidates]
    if all(t == -1 for t in target):
        known = sorted(observed - gold)
        if not known:
            raise ValueError("Candidate set has no observed annotations")
        candidates[0] = rng.choice(known)
        target[0] = 0
    return {"source_id": row["id"], "text": row["text"], "questions": [{
        "id": "labels", "type": row["domain"], "text": row.get("question", QUESTION),
        "options": [pretty_label(names[i]) for i in candidates], "option_labels": candidates,
        "target": target, "gold_labels": sorted(gold), "data_kind": row["data_kind"]}]}


def reviewed_rows(path, names, base):
    """Import explicit split assignments and complete or partially observed annotations."""
    lookup = {name: i for i, name in enumerate(names)}
    splits = {split: [] for split in SPLITS}
    text_splits, source_splits, ids = {}, {}, set()
    for split, rows in base["splits"].items():
        for row in rows:
            text_splits[normalized(row["text"])] = split
            source_splits[row["id"]] = split
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split, text, labels = row.get("split"), row.get("text"), row.get("labels")
        identifier = row.get("id")
        if split not in SPLITS or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Line {line_number}: supply a valid split and nonempty text")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"Line {line_number}: IDs must be unique nonempty strings")
        if (not isinstance(labels, list) or len(labels) != len(set(labels)) or
                any(label not in lookup for label in labels)):
            raise ValueError(f"Line {line_number}: labels must be unique Banking77 catalogue names")
        if type(row.get("annotation_complete")) is not bool:
            raise ValueError(f"Line {line_number}: annotation_complete must be explicitly true or false")
        negatives = row.get("negative_labels", [])
        if (not isinstance(negatives, list) or len(set(negatives)) != len(negatives) or
                any(label not in lookup or label in labels for label in negatives)):
            raise ValueError(f"Line {line_number}: negative_labels must be distinct known labels disjoint from positives")
        key = normalized(text)
        if key in text_splits and text_splits[key] != split:
            raise ValueError(f"Line {line_number}: text overlaps another split")
        sources = row.get("source_ids", [identifier])
        if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources):
            raise ValueError(f"Line {line_number}: source_ids must be nonempty strings")
        for source in sources:
            if source in source_splits and source_splits[source] != split:
                raise ValueError(f"Line {line_number}: source ID overlaps another split")
            source_splits[source] = split
        text_splits[key] = split
        ids.add(identifier)
        observed = list(range(len(names))) if row["annotation_complete"] else [lookup[s] for s in labels + negatives]
        if not observed:
            raise ValueError(f"Line {line_number}: provide at least one observed label")
        splits[split].append({"id": identifier, "domain": "multilabel", "text": text.strip(),
                              "labels": [lookup[s] for s in labels], "observed_labels": observed,
                              "source_ids": sources, "data_kind": "reviewed"})
    if any(not rows for rows in splits.values()):
        raise ValueError("Reviewed JSONL must provide train, validation, calibration, and test rows")
    return splits


def prepare_bundle(base, cfg, data=None, go_emotions=False, counts=None, legacy_per_domain=64):
    if bool(data) == bool(go_emotions):
        raise ValueError("Choose exactly one of reviewed --data or --go-emotions")
    counts = counts or {"train": 4096, "validation": 384, "calibration": 384, "test": 768}
    names = base["label_names"]
    if data:
        extra = reviewed_rows(data, names, base)
        provenance = {"kind": "reviewed", "sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}
    else:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        revision = "add492243ff905527e67aeb8b80c082af02207c3"
        path = Path(snapshot_download("google-research-datasets/go_emotions", repo_type="dataset",
                                      revision=revision, allow_patterns="simplified/*.parquet"))
        files = {split: [str(p) for p in (path / "simplified").glob(split + "-*.parquet")]
                 for split in ("train", "validation", "test")}
        dataset = load_dataset("parquet", data_files=files)
        names = dataset["train"].features["labels"].feature.names
        clean, blocked, seen_ids = {}, set(), set()
        removed = defaultdict(int)
        for split in ("test", "validation", "train"):
            rows = []
            for raw in dataset[split]:
                key = normalized(raw["text"])
                if not key or key in blocked or raw["id"] in seen_ids:
                    removed[split] += 1
                    continue
                blocked.add(key)
                seen_ids.add(raw["id"])
                rows.append({"id": "goemotions/" + raw["id"], "domain": "multilabel", "text": raw["text"],
                             "labels": list(raw["labels"]), "source_ids": [raw["id"]],
                             "data_kind": "human_annotated", "question": "Which emotions are expressed in this text?"})
            random.Random(f"{cfg['seed']}:{split}:goemotions-v1").shuffle(rows)
            clean[split] = rows
        development = clean.pop("validation")
        clean["validation"], clean["calibration"] = development[::2], development[1::2]
        extra = {}
        for split in SPLITS:
            if counts[split] > len(clean[split]):
                raise ValueError(f"Requested {counts[split]} examples but only {len(clean[split])} are available for {split}")
            rows = clean[split][:counts[split]]
            for row in rows:
                row["candidate_mode"] = random.Random(row["id"]).choices(
                    ("all", "partial", "none"), weights=(0.7, 0.15, 0.15))[0]
            extra[split] = rows
        provenance = {"kind": "human_annotated", "id": "google-research-datasets/go_emotions",
                      "subset": "simplified", "revision": revision, "license": "Apache-2.0",
                      "source": "https://github.com/google-research/google-research/tree/master/goemotions",
                      "split_policy": "official train/test; deduplicate text and IDs test-first; split official dev alternately after seeded shuffle",
                      "removed_duplicates": dict(removed), "subset_counts": {s: len(r) for s, r in extra.items()},
                      "evaluation": "sampled 2-10 supplied emotion candidates; includes missing-positive candidate cases"}
    splits = {}
    for split in SPLITS:
        legacy = []
        for domain in ("banking", "sst"):
            rows = [r for r in base["splits"][split] if r["domain"] == domain]
            random.Random(f"{cfg['seed']}:{split}:{domain}").shuffle(rows)
            legacy.extend(rows[:legacy_per_domain])
        splits[split] = legacy + extra[split]
    return {**base, "splits": splits, "multilabel_names": names,
            "spec": {**base["spec"], "multilabel": provenance},
            "sources": {**base["sources"], "multilabel": provenance}}


def baseline(bundle, cfg, directory, head="multilabel"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from .data import requests_for
    from .metrics import fit_temperatures, fit_multilabel_threshold, score_records

    rows = [r for r in bundle["splits"]["train"] if r["domain"] == head]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    features = vectorizer.fit_transform([r["text"] for r in rows])
    classifiers = []
    label_names = bundle.get(head + "_names", bundle["label_names"])
    for label in range(len(label_names)):
        observed = [i for i, r in enumerate(rows) if label in r.get("observed_labels", range(len(label_names)))]
        y = [int(label in rows[i]["labels"]) for i in observed]
        if len(set(y)) < 2:
            prior = (sum(y) + 1) / (len(y) + 2)
            classifiers.append(math.log(prior / (1 - prior)))
        else:
            classifiers.append(LogisticRegression(max_iter=300, random_state=cfg["seed"]).fit(features[observed], y))

    def records(split):
        result = []
        for req in requests_for(bundle, split, cfg, head=head):
            q = req["questions"][0]
            x = vectorizer.transform([req["text"]])
            logits = [float(classifiers[i].decision_function(x)[0]) if hasattr(classifiers[i], "decision_function")
                      else classifiers[i] for i in q["option_labels"]]
            result.append({**q, "source_id": req["source_id"], "logits": logits})
        return result

    temperatures, fit = fit_temperatures(records("calibration"))
    threshold, tuning = fit_multilabel_threshold(records("validation"), temperatures[head], head)
    test = [{**r, "threshold": threshold} for r in records("test")]
    scores, predictions = score_records(test, temperatures)
    result = {"model": "tfidf_logistic", "data": bundle["sources"][head],
              "metrics": scores[head], "threshold": threshold, "threshold_fit": tuning,
              "calibration": fit[head], "predictions": predictions}
    write_json(directory / (head + "_baseline.json"), result)
    return result
