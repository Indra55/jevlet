"""Pinned NLU++ banking intent modules with isolated source-fold partitions."""

import hashlib
import json
import random
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from . import ROOT
from .data import normalized

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
DATASET = "PolyAI-LDN/task-specific-datasets/nlupp/banking"
SOURCE = f"https://github.com/PolyAI-LDN/task-specific-datasets/tree/{REVISION}/nlupp"
QUESTION = "Which intent components apply to this banking message?"
FOLDS = {"train": tuple(range(14)), "validation": (14, 15), "calibration": (16, 17), "test": (18, 19)}


def prepare_banking_bundle(base, cfg, archive=None, counts=None, legacy_per_domain=64):
    if not base.get("multilabel_names") or not any(r["domain"] == "multilabel" for r in base["splits"]["train"]):
        raise ValueError("Banking extension requires the trained GoEmotions checkpoint and its rehearsal manifest")
    archive = Path(archive) if archive else ROOT / ".cache" / f"nlupp-{REVISION}.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(f"https://codeload.github.com/PolyAI-LDN/task-specific-datasets/zip/{REVISION}", timeout=90) as response:
            temporary.write_bytes(response.read())
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        return bundle_from_archive(base, cfg, source, counts, legacy_per_domain)


def bundle_from_archive(base, cfg, source, counts=None, legacy_per_domain=64):
    prefix = f"task-specific-datasets-{REVISION}/nlupp/data/"
    ontology = json.loads(source.read(prefix + "ontology.json"))["intents"]
    names = sorted(name for name, info in ontology.items() if set(info["domain"]) & {"general", "banking"})
    lookup = {name: i for i, name in enumerate(names)}
    raw, hashes = {}, {}
    for split, folds in FOLDS.items():
        rows = []
        for fold in folds:
            filename = f"banking/fold{fold}.json"
            content = source.read(prefix + filename)
            hashes[filename] = hashlib.sha256(content).hexdigest()
            for index, row in enumerate(json.loads(content)):
                intents = row.get("intents", [])
                if any(intent not in lookup for intent in intents):
                    raise ValueError("NLU++ banking annotation contains an unknown intent module")
                identifier = f"nlupp/banking/fold{fold}/{index}"
                rows.append({"id": identifier, "domain": "banking_multilabel", "text": row["text"],
                             "labels": sorted(lookup[intent] for intent in intents), "source_ids": [identifier],
                             "data_kind": "human_annotated", "question": QUESTION})
        raw[split] = rows
    clean, blocked, removed = {}, set(), Counter()
    for split in ("test", "calibration", "validation", "train"):
        rows = []
        for row in raw[split]:
            key = normalized(row["text"])
            if not key or key in blocked:
                removed[split] += 1
                continue
            blocked.add(key)
            rows.append(row)
        random.Random(f"{cfg['seed']}:{split}:nlupp-v1").shuffle(rows)
        if counts:
            rows = rows[:counts[split]]
        if not rows:
            raise ValueError(f"No banking multi-label examples in {split}")
        for row in rows:
            row["candidate_mode"] = random.Random(row["id"]).choices(
                ("all", "partial", "none"), weights=(0.7, 0.15, 0.15))[0]
        clean[split] = rows
    banking_text_splits = {normalized(row["text"]): split for split, rows in clean.items() for row in rows}
    splits, rehearsal_removed = {}, Counter()
    for split, rows in clean.items():
        rehearsal = []
        for domain, limit in (("banking", legacy_per_domain), ("sst", legacy_per_domain),
                              ("multilabel", 256 if split == "train" else 128)):
            candidates = [r for r in base["splits"][split] if r["domain"] == domain]
            random.Random(f"{cfg['seed']}:{split}:{domain}:rehearsal").shuffle(candidates)
            for row in candidates[:limit]:
                owner = banking_text_splits.get(normalized(row["text"]))
                if owner is not None and owner != split:
                    rehearsal_removed[f"{split}/{domain}"] += 1
                    continue
                rehearsal.append(row)
        splits[split] = rehearsal + rows
    provenance = {"kind": "human_annotated", "id": DATASET, "revision": REVISION, "source": SOURCE,
                  "license": "CC-BY-4.0", "source_file_sha256": hashes,
                  "folds": {split: list(folds) for split, folds in FOLDS.items()},
                  "split_policy": "fixed disjoint source folds; text deduplication test/calibration/validation/train priority",
                  "removed_duplicates": dict(removed), "subset_counts": {s: len(r) for s, r in clean.items()},
                  "removed_cross_domain_rehearsal_duplicates": dict(rehearsal_removed),
                  "label_semantics": "NLU++ intent modules; distinct from Banking77 single-intent classes",
                  "evaluation": "single seeded split with supplied 2-10 candidates; not the paper's cross-validation protocol"}
    return {**base, "splits": splits, "banking_multilabel_names": names,
            "banking_multilabel_descriptions": {name: ontology[name]["description"] for name in names},
            "spec": {**base["spec"], "banking_multilabel": provenance},
            "sources": {**base["sources"], "banking_multilabel": provenance}}
