"""Public source data, deterministic question sampling, and marker-safe packing."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from .config import write_json

TOKENS = ("[Q]", "[OPT]")
SCORE_QUESTION = "How positive is this review?"
CHOICE_QUESTION = "What is the customer's intent?"


def normalized(text):
    return " ".join(text.casefold().split())


def unique_rows(dataset, domain, split, blocked, stats):
    rows, seen = [], {}
    for i, row in enumerate(dataset):
        text, label = str(row["text"]), int(row["label"])
        key = normalized(text)
        if key in blocked:
            stats["removed_cross_split_duplicates"] += 1
            continue
        if key in seen:
            if seen[key] != label:
                raise ValueError(f"Conflicting duplicate labels in {domain}/{split}")
            stats["removed_within_split_duplicates"] += 1
            continue
        seen[key] = label
        rows.append({"id": f"{domain}/{split}/{i}", "domain": domain,
                     "text": text, "label": label})
    return rows


def stratified_partition(rows, fraction, seed):
    left, right = train_test_split(rows, test_size=fraction, random_state=seed,
                                   stratify=[row["label"] for row in rows])
    return list(left), list(right)


def prepare_data(cfg, directory):
    """Use HF's script-free Banking77 mirror; never execute dataset scripts."""
    path = Path(directory) / "data.json"
    spec = {k: cfg[k] for k in ("banking_dataset", "banking_revision",
                               "sst_dataset", "sst_revision", "is_smoke")}
    spec["split_seed"] = cfg["seed"]
    if cfg["is_smoke"]:
        spec.update({k: cfg[k] for k in ("train_per_dataset", "eval_per_dataset")})
    if path.exists():
        bundle = json.loads(path.read_text())
        if bundle["spec"] != spec:
            raise ValueError("Existing data manifest has different settings; use another run name")
        return bundle
    bank_path = Path(snapshot_download(cfg["banking_dataset"], repo_type="dataset",
                                      revision=cfg["banking_revision"], allow_patterns="*.parquet"))
    bank_files = {split: sorted(str(p) for p in bank_path.rglob("*.parquet")
                               if split in p.parts or p.name.startswith(split + "-"))
                  for split in ("train", "test")}
    if any(not files for files in bank_files.values()):
        raise ValueError("Banking77 revision must contain train/test Parquet files")
    bank = load_dataset("parquet", data_files=bank_files)
    names = bank["train"].features["label"].names
    sst_path = Path(snapshot_download(cfg["sst_dataset"], repo_type="dataset",
                                     revision=cfg["sst_revision"], allow_patterns="*.jsonl"))
    sst = load_dataset("json", data_files={"train": str(sst_path / "train.jsonl"),
                       "validation": str(sst_path / "dev.jsonl"), "test": str(sst_path / "test.jsonl")})
    stats = Counter()
    splits = {s: [] for s in ("train", "validation", "calibration", "test")}
    for domain, ds in (("banking", bank), ("sst", sst)):
        test = unique_rows(ds["test"], domain, "test", set(), stats)
        blocked = {normalized(row["text"]) for row in test}
        if domain == "banking":
            train = unique_rows(ds["train"], domain, "train", blocked, stats)
            train, held = stratified_partition(train, 0.2, cfg["seed"])
            val, cal = stratified_partition(held, 0.5, cfg["seed"])
        else:
            held = unique_rows(ds["validation"], domain, "dev", blocked, stats)
            blocked.update(normalized(row["text"]) for row in held)
            train = unique_rows(ds["train"], domain, "train", blocked, stats)
            val, cal = stratified_partition(held, 0.5, cfg["seed"])
        for split, rows in (("train", train), ("validation", val),
                            ("calibration", cal), ("test", test)):
            if cfg["is_smoke"]:
                rows = list(rows)
                random.Random(cfg["seed"]).shuffle(rows)
                rows = rows[:cfg["train_per_dataset"] if split == "train" else cfg["eval_per_dataset"]]
            splits[split].extend(rows)
    bundle = {"spec": spec, "label_names": names, "splits": splits,
              "sources": {"banking": {"id": cfg["banking_dataset"], "revision": bank_path.name},
                          "sst": {"id": cfg["sst_dataset"], "revision": sst_path.name},
                          "split_seed": cfg["seed"]},
              "deduplication": dict(stats)}
    write_json(path, bundle)
    return bundle


def pretty_label(name):
    return name.replace("_", " ").rstrip("?").lower()


def sample_request(row, names, cfg, epoch=0, training=False, fixed_k=None):
    digest = hashlib.sha256(f"{cfg['seed']}:{epoch}:{row['id']}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if row["domain"] == "sst":
        questions = [{"id": "sentiment", "type": "score", "text": SCORE_QUESTION,
                      "target": row["label"]}]
    else:
        gold = row["label"]
        k = fixed_k or rng.randint(cfg["options_min"], cfg["options_max"])
        candidates = [gold] + rng.sample([i for i in range(len(names)) if i != gold], k - 1)
        rng.shuffle(candidates)
        recipe = rng.choice(("choice", "yesno", "pair", "triple")) if training else "triple"
        questions = []
        if recipe != "yesno":
            questions.append({"id": "intent", "type": "choice", "text": CHOICE_QUESTION,
                              "options": [pretty_label(names[i]) for i in candidates],
                              "option_labels": candidates, "target": candidates.index(gold)})
        count = {"choice": 0, "yesno": 1, "pair": 1, "triple": 2}[recipe]
        for i in range(count):
            positive = rng.random() < 0.5
            pool = candidates if recipe != "yesno" else list(range(len(names)))
            queried = gold if positive else rng.choice([n for n in pool if n != gold])
            questions.append({"id": f"about_{i}", "type": "yesno",
                              "text": f"Is this about {pretty_label(names[queried])}?",
                              "query_label": queried, "target": int(positive)})
        if training:
            rng.shuffle(questions)
    return {"source_id": row["id"], "text": row["text"], "questions": questions}


def requests_for(bundle, split, cfg, epoch=0, training=False, fixed_k=None, head=None):
    requests = [sample_request(row, bundle["label_names"], cfg, epoch, training, fixed_k)
                for row in bundle["splits"][split]]
    if head:
        requests = [{**req, "questions": [q]} for req in requests
                    for q in req["questions"] if q["type"] == head]
    return requests


def validate_questions(questions):
    if not isinstance(questions, list) or not 1 <= len(questions) <= 3:
        raise ValueError("Supply a list of one to three questions")
    ids = set()
    for q in questions:
        if q.get("type") not in ("choice", "score", "yesno"):
            raise ValueError("Question type must be choice, score, or yesno")
        if not isinstance(q.get("text"), str) or not q["text"].strip():
            raise ValueError("Each question needs nonempty text")
        if not isinstance(q.get("id"), str) or not q["id"] or q["id"] in ids:
            raise ValueError("Question IDs must be unique nonempty strings")
        ids.add(q["id"])
        if q["type"] == "choice":
            opts = q.get("options")
            if not isinstance(opts, list) or not 2 <= len(opts) <= 10:
                raise ValueError("Choice questions require two to ten options")
            if any(not isinstance(o, str) or not o.strip() for o in opts) or len(set(opts)) != len(opts):
                raise ValueError("Options must be distinct nonempty strings")


def pack_request(tokenizer, request, max_length):
    validate_questions(request["questions"])
    if not isinstance(request["text"], str) or not request["text"].strip():
        raise ValueError("Context must be nonempty text")
    reserved = tuple(tokenizer.all_special_tokens) + TOKENS

    def encode(text):
        # Literal marker text in input cannot create extra model markers.
        for token in reserved:
            text = text.replace(token, " ")
        return tokenizer.encode(text, add_special_tokens=False)

    ids, positions = [tokenizer.cls_token_id], []
    for question in request["questions"]:
        qpos = len(ids)
        ids.append(tokenizer.convert_tokens_to_ids("[Q]"))
        question_ids = encode(question["text"])
        if not question_ids:
            raise ValueError("Question has no tokens after removing reserved markers")
        ids.extend(question_ids)
        options = []
        for option in question.get("options", []) if question["type"] == "choice" else []:
            options.append(len(ids))
            ids.append(tokenizer.convert_tokens_to_ids("[OPT]"))
            option_ids = encode(option)
            if not option_ids:
                raise ValueError("Option has no tokens after removing reserved markers")
            ids.extend(option_ids)
        positions.append((qpos, options))
    ids.append(tokenizer.sep_token_id)
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("Question block leaves no context room; shorten questions/options or increase max_length")
    context = encode(request["text"])
    if not context:
        raise ValueError("Context has no tokens after removing reserved markers")
    ids.extend(context[:room])
    ids.append(tokenizer.sep_token_id)
    return {"input_ids": ids, "positions": positions, "request": request,
            "context_tokens_dropped": max(0, len(context) - room)}


def collate(items, pad_id):
    batch, length = len(items), max(len(item["input_ids"]) for item in items)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    qpos = torch.zeros((batch, 3), dtype=torch.long)
    opos = torch.zeros((batch, 3, 10), dtype=torch.long)
    omask = torch.zeros_like(opos, dtype=torch.bool)
    for b, item in enumerate(items):
        seq = item["input_ids"]
        ids[b, :len(seq)] = torch.tensor(seq)
        attention[b, :len(seq)] = 1
        for q, (position, options) in enumerate(item["positions"]):
            qpos[b, q] = position
            if options:
                opos[b, q, :len(options)] = torch.tensor(options)
                omask[b, q, :len(options)] = True
    return {"input_ids": ids, "attention_mask": attention, "qpos": qpos,
            "opos": opos, "omask": omask, "items": items}


def packed_items(tokenizer, requests, max_length):
    return [pack_request(tokenizer, req, max_length) for req in requests]
