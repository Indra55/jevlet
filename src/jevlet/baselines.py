"""Local TF-IDF, separate encoders, and FLAN answer-likelihood baselines."""

import json
import time

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import HEADS, device_for, write_json
from .data import requests_for
from .eval import unpack_items, write_evaluation
from .metrics import fit_temperatures
from .model import Jevlet
from .train import collect_records, train


def probabilities_to_ordinal(p):
    cumulative = np.array([sum(p[i + 1:]) for i in range(4)])
    cumulative = cumulative.clip(1e-7, 1 - 1e-7)
    return (np.log(cumulative) - np.log1p(-cumulative)).tolist()


def likelihood_logits(probabilities, question):
    p = np.asarray(probabilities)
    if question["type"] == "score":
        return probabilities_to_ordinal(p)
    if question["type"] == "yesno":
        return float(np.log(max(p[1], 1e-12)) - np.log(max(p[0], 1e-12)))
    return np.log(p.clip(1e-12)).tolist()


def full_class_probs(classifier, matrix, count):
    output = np.full((matrix.shape[0], count), 1e-12)
    output[:, classifier.classes_.astype(int)] = classifier.predict_proba(matrix)
    return output / output.sum(axis=1, keepdims=True)


class TfidfBaseline:
    def fit(self, bundle, cfg):
        rows = bundle["splits"]["train"]
        bank = [r for r in rows if r["domain"] == "banking"]
        sst = [r for r in rows if r["domain"] == "sst"]
        self.bank_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        self.sst_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        bank_matrix = self.bank_vectorizer.fit_transform([r["text"] for r in bank])
        sst_matrix = self.sst_vectorizer.fit_transform([r["text"] for r in sst])
        self.intent = LogisticRegression(max_iter=1000, random_state=cfg["seed"]).fit(bank_matrix, [r["label"] for r in bank])
        self.sentiment = LogisticRegression(max_iter=1000, random_state=cfg["seed"]).fit(sst_matrix, [r["label"] for r in sst])
        self.intent_count = len(bundle["label_names"])
        self.binary, self.binary_priors = {}, {}
        bank_requests = [r for r in requests_for(bundle, "train", cfg) if len(r["questions"]) == 3]
        indices = {r["id"]: i for i, r in enumerate(bank)}
        for label in range(self.intent_count):
            pairs = [(indices[r["source_id"]], q["target"]) for r in bank_requests
                     for q in r["questions"] if q["type"] == "yesno" and q["query_label"] == label]
            targets = [target for _, target in pairs]
            self.binary_priors[label] = (sum(targets) + 1) / (len(targets) + 2)
            if len(set(targets)) == 2:
                self.binary[label] = LogisticRegression(max_iter=1000, random_state=cfg["seed"]).fit(
                    bank_matrix[[index for index, _ in pairs]], targets)
        return self

    def records(self, requests):
        rows = []
        # Transform each shared context once, then answer all its questions.
        bank_matrix = self.bank_vectorizer.transform([r["text"] for r in requests])
        sst_matrix = self.sst_vectorizer.transform([r["text"] for r in requests])
        intents = full_class_probs(self.intent, bank_matrix, self.intent_count)
        sentiments = full_class_probs(self.sentiment, sst_matrix, 5)
        for i, request in enumerate(requests):
            for question in request["questions"]:
                head = question["type"]
                if head == "choice":
                    p = intents[i, question["option_labels"]]
                    p = p / p.sum()
                elif head == "score":
                    p = sentiments[i]
                else:
                    label = question["query_label"]
                    positive = (float(self.binary[label].predict_proba(bank_matrix[i])[0, 1])
                                if label in self.binary else self.binary_priors[label])
                    p = [1 - positive, positive]
                rows.append({**question, "source_id": request["source_id"],
                             "logits": likelihood_logits(p, question)})
        return rows


def timing(function, requests, cfg, device="cpu"):
    """End-to-end baseline timing over banking requests, without asset loading."""
    from itertools import cycle, islice
    requests = [r for r in requests if len(r["questions"]) == 3]
    measurements = []
    cuda = str(device).startswith("cuda")
    for size in (1, 32):
        selected = list(islice(cycle(requests), size))
        for _ in range(cfg["eval"]["latency_warmup"]):
            function(selected)
        durations = []
        for _ in range(cfg["eval"]["latency_repeats"]):
            if cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            function(selected)
            if cuda:
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000 / size)
        measurements.append({"device": str(device), "batch_size": size,
                             "p50_pack_and_forward_ms_per_request": float(np.median(durations)),
                             "p95_pack_and_forward_ms_per_request": float(np.percentile(durations, 95)),
                             "repeats": len(durations), "questions_per_request": 3})
    return {"measurements": measurements, "scope": "preprocessing and scoring; excludes loading and JSON"}


def tfidf(cfg, directory, bundle, run):
    path = directory / "tfidf.joblib"
    if path.exists():
        baseline = joblib.load(path)
    else:
        baseline = TfidfBaseline().fit(bundle, cfg)
        joblib.dump(baseline, path)
    temperatures, details = fit_temperatures(baseline.records(requests_for(bundle, "calibration", cfg)))
    requests = requests_for(bundle, "test", cfg)
    write_json(directory / "tfidf_calibration.json", {"temperatures": temperatures, "fit": details,
               "binary_models_fitted": len(baseline.binary), "binary_prior_only_labels":
               sorted(set(baseline.binary_priors) - set(baseline.binary))})
    latency = timing(baseline.records, requests, cfg)
    return write_evaluation(baseline.records(requests), temperatures, cfg, run, "tfidf", bundle["sources"], latency)


def separate(cfg, directory, bundle, run):
    models = {}
    for head in HEADS:
        target = directory / f"separate_{head}"
        if (target / "training.json").exists():
            models[head] = Jevlet.load(target, cfg["device"])
        else:
            models[head], _ = train(cfg, target, bundle, head=head)
    requests = requests_for(bundle, "test", cfg)
    from .data import packed_items
    matched_items = unpack_items(models["choice"].tokenizer,
        packed_items(models["choice"].tokenizer, requests, models["choice"].max_length), models["choice"].max_length)
    records = []
    for head in HEADS:
        items = [item for item in matched_items if item["request"]["questions"][0]["type"] == head]
        records.extend(collect_records(models[head], [], cfg, items=items))
    temperatures = {head: float(models[head].temperatures[HEADS.index(head)]) for head in HEADS}
    latency = {"measurements": []}
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        for model in models.values():
            model.to(device)

        def infer(selected):
            # One pass per question position, using the corresponding task model.
            rows = []
            items = unpack_items(models["choice"].tokenizer,
                packed_items(models["choice"].tokenizer, selected, models["choice"].max_length),
                models["choice"].max_length)
            for i in range(3):
                group = items[i::3]
                head = group[0]["request"]["questions"][0]["type"]
                rows.extend(collect_records(models[head], [], cfg, items=group))
            return rows

        measurement = timing(infer, requests, cfg, device)
        for row in measurement["measurements"]:
            row["question_passes"] = 3
            row["forward_calls"] = 3 * int(np.ceil(row["batch_size"] / cfg["eval"]["batch_size"]))
        latency["measurements"].extend(measurement["measurements"])
    return write_evaluation(records, temperatures, cfg, run, "separate", bundle["sources"], latency)


class FlanBaseline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["eval"]["flan_model"],
                                                       revision=cfg["eval"]["flan_revision"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg["eval"]["flan_model"],
                      revision=cfg["eval"]["flan_revision"]).to(device_for(cfg["device"])).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    @torch.inference_mode()
    def records(self, requests):
        rows = []
        device = next(self.model.parameters()).device
        levels = ["very negative", "negative", "neutral", "positive", "very positive"]
        for request in requests:
            for question in request["questions"]:
                head = question["type"]
                answers = question["options"] if head == "choice" else levels if head == "score" else ["no", "yes"]
                prefix = f"Answer using one of: {'; '.join(answers)}.\nQuestion: {question['text']}\nText: "
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                room = self.cfg["max_length"] - len(prefix_ids) - 1
                if room < 1:
                    raise ValueError("FLAN question block exceeds max_length")
                context = self.tokenizer.encode(request["text"], add_special_tokens=False)
                source = torch.tensor([prefix_ids + context[:room] + [self.tokenizer.eos_token_id]], device=device)
                targets = self.tokenizer(answers, return_tensors="pt", padding=True).to(device)
                labels = targets.input_ids.masked_fill(~targets.attention_mask.bool(), -100)
                output = self.model(input_ids=source.expand(len(answers), -1), labels=labels).logits.float()
                log_probs = output.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                mask = labels != -100
                likelihoods = (log_probs * mask).sum(-1) / mask.sum(-1)
                p = likelihoods.softmax(-1).cpu().numpy()
                rows.append({**question, "source_id": request["source_id"],
                             "logits": likelihood_logits(p, question),
                             "context_tokens_dropped": max(0, len(context) - room)})
        return rows


def subset_requests(bundle, split, cfg, limit):
    requests = requests_for(bundle, split, cfg)
    # A fixed source-ID sort within each domain gives a reproducible domain-balanced subset.
    bank = sorted([r for r in requests if len(r["questions"]) == 3], key=lambda r: r["source_id"])
    sst = sorted([r for r in requests if len(r["questions"]) == 1], key=lambda r: r["source_id"])
    return bank[:max(1, limit // 2)] + sst[:max(1, limit // 2)]


def flan(cfg, directory, bundle, run, main_model):
    baseline = FlanBaseline(cfg)
    limit = cfg["eval"]["flan_limit"]
    calibration = subset_requests(bundle, "calibration", cfg, limit)
    requests = subset_requests(bundle, "test", cfg, limit)
    temperatures, details = fit_temperatures(baseline.records(calibration))
    rows = baseline.records(requests)
    # FLAN timing is costly: use three repeats, reported explicitly.
    timing_cfg = json.loads(json.dumps(cfg))
    timing_cfg["eval"].update(latency_repeats=min(3, cfg["eval"]["latency_repeats"]), latency_warmup=1)
    latency = {"measurements": []}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        baseline.model.to(device)
        measurement = timing(baseline.records, requests, timing_cfg, device)
        latency["measurements"].extend(measurement["measurements"])
    write_json(directory / "flan_calibration.json", {"temperatures": temperatures, "fit": details,
               "model_revision": baseline.revision, "source_ids": [r["source_id"] for r in requests]})
    sources = {**bundle["sources"], "flan_model": {"id": cfg["eval"]["flan_model"], "revision": baseline.revision}}
    write_evaluation(rows, temperatures, cfg, run, "flan_subset", sources, latency)
    main_temperatures = dict(zip(HEADS, main_model.temperatures.cpu().tolist()))
    write_evaluation(collect_records(main_model, requests, cfg), main_temperatures, cfg, run,
                     "jevlet_flan_subset", bundle["sources"])
