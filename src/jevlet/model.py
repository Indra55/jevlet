"""One shared encoder with choice, ordinal, and binary marker heads."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import HEADS, MULTILABEL_HEADS, device_for, write_json
from .data import TOKENS, collate, packed_items


class OrdinalHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.first_threshold = nn.Parameter(torch.tensor(-1.5))
        self.raw_gaps = nn.Parameter(torch.zeros(3))

    def forward(self, hidden):
        offsets = torch.cat((self.raw_gaps.new_zeros(1), F.softplus(self.raw_gaps).cumsum(0)))
        return self.score(hidden) - (self.first_threshold + offsets)


def ordinal_probs(logits):
    cumulative = logits.float().sigmoid()
    return torch.cat((1 - cumulative[..., :1], cumulative[..., :-1] - cumulative[..., 1:],
                      cumulative[..., -1:]), dim=-1)


def masked_choice_probs(logits, mask, temperature=1.0):
    # Non-choice rows have no options. Keep their unused distribution finite.
    safe_mask = mask.clone()
    safe_mask[..., 0] |= ~mask.any(-1)
    probs = (logits.float() / temperature).masked_fill(~safe_mask, -torch.inf).softmax(-1)
    return probs * mask


class Jevlet(nn.Module):
    def __init__(self, encoder, score_mode="ordinal"):
        super().__init__()
        self.encoder = encoder
        size = encoder.config.hidden_size
        self.choice_head = nn.Linear(size, 1)
        if score_mode not in ("ordinal", "categorical"):
            raise ValueError("score_mode must be ordinal or categorical")
        self.score_mode = score_mode
        self.score_head = OrdinalHead(size) if score_mode == "ordinal" else nn.Linear(size, 5)
        self.yesno_head = nn.Linear(size, 1)
        self.register_buffer("temperatures", torch.ones(3))
        self.tokenizer = None
        self.max_length = 256

    @classmethod
    def pretrained(cls, cfg):
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
        tokenizer.add_special_tokens({"additional_special_tokens": list(TOKENS)})
        encoder = AutoModel.from_pretrained(cfg["model"], revision=cfg["model_revision"])
        encoder.resize_token_embeddings(len(tokenizer))
        model = cls(encoder, cfg.get("score_mode", "ordinal"))
        model.tokenizer, model.max_length = tokenizer, cfg["max_length"]
        model.source_revision = getattr(encoder.config, "_commit_hash", None)
        return model

    @property
    def heads(self):
        return HEADS + tuple(head for head in MULTILABEL_HEADS if hasattr(self, head + "_head"))

    def enable_multilabel(self):
        if not hasattr(self, "multilabel_head"):
            self.multilabel_head = nn.Linear(self.encoder.config.hidden_size, 1).to(self.temperatures.device)
            self.multilabel_head.load_state_dict(self.choice_head.state_dict())
            self.temperatures = torch.cat((self.temperatures, self.temperatures.new_ones(1)))
            self.multilabel_threshold = 0.5
        return self

    def enable_banking_multilabel(self):
        self.enable_multilabel()
        if not hasattr(self, "banking_multilabel_head"):
            self.banking_multilabel_head = nn.Linear(self.encoder.config.hidden_size, 1).to(self.temperatures.device)
            self.banking_multilabel_head.load_state_dict(self.multilabel_head.state_dict())
            self.temperatures = torch.cat((self.temperatures, self.temperatures.new_ones(1)))
            self.banking_multilabel_threshold = 0.5
        return self

    def forward(self, batch):
        hidden = self.encoder(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"]).last_hidden_state
        qhidden = hidden.gather(1, batch["qpos"].unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        option_positions = batch["opos"].flatten(1)
        ohidden = hidden.gather(1, option_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        ohidden = ohidden.reshape(hidden.shape[0], 3, 10, hidden.shape[-1])
        output = {"choice": self.choice_head(ohidden).squeeze(-1).float(),
                "score": self.score_head(qhidden).float(),
                "yesno": self.yesno_head(qhidden).squeeze(-1).float()}
        for head in MULTILABEL_HEADS:
            if head in self.heads:
                output[head] = getattr(self, head + "_head")(ohidden).squeeze(-1).float()
        return output

    def probabilities(self, logits, mask, calibrated=True):
        temp = self.temperatures if calibrated else torch.ones_like(self.temperatures)
        yes = (logits["yesno"] / temp[2]).sigmoid()
        output = {"choice": masked_choice_probs(logits["choice"], mask, temp[0]),
                "score": (ordinal_probs(logits["score"] / temp[1]) if self.score_mode == "ordinal"
                          else (logits["score"] / temp[1]).softmax(-1)),
                "yesno": torch.stack((1 - yes, yes), dim=-1)}
        for head in MULTILABEL_HEADS:
            if head in self.heads:
                output[head] = (logits[head] / temp[self.heads.index(head)]).sigmoid() * mask
        return output

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(directory / "model.safetensors"))
        self.encoder.config.save_pretrained(directory / "encoder")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(directory / "jevlet.json", {"max_length": self.max_length,
                   "source_revision": self.source_revision, "heads": list(self.heads),
                   "format_version": 3 if "banking_multilabel" in self.heads else 2 if "multilabel" in self.heads else 1,
                   "multilabel_threshold": getattr(self, "multilabel_threshold", None),
                   "banking_multilabel_threshold": getattr(self, "banking_multilabel_threshold", None),
                   "score_mode": self.score_mode})

    @classmethod
    def load(cls, checkpoint, device="cpu"):
        directory = Path(checkpoint)
        spec = json.loads((directory / "jevlet.json").read_text())
        if spec.get("format_version", 1) not in (1, 2, 3):
            raise ValueError("Unsupported checkpoint format")
        model = cls(AutoModel.from_config(AutoConfig.from_pretrained(directory / "encoder")),
                    spec.get("score_mode", "ordinal"))
        model.tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        model.max_length, model.source_revision = spec["max_length"], spec["source_revision"]
        if "multilabel" in spec.get("heads", []):
            model.enable_multilabel()
            model.multilabel_threshold = spec["multilabel_threshold"]
            if not isinstance(model.multilabel_threshold, (int, float)) or not 0 < model.multilabel_threshold < 1:
                raise ValueError("Checkpoint multi-label threshold must be between zero and one")
        if "banking_multilabel" in spec.get("heads", []):
            model.enable_banking_multilabel()
            model.banking_multilabel_threshold = spec["banking_multilabel_threshold"]
            if not isinstance(model.banking_multilabel_threshold, (int, float)) or not 0 < model.banking_multilabel_threshold < 1:
                raise ValueError("Checkpoint banking threshold must be between zero and one")
        model.load_state_dict(load_file(str(directory / "model.safetensors")))
        if not torch.isfinite(model.temperatures).all() or (model.temperatures <= 0).any():
            raise ValueError("Checkpoint temperatures must be finite and positive")
        return model.to(device_for(device)).eval()

    @torch.inference_mode()
    def predict(self, text, questions, calibrated=True):
        self.eval()
        for head in MULTILABEL_HEADS:
            if any(q.get("type") == head for q in questions) and head not in self.heads:
                raise ValueError(f"This checkpoint has no trained multi-label head for {head}; use a matching checkpoint")
        items = packed_items(self.tokenizer, [{"text": text, "questions": questions}], self.max_length)
        batch = collate(items, self.tokenizer.pad_token_id)
        device = next(self.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        probs = self.probabilities(self(batch), batch["omask"], calibrated)
        answers = []
        for i, question in enumerate(questions):
            kind = question["type"]
            p = probs[kind][0, i]
            if kind in ("choice",) + MULTILABEL_HEADS:
                p = p[:len(question["options"])]
            p = p.cpu().tolist()
            if kind in MULTILABEL_HEADS:
                threshold = question.get("threshold", getattr(self, kind + "_threshold"))
                selected = [option for option, probability in zip(question["options"], p) if probability >= threshold]
                answers.append({"id": question["id"], "type": kind, "answer": selected,
                                "options": question["options"], "probabilities": p, "threshold": threshold,
                                "probability_semantics": "independent_label_probabilities"})
                continue
            winner = max(range(len(p)), key=p.__getitem__)
            value = question["options"][winner] if kind == "choice" else winner + 1 if kind == "score" else bool(winner)
            answer = {"id": question["id"], "type": kind, "answer": value,
                      "probabilities": p, "confidence": max(p)}
            if kind == "score":
                answer["expected_score"] = sum((j + 1) * v for j, v in enumerate(p))
            if kind == "choice":
                answer["options"] = question["options"]
            answers.append(answer)
        return {"answers": answers, "context_tokens_dropped": items[0]["context_tokens_dropped"]}


def training_loss(model, logits, batch, brier_weight):
    probabilities = model.probabilities(logits, batch["omask"], calibrated=False)
    losses = {head: [] for head in model.heads}
    for b, item in enumerate(batch["items"]):
        for q, question in enumerate(item["request"]["questions"]):
            head, target = question["type"], question["target"]
            z, p = logits[head][b, q], probabilities[head][b, q]
            if head in MULTILABEL_HEADS:
                k = len(question["options"])
                target = z.new_tensor(target)
                observed = target >= 0
                z, p, target = z[:k][observed], p[:k][observed], target[observed]
                base = F.binary_cross_entropy_with_logits(z, target)
                losses[head].append(base + brier_weight * (p - target).square().mean())
                continue
            if head == "choice":
                k = len(question["options"])
                z, p = z[:k], p[:k]
                base = F.cross_entropy(z.unsqueeze(0), z.new_tensor([target], dtype=torch.long))
            elif head == "score" and model.score_mode == "categorical":
                base = F.cross_entropy(z.unsqueeze(0), z.new_tensor([target], dtype=torch.long))
            elif head == "score":
                cumulative_target = (torch.arange(4, device=z.device) < target).float()
                base = F.binary_cross_entropy_with_logits(z, cumulative_target)
            else:
                base = F.binary_cross_entropy_with_logits(z, z.new_tensor(float(target)))
            one_hot = F.one_hot(torch.tensor(target, device=p.device), len(p)).float()
            losses[head].append(base + brier_weight * (p - one_hot).square().sum())
    return torch.stack([torch.stack(values).mean() for values in losses.values() if values]).mean()
