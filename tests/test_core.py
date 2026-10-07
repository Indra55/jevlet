"""Fast, offline tests using a real tokenizer and a tiny randomly initialized encoder."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertModel

from jevlet.config import load_config
from jevlet.data import TOKENS, collate, normalized, pack_request, sample_request, unique_rows
from jevlet.eval import unpack_items
from jevlet.metrics import ece, fit_temperatures, risk_coverage, score_records
from jevlet.model import Jevlet, OrdinalHead, masked_choice_probs, ordinal_probs, training_loss


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                "what", "is", "this", "about", "card", "payment", "good", "bad", "?", "review"]))
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tok.add_special_tokens({"additional_special_tokens": list(TOKENS)})
    return tok


@pytest.fixture
def example():
    return {"text": "card payment " * 100,
            "questions": [{"id": "intent", "type": "choice", "text": "What is this about?",
                           "options": ["card", "payment"], "option_labels": [0, 1], "target": 1},
                          {"id": "sentiment", "type": "score", "text": "Good review?", "target": 3},
                          {"id": "about", "type": "yesno", "text": "Is this about payment?", "target": 1}]}


@pytest.fixture
def model(tokenizer):
    cfg = DistilBertConfig(vocab_size=len(tokenizer), dim=16, hidden_dim=32, n_layers=1,
                          n_heads=2, dropout=0.0, attention_dropout=0.0)
    obj = Jevlet(DistilBertModel(cfg))
    obj.tokenizer, obj.max_length, obj.source_revision = tokenizer, 48, "offline-test"
    return obj.eval()


def test_pack_markers_and_context_truncation(tokenizer, example):
    item = pack_request(tokenizer, example, 48)
    assert len(item["input_ids"]) == 48
    assert item["context_tokens_dropped"] > 0
    assert len(item["positions"]) == 3
    for qpos, opos in item["positions"]:
        assert item["input_ids"][qpos] == tokenizer.convert_tokens_to_ids("[Q]")
        assert all(item["input_ids"][p] == tokenizer.convert_tokens_to_ids("[OPT]") for p in opos)
    short = {**example, "text": "card"}
    assert pack_request(tokenizer, short, 48)["positions"] == item["positions"]


def test_literal_markers_cannot_inject_positions(tokenizer, example):
    example["text"] = "[Q] card [OPT] payment [SEP]"
    example["questions"][0]["options"][0] = "card [Q]"
    item = pack_request(tokenizer, example, 48)
    assert item["input_ids"].count(tokenizer.convert_tokens_to_ids("[Q]")) == 3
    assert item["input_ids"].count(tokenizer.convert_tokens_to_ids("[OPT]")) == 2


def test_question_overflow_is_rejected(tokenizer, example):
    example["questions"][0]["text"] = "card " * 100
    with pytest.raises(ValueError, match="Question block"):
        pack_request(tokenizer, example, 48)


@pytest.mark.parametrize("change", ["duplicate_ids", "one_option", "duplicate_options", "unknown_type"])
def test_invalid_questions(tokenizer, example, change):
    if change == "duplicate_ids":
        example["questions"][1]["id"] = "intent"
    elif change == "one_option":
        example["questions"][0]["options"] = ["card"]
    elif change == "duplicate_options":
        example["questions"][0]["options"] = ["card", "card"]
    else:
        example["questions"][0]["type"] = "unknown"
    with pytest.raises(ValueError):
        pack_request(tokenizer, example, 48)


def test_option_masking_and_empty_rows():
    logits = torch.tensor([[1., 2., 100.], [0., 0., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    probs = masked_choice_probs(logits, mask)
    assert probs[0, 2] == 0
    assert probs[0].sum().item() == pytest.approx(1)
    assert probs[1].sum() == 0
    probs[0, 0].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


def test_ordinal_head_shapes_and_monotonicity():
    head = OrdinalHead(16)
    for _ in range(3):
        z = head(torch.randn(2, 3, 16))
        assert z.shape == (2, 3, 4)
        assert (z[..., :-1] > z[..., 1:]).all()
        for temperature in (0.3, 1, 5):
            p = ordinal_probs(z / temperature)
            assert p.shape == (2, 3, 5)
            assert (p >= 0).all()
            assert torch.allclose(p.sum(-1), torch.ones(2, 3))


def test_mixed_heads_backward(model, tokenizer, example):
    item = pack_request(tokenizer, example, 48)
    batch = collate([item], tokenizer.pad_token_id)
    outputs = model(batch)
    loss = training_loss(model, outputs, batch, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.choice_head.weight.grad.abs().sum() > 0
    assert model.score_head.score.weight.grad.abs().sum() > 0
    assert model.yesno_head.weight.grad.abs().sum() > 0


def test_checkpoint_reload_and_one_encoder_call(model, example, tmp_path):
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as forward:
        expected = model.predict(example["text"], example["questions"])
        assert forward.call_count == 1
    model.temperatures.copy_(torch.tensor([1.2, 2.0, 0.9]))
    expected = model.predict(example["text"], example["questions"])
    model.save(tmp_path)
    loaded = Jevlet.load(tmp_path)
    actual = loaded.predict(example["text"], example["questions"])
    assert [r["id"] for r in actual["answers"]] == [q["id"] for q in example["questions"]]
    for left, right in zip(expected["answers"], actual["answers"]):
        assert left["answer"] == right["answer"]
        np.testing.assert_allclose(left["probabilities"], right["probabilities"], atol=1e-6)


def test_unpacked_keeps_identical_context_tokens(tokenizer, example):
    original = pack_request(tokenizer, example, 48)
    separate = unpack_items(tokenizer, [original], 48)
    context = original["input_ids"][original["input_ids"].index(tokenizer.sep_token_id) + 1:-1]
    assert len(separate) == 3
    for item in separate:
        assert item["input_ids"][item["input_ids"].index(tokenizer.sep_token_id) + 1:-1] == context


def test_ece_includes_both_endpoints():
    assert ece([0, 1], [0, 1]) == 0
    assert ece([0, 1], [1, 0]) == 1
    assert ece([0.5, 0.5], [0, 1]) == 0
    with pytest.raises(ValueError):
        ece([], [])


def test_temperature_fit_does_not_worsen_calibration_nll():
    records = [{"type": "choice", "logits": [8., 0.], "target": i % 2} for i in range(12)]
    records.append({"type": "choice", "logits": [1., 0., -1.], "target": 1})
    temps, details = fit_temperatures(records)
    assert np.isfinite(temps["choice"]) and temps["choice"] > 0
    assert details["choice"]["nll_after"] <= details["choice"]["nll_before"] + 1e-6


def test_choice_metrics_map_to_original_intents():
    records = [{"type": "choice", "logits": [8., 0.], "target": 0, "option_labels": [1, 0]},
               {"type": "choice", "logits": [0., 8.], "target": 1, "option_labels": [1, 0]}]
    scores, _ = score_records(records, intent_count=2)
    assert scores["choice"]["macro_f1"] == 1


def test_risk_coverage_groups_tied_confidence():
    preds = [{"type": "yesno", "confidence": 0.8, "target": 1, "predicted": y} for y in (1, 0)]
    coverage, risk = risk_coverage(preds, "yesno")
    assert coverage == [1.0] and risk == [0.5]


def test_sampling_is_deterministic_and_no_presence_shortcut():
    cfg = load_config()
    names = ["label_a", "label_b", "label_c", "label_d"]
    cfg["options_max"] = 4
    for i in range(50):
        row = {"id": str(i), "domain": "banking", "label": i % 4, "text": "card"}
        req = sample_request(row, names, cfg)
        assert req == sample_request(row, names, cfg)
        choice = req["questions"][0]
        assert choice["option_labels"][choice["target"]] == row["label"]
        for q in req["questions"][1:]:
            assert q["query_label"] in choice["option_labels"]
            assert q["target"] == (q["query_label"] == row["label"])


def test_context_duplicate_filter_keeps_splits_isolated():
    from collections import Counter
    blocked = {normalized("  CARD payment ")}
    rows = unique_rows([{"text": "card  PAYMENT", "label": 1}, {"text": "good", "label": 0}],
                       "banking", "train", blocked, Counter())
    assert [r["text"] for r in rows] == ["good"]


def test_flan_scoring_with_a_real_tiny_seq2seq_model(tokenizer, example):
    from transformers import T5Config, T5ForConditionalGeneration
    from jevlet.baselines import FlanBaseline
    from jevlet.metrics import distribution
    tokenizer.eos_token = tokenizer.sep_token
    baseline = FlanBaseline.__new__(FlanBaseline)
    baseline.tokenizer = tokenizer
    baseline.cfg = {"max_length": 128}
    baseline.model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(tokenizer), d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
        num_heads=2, decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)).eval()
    example["source_id"] = "test"
    rows = baseline.records([example])
    assert len(rows) == 3
    for row in rows:
        p = distribution(row["type"], torch.tensor(row["logits"]))
        assert torch.isfinite(p).all() and (p >= 0).all()
        assert p.sum().item() == pytest.approx(1)
