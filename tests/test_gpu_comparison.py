"""CUDA regression checks for the new categorical and comparison paths."""
import torch
from transformers import DistilBertConfig, DistilBertModel, BertTokenizerFast

from jevlet.data import TOKENS, collate, pack_request
from jevlet.gpu_compare import ContextClassifier, loss_for
from jevlet.metrics import fit_temperatures, score_records
from jevlet.model import Jevlet, training_loss


def test_categorical_gpu_backward_reload_and_calibration(tmp_path):
    assert torch.cuda.is_available(), "GPU-only experiment requires CUDA"
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "review", "?"]))
    tok = BertTokenizerFast(vocab_file=str(path))
    tok.add_special_tokens({"additional_special_tokens": list(TOKENS)})
    cfg = DistilBertConfig(vocab_size=len(tok), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
                          dropout=0., attention_dropout=0.)
    model = Jevlet(DistilBertModel(cfg), "categorical").cuda().eval()
    model.tokenizer, model.max_length, model.source_revision = tok, 48, "gpu-test"
    request = {"text": "good review", "questions": [{"id": "sentiment", "type": "score",
               "text": "good review?", "target": 1}]}
    batch = collate([pack_request(tok, request, 48)], tok.pad_token_id)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    outputs = model(batch)
    assert outputs["score"].shape == (1, 3, 5)
    loss = training_loss(model, outputs, batch, 0.5)
    loss.backward()
    assert torch.isfinite(loss) and model.score_head.weight.grad.abs().sum() > 0
    expected = model.predict(request["text"], request["questions"])
    model.save(tmp_path / "model")
    loaded = Jevlet.load(tmp_path / "model", "cuda")
    actual = loaded.predict(request["text"], request["questions"])
    assert actual == expected and loaded.score_mode == "categorical"
    rows = [{"type": "score", "score_mode": "categorical", "logits": [3., 1., 0., 0., 0.],
             "target": i % 5} for i in range(20)]
    temps, details = fit_temperatures(rows, "cuda")
    assert details["score"]["nll_after"] <= details["score"]["nll_before"] + 1e-6
    metrics, _ = score_records(rows, temps)
    assert metrics["score"]["samples"] == 20


def test_gpu_comparison_mixed_losses_and_masked_labels():
    model = ContextClassifier(features=10).cuda()
    requests = [{"questions": [{"type": "choice", "option_labels": [4, 2, 1], "target": 1},
                              {"type": "yesno", "query_label": 4, "target": 0}]},
                {"questions": [{"type": "score", "target": 3}]}]
    for svm in (False, True):
        model.zero_grad()
        loss = loss_for(model(torch.randn(2, 10, device="cuda")), requests, svm)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.intent.weight.grad[0].abs().sum() == 0
