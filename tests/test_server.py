"""Dashboard contract tests; live checkpoint verification is performed separately."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from jevlet.server import create_app


class RecordingEngine:
    labels = ["card arrival", "transaction charged twice", "card payment fee charged"]
    info = {"ready": True, "labels": labels, "device": "cpu", "checkpoint": "test"}

    def __init__(self):
        self.calls = []

    def predict(self, text, questions):
        self.calls.append((text, questions))
        return {"answers": [], "questions": questions, "encoder_passes": 1,
                "context_tokens_dropped": 0, "inference_ms": 1.0, "calibrated": True}


@pytest.fixture
def client():
    engine = RecordingEngine()
    with TestClient(create_app(engine=engine)) as session:
        yield session, engine


def banking(case="banking"):
    return {"case": case, "text": " I was charged twice. ",
            "options": RecordingEngine.labels, "checks": RecordingEngine.labels[:2]}


@pytest.mark.parametrize("case", ["banking", "routing"])
def test_three_questions_share_one_prediction(client, case):
    session, engine = client
    response = session.post("/api/predict", json=banking(case))
    assert response.status_code == 200
    assert len(engine.calls) == 1
    text, questions = engine.calls[0]
    assert text == "I was charged twice."
    assert [q["type"] for q in questions] == ["choice", "yesno", "yesno"]
    assert questions[1]["text"] == "Is this about card arrival?"
    assert response.json()["encoder_passes"] == 1


def test_score_uses_trained_rubric(client):
    session, engine = client
    response = session.post("/api/predict", json={"case": "reviews", "text": "Great film."})
    assert response.status_code == 200
    assert engine.calls[0][1] == [
        {"id": "sentiment", "type": "score", "text": "How positive is this review?"}]


@pytest.mark.parametrize("override", [
    {"text": "   "}, {"text": "x" * 12001}, {"case": "fraud"},
    {"options": ["card arrival"]}, {"options": ["card arrival"] * 11},
    {"options": ["card arrival", "card arrival"]},
    {"options": ["card arrival", "invented intent"]},
    {"checks": ["fraud risk"]}, {"checks": ["card arrival"] * 2},
    {"checks": RecordingEngine.labels}, {"case": "reviews"},
    {"question": "Am I rich?"},
])
def test_bad_requests_never_reach_model(client, override):
    session, engine = client
    response = session.post("/api/predict", json={**banking(), **override})
    assert response.status_code == 422
    assert engine.calls == []


def test_model_input_error_is_visible(client):
    session, engine = client

    def fail(*args):
        raise ValueError("Question block exceeds token budget")

    engine.predict = fail
    response = session.post("/api/predict", json=banking())
    assert response.status_code == 422
    assert response.json()["detail"] == "Question block exceeds token budget"


def test_pages_assets_and_catalogue(client):
    session, _ = client
    for path in ("/", "/cases/banking", "/cases/reviews", "/cases/routing"):
        response = session.get(path)
        assert response.status_code == 200
        assert "Jevlet Lab" in response.text
    assert session.get("/cases/fraud").status_code == 404
    assert session.get("/assets/app.js").status_code == 200
    assert session.get("/api/info").json()["labels"] == RecordingEngine.labels
