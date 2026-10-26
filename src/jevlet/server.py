"""Local, real-checkpoint case studies. No generated responses or mock fallback."""

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import ROOT

WEB = Path(__file__).parent / "web"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/multilabel-banking"


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: Literal["banking", "reviews", "routing", "emotions", "banking_multi"]
    text: str = Field(min_length=1, max_length=12000)
    options: list[str] = Field(default_factory=list, max_length=10)
    checks: list[str] = Field(default_factory=list, max_length=2)
    threshold: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_questions(self):
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Enter some text first.")
        if self.case == "reviews":
            if self.options or self.checks:
                raise ValueError("Review sentiment uses the fixed five-level rubric.")
        else:
            if not 2 <= len(self.options) <= 10:
                raise ValueError("Choose between 2 and 10 candidate intents.")
            if len(set(self.options)) != len(self.options) or len(set(self.checks)) != len(self.checks):
                raise ValueError("Candidate intents and checks must each be unique.")
        if self.case in ("emotions", "banking_multi") and self.checks:
            raise ValueError("Multi-label selection uses independent candidate labels without separate topic checks")
        if self.case not in ("emotions", "banking_multi") and self.threshold is not None:
            raise ValueError("An API threshold is only supported for multi-label selection")
        return self


def build_questions(request, labels, multilabel_labels=None, banking_multilabel_labels=None):
    if request.case in ("emotions", "banking_multi"):
        banking = request.case == "banking_multi"
        catalogue = banking_multilabel_labels if banking else multilabel_labels
        if not catalogue:
            raise ValueError("This checkpoint has no trained multi-label task for this case; load a matching checkpoint")
        if set(request.options) - set(catalogue):
            raise ValueError("Use labels from the loaded multi-label catalogue for this case")
        question = {"id": request.case, "type": "banking_multilabel" if banking else "multilabel",
                    "text": "Which intent components apply to this banking message?" if banking else "Which emotions are expressed in this text?",
                    "options": request.options}
        if request.threshold is not None:
            question["threshold"] = request.threshold
        return [question]
    if request.case == "reviews":
        return [{"id": "sentiment", "type": "score", "text": "How positive is this review?"}]
    unknown = set(request.options + request.checks) - set(labels)
    if unknown:
        raise ValueError("Use intents from the Banking77 catalogue.")
    return [{"id": "intent", "type": "choice", "text": "What is the customer's intent?",
             "options": request.options}] + [
        {"id": f"check-{i}", "type": "yesno", "text": f"Is this about {label}?"}
        for i, label in enumerate(request.checks)
    ]


class LocalEngine:
    def __init__(self, checkpoint, device, threads):
        from .config import seed_all
        from .data import pretty_label
        from .model import Jevlet

        checkpoint = Path(checkpoint)
        required = ("model.safetensors", "jevlet.json", "data.json", "calibration.json")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(f"Checkpoint {checkpoint} is incomplete: {', '.join(missing)}. "
                               "Supply a trained checkpoint with --checkpoint.")
        calibration = json.loads((checkpoint / "calibration.json").read_text())
        if any(calibration.get("fit", {}).get(head, {}).get("status") != "fitted"
               for head in ("choice", "score", "yesno")):
            raise RuntimeError("This dashboard requires a checkpoint with all three heads calibrated.")
        seed_all(42, threads)
        self.model = Jevlet.load(checkpoint, device)
        bundle = json.loads((checkpoint / "data.json").read_text())
        self.labels = [pretty_label(label) for label in bundle["label_names"]]
        self.multilabel_labels = []
        self.banking_multilabel_labels = []
        provenance = bundle.get("sources", {}).get("multilabel", {})
        banking_provenance = bundle.get("sources", {}).get("banking_multilabel", {})
        if "multilabel" in self.model.heads:
            if calibration.get("fit", {}).get("multilabel", {}).get("status") != "fitted":
                raise RuntimeError("Multi-label head requires fitted calibration")
            if provenance.get("id") == "google-research-datasets/go_emotions":
                self.multilabel_labels = bundle["multilabel_names"]
        if "banking_multilabel" in self.model.heads:
            if calibration.get("fit", {}).get("banking_multilabel", {}).get("status") != "fitted":
                raise RuntimeError("Banking multi-label head requires fitted calibration")
            if banking_provenance.get("id") == "PolyAI-LDN/task-specific-datasets/nlupp/banking":
                self.banking_multilabel_labels = [pretty_label(name) for name in bundle["banking_multilabel_names"]]
        self.lock = threading.Lock()
        evaluation = ROOT / "results" / f"{checkpoint.name}_jevlet.json"
        metrics = None
        if evaluation.is_file():
            report = json.loads(evaluation.read_text())
            if not report.get("smoke", True):
                metrics = report.get("calibrated")
        self.info = {"ready": True, "checkpoint": checkpoint.name,
                     "device": str(next(self.model.parameters()).device),
                     "backbone": "DistilBERT", "max_questions": 3, "calibrated": True,
                     "labels": self.labels, "benchmark": metrics, "multilabel_labels": self.multilabel_labels,
                     "multilabel_threshold": getattr(self.model, "multilabel_threshold", None),
                     "multilabel_data": provenance, "banking_multilabel_labels": self.banking_multilabel_labels,
                     "banking_multilabel_threshold": getattr(self.model, "banking_multilabel_threshold", None),
                     "banking_multilabel_data": banking_provenance,
                     "banking_multilabel_descriptions": {pretty_label(k): v for k, v in bundle.get("banking_multilabel_descriptions", {}).items()}}

    def predict(self, text, questions):
        # A request runs exactly one shared encoder pass; concurrent callers queue here.
        with self.lock:
            started = time.perf_counter()
            result = self.model.predict(text, questions, calibrated=True)
            elapsed = (time.perf_counter() - started) * 1000
        return {**result, "inference_ms": round(elapsed, 2), "questions": questions,
                "encoder_passes": 1, "calibrated": True}


def create_app(checkpoint=DEFAULT_CHECKPOINT, device="cpu", threads=4, *, engine=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine if engine is not None else LocalEngine(checkpoint, device, threads)
        yield

    app = FastAPI(title="Jevlet Lab", version="0.1.0", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=WEB), name="assets")

    @app.get("/api/info")
    def info():
        return app.state.engine.info

    @app.post("/api/predict")
    def predict(request: DecisionRequest):
        active = app.state.engine
        try:
            questions = build_questions(request, active.labels, getattr(active, "multilabel_labels", []),
                                        getattr(active, "banking_multilabel_labels", []))
            return {"case": request.case, **active.predict(request.text, questions)}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logging.getLogger(__name__).exception("Local prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed. Check the server terminal.") from error

    @app.get("/", include_in_schema=False)
    @app.get("/cases/{case}", include_in_schema=False)
    def page(case: str = ""):
        if case and case not in ("banking", "reviews", "routing", "emotions", "banking_multi"):
            raise HTTPException(status_code=404, detail="Case study not found")
        return FileResponse(WEB / "index.html")

    return app
