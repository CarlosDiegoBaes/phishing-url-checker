"""FastAPI app: ``POST /check`` scores a URL and ``/`` serves the single page.

Run it with ``uvicorn --factory phishguard.web:create_app``. The model is
loaded once, when the app is created, and is never retrained here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from phishguard import __version__
from phishguard.paths import MODEL_PATH
from phishguard.web.service import CheckService

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CheckRequest(BaseModel):
    """Body of ``POST /check``. Defanged text such as ``hxxp`` or ``[.]`` is fine."""

    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject input that is only whitespace."""
        if not value.strip():
            raise ValueError("url must not be blank")
        return value


class FeatureOut(BaseModel):
    """One model feature and the value it had for this URL."""

    name: str
    label: str
    value: float
    description: str


class ContributionOut(BaseModel):
    """How far one feature pushed the score, for models that can say."""

    name: str
    value: float


class CheckResponse(BaseModel):
    """Body returned by ``POST /check``.

    ``verdict`` is ``"phishing"`` or ``"legitimate"`` at ``threshold``.
    ``band`` is LIKELY SAFE, SUSPICIOUS, or LIKELY PHISHING. ``contributions``
    is ``null`` for the shipped random forest.
    """

    input: str
    url: str
    verdict: str
    band: str
    score: float
    threshold: float
    model: str
    features: list[FeatureOut]
    rule_flags: list[str]
    contributions: list[ContributionOut] | None
    explanation: str


def create_app(model_path: Path = MODEL_PATH, service: CheckService | None = None) -> FastAPI:
    """Build the app around a saved model, or around a ready ``CheckService``."""
    checker = service if service is not None else CheckService.from_path(model_path)
    app = FastAPI(title="phishguard", version=__version__)
    app.state.checker = checker

    @app.post("/check", response_model=CheckResponse)
    def check(request: CheckRequest) -> CheckResponse:
        """Score one URL without fetching it."""
        result = checker.check(request.url)
        payload = asdict(result)
        if result.contributions is not None:
            payload["contributions"] = [
                {"name": name, "value": value} for name, value in result.contributions
            ]
        return CheckResponse.model_validate(payload)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        """Liveness check that also reports which model is loaded."""
        return {"status": "ok", "model": checker.model_name, "threshold": checker.threshold}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """The single-page frontend."""
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
