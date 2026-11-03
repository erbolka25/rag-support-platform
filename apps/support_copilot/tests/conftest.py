import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rag_core.generation import BaseLLMClient
from rag_core.retrieval import BaseEmbedder, create_index
from rag_core.retrieval.embeddings import l2_normalize
from rag_core.service import RagService
from sqlmodel import Session

from support_copilot.api.app import create_app
from support_copilot.config import CopilotSettings
from support_copilot.db import init_db, make_engine
from support_copilot.rag import rag_settings


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-words embedder: no model download, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))


class FakeLLM(BaseLLMClient):
    """Returns a canned reply (or raises) and records every prompt; never calls an API."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def make_settings(tmp_path, **overrides) -> CopilotSettings:
    # min_score 0: the default is calibrated for e5, the hashing embedder scores lower.
    values = {"data_dir": tmp_path, "index_backend": "numpy", "min_score": 0.0,
              "llm_provider": "none", "max_upload_mb": 1}
    return CopilotSettings(_env_file=None, **(values | overrides))


@pytest.fixture
def make_client(tmp_path):
    """make_client(llm=None, **settings_overrides) -> TestClient on a fresh DB and index."""
    clients = []

    def factory(llm: BaseLLMClient | None = None, **overrides) -> TestClient:
        settings = make_settings(tmp_path, **overrides)
        rag = RagService(rag_settings(settings), HashingEmbedder(), create_index("numpy"), llm)
        client = TestClient(create_app(settings, rag), raise_server_exceptions=False)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.app.state.engine.dispose()


@pytest.fixture
def client(make_client):
    return make_client()


TICKET = {
    "customer_name": "Айгуль",
    "customer_email": "aigul@example.com",
    "question": "Сколько стоит доставка в Ош?",
}


def create_ticket(client, **overrides) -> dict:
    resp = client.post("/tickets", json=TICKET | overrides)
    assert resp.status_code == 201, resp.text
    return resp.json()


def upload(client, name="delivery.txt", text="Доставка в Ош стоит 300 сом и занимает 2 дня."):
    data = text.encode() if isinstance(text, str) else text
    return client.post("/kb/upload", files={"file": (name, data, "text/plain")})
