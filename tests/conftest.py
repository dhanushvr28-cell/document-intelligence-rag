import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.retrieval import tokenize


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download needed."""
    dim = 256

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeReranker:
    def score(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) for t in texts]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def answer(self, question, contexts, history):
        self.calls.append({"question": question, "contexts": contexts, "history": history})
        return f"Answer based on {len(contexts)} passages [1]"


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(tmp_path, llm):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/t.db", chroma_dir=str(tmp_path / "chroma"),
                        chunk_size=200, chunk_overlap=40, max_upload_mb=1)
    app = create_app(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)
    return TestClient(app)
