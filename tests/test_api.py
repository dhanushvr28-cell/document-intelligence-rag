from tests.test_retrieval import DOC


def test_health_and_ui(client):
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 0}
    assert "Document Intelligence" in client.get("/").text


def test_upload_validation(client):
    assert client.post("/documents", files={"file": ("a.exe", b"x")}).status_code == 415
    assert client.post("/documents", files={"file": ("a.txt", b"")}).status_code == 400
    assert client.post("/documents", files={"file": ("a.txt", b"   \n ")}).status_code == 422
    assert client.post("/documents", files={"file": ("a.pdf", b"not a pdf")}).status_code == 422
    assert client.post("/documents", files={"file": ("a.txt", b"x" * (1024 * 1024 + 1))}).status_code == 413
    assert client.get("/documents").json() == []


def test_upload_ask_and_conversation(client, llm):
    up = client.post("/documents", files={"file": ("manual.txt", DOC)})
    assert up.status_code == 201 and up.json()["chunks"] >= 3

    r = client.post("/ask", json={"question": "How long is the warranty period?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid_rerank" and body["sources"][0]["filename"] == "manual.txt"
    assert "18 months" in body["sources"][0]["text"]

    r2 = client.post("/ask", json={"question": "And how long does charging take?",
                                   "conversation_id": body["conversation_id"], "mode": "bm25"})
    assert r2.json()["conversation_id"] == body["conversation_id"]
    assert len(llm.calls[1]["history"]) == 2  # previous turn is passed to the model

    convo = client.get(f"/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant", "user", "assistant"]


def test_ask_validation_and_empty_index(client, llm):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "conversation_id": 999}).status_code == 404
    r = client.post("/ask", json={"question": "What is the warranty?"})
    assert "could not find" in r.json()["answer"] and llm.calls == []


def test_llm_failure_returns_502(client, llm):
    client.post("/documents", files={"file": ("manual.txt", DOC)})
    def boom(*a, **k): raise RuntimeError("rate limited")
    llm.answer = boom
    assert client.post("/ask", json={"question": "How long is the warranty?"}).status_code == 502
    assert client.delete("/documents/42").status_code == 404
