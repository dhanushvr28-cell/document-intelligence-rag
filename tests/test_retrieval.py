from app.retrieval import rrf
from eval.metrics import hit_at_k, mean, reciprocal_rank
from eval.run_eval import evaluate_retrieval

DOC = b"""The warranty period for the X200 drone is 18 months from the date of purchase.

Battery charging takes 90 minutes using the supplied fast charger.

The maximum flight altitude is limited to 120 metres by the firmware.

Returns are accepted within 30 days if the product is unused and in original packaging.

Customer support is available by email at any time and by phone on weekdays."""


def test_rrf_prefers_items_ranked_high_in_both_lists():
    assert rrf([[1, 2, 3], [2, 1, 4]])[:2] == [1, 2]
    assert rrf([[5, 6], [6, 7]])[0] == 6
    assert rrf([[], []]) == []


def test_metrics():
    assert hit_at_k([4, 9, 2], 9, 1) == 0.0 and hit_at_k([4, 9, 2], 9, 2) == 1.0
    assert reciprocal_rank([4, 9, 2], 2) == 1 / 3 and reciprocal_rank([4], 7) == 0.0
    assert mean([]) == 0.0


def test_all_modes_find_the_right_chunk(client):
    client.post("/documents", files={"file": ("manual.txt", DOC)})
    retriever = client.app.state.retriever
    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        top = retriever.search("How long is the warranty period?", mode, top_k=2, candidate_k=5)
        assert "18 months" in top[0].text, mode


def test_document_filter_and_delete(client):
    a = client.post("/documents", files={"file": ("a.txt", DOC)}).json()["id"]
    b = client.post("/documents", files={"file": ("b.txt", b"Warranty for the Z9 camera lasts 24 months in total.")}).json()["id"]
    retriever = client.app.state.retriever
    only_b = retriever.search("warranty", "hybrid", 5, 10, document_ids=[b])
    assert only_b and all(c.document_id == b for c in only_b)
    assert client.delete(f"/documents/{b}").status_code == 204
    assert all(c.document_id == a for c in retriever.search("warranty", "hybrid", 5, 10))


def test_eval_harness_reports_metrics(client):
    client.post("/documents", files={"file": ("manual.txt", DOC)})
    retriever = client.app.state.retriever
    gold = retriever.search("fast charger", "bm25", 1)[0].id
    row = evaluate_retrieval(retriever, [{"question": "How long does battery charging take?", "gold_chunk_id": gold}],
                             "hybrid", 3, 5)
    assert row["hit@3"] == 1.0 and 0 < row["mrr"] <= 1.0
