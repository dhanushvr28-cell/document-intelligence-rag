"""Compare retrieval modes on the test set and (optionally) judge answer quality.

Usage:  python -m eval.run_eval                 # retrieval metrics for all four modes
        python -m eval.run_eval --answers       # also LLM-judged correctness and faithfulness
Writes eval/results.md.
"""
import argparse
import json
import time

from app.config import get_settings
from app.db import make_session_factory
from app.llm import LLMClient
from app.retrieval import MODES, CrossEncoderReranker, Retriever, SentenceTransformerEmbedder
from eval.metrics import hit_at_k, mean, reciprocal_rank

JUDGE = """You are grading a question-answering system.
Question: {question}
Reference answer: {reference}
Context given to the system:
{context}
System answer: {answer}

Return JSON only: {{"correct": true/false, "faithful": true/false}}
- correct: the system answer agrees with the reference answer.
- faithful: every claim in the system answer is supported by the context."""


def evaluate_retrieval(retriever, testset: list[dict], mode: str, k: int, candidate_k: int) -> dict:
    hits1, hitsk, rr, latency = [], [], [], []
    for item in testset:
        t = time.perf_counter()
        ranked = [c.id for c in retriever.search(item["question"], mode, k, candidate_k)]
        latency.append((time.perf_counter() - t) * 1000)
        hits1.append(hit_at_k(ranked, item["gold_chunk_id"], 1))
        hitsk.append(hit_at_k(ranked, item["gold_chunk_id"], k))
        rr.append(reciprocal_rank(ranked, item["gold_chunk_id"]))
    return {"mode": mode, "hit@1": mean(hits1), f"hit@{k}": mean(hitsk), "mrr": mean(rr),
            "latency_ms": mean(latency)}


def evaluate_answers(retriever, llm, testset: list[dict], mode: str, k: int, candidate_k: int) -> dict:
    correct, faithful = [], []
    for item in testset:
        chunks = retriever.search(item["question"], mode, k, candidate_k)
        contexts = [c.text for c in chunks]
        answer = llm.answer(item["question"], contexts, [])
        verdict = llm.chat([{"role": "user", "content": JUDGE.format(
            question=item["question"], reference=item["answer"],
            context="\n\n".join(contexts), answer=answer)}], 0.0)
        try:
            parsed = json.loads(verdict[verdict.find("{"):verdict.rfind("}") + 1])
        except json.JSONDecodeError:
            continue
        correct.append(1.0 if parsed.get("correct") else 0.0)
        faithful.append(1.0 if parsed.get("faithful") else 0.0)
    return {"mode": mode, "judged": len(correct), "answer_correctness": mean(correct),
            "faithfulness": mean(faithful)}


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v <= 1 else (f"{v:.0f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/testset.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    with open(args.testset, encoding="utf-8") as f:
        testset = [json.loads(line) for line in f if line.strip()]
    retriever = Retriever(make_session_factory(s.database_url), SentenceTransformerEmbedder(s.embedding_model),
                          s.chroma_dir, reranker=CrossEncoderReranker(s.reranker_model))

    report = [f"# Evaluation results\n\n{len(testset)} questions, top_k = {args.k}\n\n## Retrieval\n"]
    rows = [evaluate_retrieval(retriever, testset, m, args.k, s.candidate_k) for m in MODES]
    report.append(to_markdown(rows))
    if args.answers:
        llm = LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model)
        row = evaluate_answers(retriever, llm, testset, s.default_mode, args.k, s.candidate_k)
        report.append(f"\n\n## Answers ({s.llm_model}, LLM-judged)\n\n" + to_markdown([row]))
    text = "\n".join(report) + "\n"
    with open("eval/results.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
