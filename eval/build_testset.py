"""Generate a synthetic test set: one question per sampled chunk, with that chunk as the gold answer source.

Usage:  python -m eval.build_testset --n 50
Upload your documents through the API first. Review eval/testset.jsonl and delete poor questions.
"""
import argparse
import json
import random

from sqlalchemy import select

from app.config import get_settings
from app.db import Chunk, make_session_factory
from app.llm import LLMClient

PROMPT = """Write ONE specific question that can be answered using only the passage below.
The question must make sense on its own (do not say "the passage" or "this text").
Then give the short correct answer.
Return JSON only: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        item = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return item if item.get("question") and item.get("answer") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="eval/testset.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s = get_settings()
    llm = LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model)
    with make_session_factory(s.database_url)() as session:
        chunks = [c for c in session.scalars(select(Chunk)) if len(c.text) > 200]
    random.Random(args.seed).shuffle(chunks)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if written >= args.n:
                break
            item = parse_json(llm.chat([{"role": "user", "content": PROMPT.format(passage=chunk.text)}], 0.3))
            if item is None:
                continue
            f.write(json.dumps({"question": item["question"], "answer": item["answer"],
                                "gold_chunk_id": chunk.id}) + "\n")
            written += 1
            print(f"{written}/{args.n}  {item['question']}")
    print(f"Wrote {written} questions to {args.out}")


if __name__ == "__main__":
    main()
