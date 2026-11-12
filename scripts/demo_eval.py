"""One-command demo: download public research papers, index them, build a test set and evaluate.

Usage:  python -m scripts.demo_eval [--n 30]
Progress is also written to eval/run_log.txt.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import httpx

PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "sentence_bert.pdf": "https://arxiv.org/pdf/1908.10084",
    "retrieval_augmented_generation.pdf": "https://arxiv.org/pdf/2005.11401",
    "lora.pdf": "https://arxiv.org/pdf/2106.09685",
}
PDF_DIR = "data/sample_pdfs"
LOG = "eval/run_log.txt"


def log(message: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def download() -> list[str]:
    os.makedirs(PDF_DIR, exist_ok=True)
    paths = []
    for name, url in PAPERS.items():
        path = os.path.join(PDF_DIR, name)
        if not os.path.exists(path):
            log(f"Downloading {name}")
            res = httpx.get(url, follow_redirects=True, timeout=120)
            res.raise_for_status()
            with open(path, "wb") as out:
                out.write(res.content)
        paths.append(path)
    return paths


def ingest(paths: list[str]) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    log("Loading embedding model (first run downloads it)")
    client = TestClient(create_app())
    indexed = {d["filename"] for d in client.get("/documents").json()}
    for path in paths:
        name = os.path.basename(path)
        if name in indexed:
            continue
        with open(path, "rb") as f:
            res = client.post("/documents", files={"file": (name, f.read())})
        log(f"Indexed {name}: {res.json()}")


def run(module: str, *args: str) -> None:
    log(f"Running {module} {' '.join(args)}")
    proc = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    for line in (proc.stdout + proc.stderr).strip().splitlines()[-25:]:
        log("  " + line)
    proc.check_returncode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    os.makedirs("eval", exist_ok=True)
    log("=== demo evaluation started ===")
    try:
        ingest(download())
        testset = "eval/testset.jsonl"
        have = sum(1 for _ in open(testset, encoding="utf-8")) if os.path.exists(testset) else 0
        if have < args.n:
            run("eval.build_testset", "--n", str(args.n))
        run("eval.run_eval", "--answers")
        log("=== finished: see eval/results.md ===")
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
