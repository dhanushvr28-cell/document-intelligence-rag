# Document Intelligence Assistant

Ask questions about your own documents (PDF, DOCX, TXT, MD) and get answers with numbered citations to the exact page and passage.

Built to show the parts of a retrieval-augmented generation (RAG) system that matter in production: hybrid retrieval, reranking, grounded answers, and a measured evaluation of each retrieval strategy.

## Features

- **Hybrid retrieval**: dense vector search (Chroma + sentence-transformers) fused with BM25 keyword search using reciprocal rank fusion.
- **Cross-encoder reranking** of the fused candidates.
- **Grounded answers with citations** from any OpenAI-compatible LLM (Groq, Gemini), and a fixed refusal when the answer is not in the documents.
- **Multi-turn conversations** stored in SQL along with the sources used for every answer.
- **Evaluation harness**: generates a test set from your documents and reports Hit@1, Hit@k, MRR and latency for four retrieval modes, plus LLM-judged answer correctness and faithfulness.
- **REST API** (FastAPI) with input validation and error handling, a small web UI, Docker, automated tests and CI.

## Architecture

```
Upload ─► extract text ─► sentence-aware chunking ─► SQL (chunks) + Chroma (embeddings)

Question ─► vector search ─┐
         └► BM25 search  ──┴► rank fusion ─► cross-encoder rerank ─► top-k passages
                                                                        │
                              answer with [n] citations ◄── LLM ◄───────┘
```

## Quick start

```bash
git clone <your-repo-url> && cd document-intelligence-rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your LLM_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the UI or http://localhost:8000/docs for the API. The embedding and reranker models download automatically on first run.

With Docker: `docker compose up --build`

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload and index a file |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document from SQL and the vector index |
| POST | `/ask` | Ask a question (`question`, optional `conversation_id`, `document_ids`, `mode`, `top_k`) |
| GET | `/conversations/{id}` | Full conversation history with sources |
| GET | `/health` | Service status |

`mode` is one of `vector`, `bm25`, `hybrid`, `hybrid_rerank` (default).

## Evaluation

Quick demo on five public research papers (Windows: double-click `run_evaluation.bat`):

```bash
python -m scripts.demo_eval
```

On your own documents, after uploading them:

```bash
python -m eval.build_testset --n 50      # one question per sampled chunk; review and prune the file
python -m eval.run_eval --answers        # writes eval/results.md
```

Each test question is generated from one chunk, which becomes the gold passage. Retrieval is scored on whether that passage is returned (Hit@k) and how high it ranks (MRR).

### Results

Measured on five public research papers (Attention Is All You Need, BERT, Sentence-BERT, RAG, LoRA): 87 pages, 518 chunks, 30 generated questions, top_k = 5, laptop CPU.

| Retrieval mode | Hit@1 | Hit@5 | MRR | Latency (ms) |
|---|---|---|---|---|
| Vector only | 0.467 | 0.667 | 0.533 | 20 |
| BM25 only | 0.633 | 0.767 | 0.682 | 8 |
| Hybrid (rank fusion) | 0.600 | 0.767 | 0.669 | 23 |
| **Hybrid + reranker** | **0.867** | **0.900** | **0.883** | 1040 |

Answers with hybrid + reranker (`openai/gpt-oss-20b` on Groq, LLM-judged): **90.0% correct, 96.7% faithful** to the retrieved context.

What the numbers show:

- Reranking is the largest gain: Hit@1 rises from 0.60 to 0.87, at the cost of about one second per query on CPU.
- BM25 beats vector search here. Research papers are full of exact terms (model names, dataset names, numbers) that keyword search matches directly.
- Plain rank fusion did not beat BM25 alone on this set; its value is in supplying the reranker with candidates from both retrievers.

Caveats: 30 questions is a small sample, the questions were generated from the gold passage (which favours keyword overlap), a few are ambiguous without their passage, and answer quality was judged by an LLM rather than by people.

## Design decisions

- **Why hybrid search?** Embeddings miss exact terms such as part numbers, names and codes; BM25 misses paraphrases. Rank fusion needs no score calibration between the two.
- **Why rerank?** A cross-encoder reads the question and passage together, so it orders candidates more accurately than embedding similarity, at the cost of extra latency. The evaluation shows that trade-off.
- **Why a custom chunker?** Chunks end on sentence boundaries and carry overlap forward, and page numbers are kept for citations.
- **Why SQL as the source of truth?** Chunk text lives in SQL; Chroma stores only vectors and ids. Deleting a document removes it from both, and BM25 is rebuilt from SQL.
- **Limitations:** scanned PDFs need OCR first; BM25 is held in memory, which suits thousands of chunks, not millions; the test set is synthetic and LLM-judged scores are approximate.

## Tests

```bash
pytest -q
```

Tests use a fake embedder and fake LLM, so they run offline in about a second.

## Project structure

```
app/        config, SQL models, ingestion, retrieval, LLM client, FastAPI routes, web UI
eval/       test-set generator, metrics, evaluation runner
tests/      unit and API tests
```
