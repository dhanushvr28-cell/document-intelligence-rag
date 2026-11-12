# Evaluation results

30 questions, top_k = 5

## Retrieval

| mode | hit@1 | hit@5 | mrr | latency_ms |
|---|---|---|---|---|
| vector | 0.467 | 0.667 | 0.533 | 20 |
| bm25 | 0.633 | 0.767 | 0.682 | 8 |
| hybrid | 0.600 | 0.767 | 0.669 | 23 |
| hybrid_rerank | 0.867 | 0.900 | 0.883 | 1040 |


## Answers (openai/gpt-oss-20b, LLM-judged)

| mode | judged | answer_correctness | faithfulness |
|---|---|---|---|
| hybrid_rerank | 30 | 0.900 | 0.967 |
