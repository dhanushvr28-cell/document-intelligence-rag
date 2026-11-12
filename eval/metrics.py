"""Retrieval metrics. `ranked` is the list of retrieved chunk ids, best first."""


def hit_at_k(ranked: list[int], gold: int, k: int) -> float:
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[int], gold: int) -> float:
    return 1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
