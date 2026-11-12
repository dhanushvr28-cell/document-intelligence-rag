import pytest

from app.ingestion import UnsupportedFile, chunk_pages, chunk_text, extract_pages


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    for i in range(60):
        assert any(f"topic {i}." in c for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"Fact {i} is here." for i in range(40))
    chunks = chunk_text(text, size=120, overlap=40)
    assert any(set(a.split()) & set(b.split()) - {"Fact", "is", "here."} for a, b in zip(chunks, chunks[1:]))


def test_long_sentence_is_hard_split():
    chunks = chunk_text("x" * 1000, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks) and len(chunks) >= 4


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_extract_txt_and_reject_unknown():
    assert extract_pages("a.txt", b"hello world") == [(1, "hello world")]
    assert extract_pages("a.txt", b"   ") == []
    with pytest.raises(UnsupportedFile):
        extract_pages("a.exe", b"data")


def test_chunk_pages_keeps_page_numbers():
    out = chunk_pages([(1, "First page text."), (3, "Third page text.")], 200, 20)
    assert [(c.position, c.page) for c in out] == [(0, 1), (1, 3)]
