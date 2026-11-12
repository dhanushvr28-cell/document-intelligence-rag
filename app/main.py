"""FastAPI service: upload documents, ask questions, keep conversation history."""
import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from .config import Settings, get_settings
from .db import Chunk, Conversation, Document, Message, make_session_factory
from .ingestion import UnsupportedFile, chunk_pages, extract_pages
from .llm import LLMClient
from .retrieval import MODES, CrossEncoderReranker, Retriever, SentenceTransformerEmbedder

log = logging.getLogger("docintel")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: int | None = None
    document_ids: list[int] | None = None
    mode: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    number: int
    chunk_id: int
    document_id: int
    filename: str
    page: int
    text: str


class AskResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[Source]
    mode: str
    latency_ms: int


def create_app(settings: Settings | None = None, embedder=None, reranker=None, llm=None) -> FastAPI:
    """Dependencies can be injected so tests run without downloading models or calling an API."""
    settings = settings or get_settings()
    sessions = make_session_factory(settings.database_url)
    retriever = Retriever(
        sessions,
        embedder or SentenceTransformerEmbedder(settings.embedding_model),
        settings.chroma_dir,
        reranker=reranker,
        reranker_factory=lambda: CrossEncoderReranker(settings.reranker_model),
    )
    llm = llm or LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)

    app = FastAPI(title="Document Intelligence Assistant", version="1.0.0")
    app.state.retriever = retriever
    app.state.sessions = sessions

    def db():
        with sessions() as session:
            yield session

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks_indexed": retriever.collection.count()}

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.post("/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...), session=Depends(db)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
        try:
            pages = extract_pages(file.filename or "", data)
        except UnsupportedFile as exc:
            raise HTTPException(415, str(exc))
        except Exception:
            raise HTTPException(422, "Could not read this file. It may be corrupted or password-protected.")
        pieces = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not pieces:
            raise HTTPException(422, "No extractable text found (scanned PDFs need OCR first).")

        doc = Document(filename=file.filename, n_chunks=len(pieces))
        doc.chunks = [Chunk(position=p.position, page=p.page, text=p.text) for p in pieces]
        session.add(doc)
        session.commit()
        try:
            retriever.add_chunks(doc.chunks)
        except Exception:
            session.delete(doc)  # keep SQL and the vector index consistent
            session.commit()
            raise
        return {"id": doc.id, "filename": doc.filename, "pages": len(pages), "chunks": doc.n_chunks}

    @app.get("/documents")
    def list_documents(session=Depends(db)):
        docs = session.scalars(select(Document).order_by(Document.id)).all()
        return [{"id": d.id, "filename": d.filename, "chunks": d.n_chunks, "created_at": d.created_at} for d in docs]

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: int, session=Depends(db)):
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "Document not found.")
        retriever.delete_document(document_id)
        session.delete(doc)
        session.commit()

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, session=Depends(db)):
        mode = body.mode or settings.default_mode
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if body.conversation_id is not None:
            convo = session.get(Conversation, body.conversation_id)
            if convo is None:
                raise HTTPException(404, "Conversation not found.")
        else:
            convo = Conversation()
            session.add(convo)
            session.flush()

        started = time.perf_counter()
        chunks = retriever.search(body.question, mode, body.top_k or settings.top_k,
                                  settings.candidate_k, body.document_ids)
        if not chunks:
            answer = "I could not find this in the uploaded documents."
        else:
            history = [{"role": m.role, "content": m.content} for m in convo.messages]
            try:
                answer = llm.answer(body.question, [c.text for c in chunks], history)
            except Exception as exc:
                log.error("LLM call failed: %s", exc)
                raise HTTPException(502, "The language model request failed. Check LLM_API_KEY and LLM_MODEL.")
        sources = [
            Source(number=i, chunk_id=c.id, document_id=c.document_id, filename=c.document.filename,
                   page=c.page, text=c.text)
            for i, c in enumerate(chunks, start=1)
        ]
        session.add(Message(conversation_id=convo.id, role="user", content=body.question))
        session.add(Message(conversation_id=convo.id, role="assistant", content=answer,
                            sources=[s.model_dump() for s in sources]))
        session.commit()
        return AskResponse(conversation_id=convo.id, answer=answer, sources=sources, mode=mode,
                           latency_ms=int((time.perf_counter() - started) * 1000))

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: int, session=Depends(db)):
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            raise HTTPException(404, "Conversation not found.")
        return {"id": convo.id, "messages": [
            {"role": m.role, "content": m.content, "sources": m.sources, "created_at": m.created_at}
            for m in convo.messages]}

    return app


def __getattr__(name: str):
    # `uvicorn app.main:app` builds the real app on first access; tests use create_app() directly.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
