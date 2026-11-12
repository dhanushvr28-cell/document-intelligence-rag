"""Answer generation through any OpenAI-compatible chat endpoint (Groq, Gemini, ...)."""
from openai import OpenAI

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I could not find this in the uploaded documents."
- Be concise and do not use outside knowledge."""


def build_messages(question: str, contexts: list[str], history: list[dict]) -> list[dict]:
    context_block = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],  # last three turns keep follow-up questions grounded
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "missing", timeout=90, max_retries=6)

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        res = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
        return (res.choices[0].message.content or "").strip()

    def answer(self, question: str, contexts: list[str], history: list[dict]) -> str:
        return self.chat(build_messages(question, contexts, history))
