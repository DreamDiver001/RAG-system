"""Answer a question about the docs by retrieving relevant chunks and asking Claude."""

from __future__ import annotations

import os

import anthropic

from ingest import Chunk
from vectorstore import VectorStore

SYSTEM_PROMPT = """You answer questions about a software library using only the provided \
documentation excerpts. If the excerpts don't contain the answer, say so plainly instead \
of guessing. Cite the source file(s) you used in your answer."""


def format_context(results: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in results:
        parts.append(f"[source: {chunk.source}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, store: VectorStore, top_k: int = 5) -> dict:
    """Retrieve relevant chunks and generate an answer grounded in them.

    Returns a dict with the answer text and the retrieved chunks, so callers
    (like the evaluation harness) can inspect what was actually retrieved.
    """
    results = store.search(question, top_k=top_k)
    context = format_context(results)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return {
        "question": question,
        "answer": response.content[0].text,
        "context": context,  # the actual text fed to the model, for faithfulness checking
        "retrieved": [(c.source, score) for c, score in results],
    }


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    store = VectorStore()
    store.load(str(data_dir / "index"))
    result = answer_question("How do I define a path parameter in FastAPI?", store)
    print("Answer:", result["answer"])
    print("\nRetrieved from:", [r[0] for r in result["retrieved"]])
