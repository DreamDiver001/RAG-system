"""Evaluate the RAG system on two axes:

1. Retrieval quality: recall@k -- for questions with a known correct source
   file, does that file show up in the top-k retrieved chunks?
2. Answer faithfulness: for generated answers, does an LLM judge find the
   answer fully supported by the retrieved context (a proxy for hallucination)?

A hand-labeled eval set lives in data/eval_set.json. Each item has a question
and the doc file that actually answers it (found manually by reading the docs).
"""

from __future__ import annotations

import json
import os
import re
import time

import anthropic

from rag import answer_question
from vectorstore import VectorStore

JUDGE_PROMPT = """You are checking whether an answer is fully supported by the given context.

Context:
{context}

Answer to check:
{answer}

Is every factual claim in the answer supported by the context? Respond with ONLY a JSON object, \
no other text, in exactly this form:
{{"verdict": "SUPPORTED", "reason": "one sentence explaining why"}}
The "verdict" value must be exactly one of: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED."""


def load_eval_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_retrieval(store: VectorStore, eval_set: list[dict], top_k: int = 5) -> dict:
    """Recall@k: fraction of questions where the labeled correct source file
    appears somewhere in the top-k retrieved chunks."""
    hits = 0
    per_question = []
    for item in eval_set:
        results = store.search(item["question"], top_k=top_k)
        # Normalize path separators: on Windows, chunk.source uses backslashes,
        # while eval_set.json (and this codebase generally) uses forward slashes.
        retrieved_sources = {c.source.replace("\\", "/") for c, _ in results}
        hit = item["expected_source"] in retrieved_sources
        hits += hit
        per_question.append({"question": item["question"], "hit": hit})

    return {
        "recall_at_k": hits / len(eval_set),
        "k": top_k,
        "per_question": per_question,
    }


def judge_faithfulness(question: str, context: str, answer: str, client: anthropic.Anthropic) -> tuple[str, str]:
    """Return (verdict, reason)."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
    )
    raw = response.content[0].text.strip()
    # Some responses wrap the JSON in a markdown code fence despite instructions
    # not to -- strip it before attempting to parse.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(raw)
        return parsed["verdict"].strip(), parsed.get("reason", "").strip()
    except (json.JSONDecodeError, KeyError, AttributeError):
        # Fallback if the model didn't return clean JSON: find a known label anywhere in the text.
        match = re.search(r"UNSUPPORTED|PARTIALLY_SUPPORTED|SUPPORTED", raw)
        verdict = match.group(0) if match else (raw.split()[0] if raw else raw)
        return verdict, raw


def evaluate_faithfulness(store: VectorStore, eval_set: list[dict], top_k: int = 5) -> dict:
    """For each question, generate an answer, then have a judge model check
    whether the answer's claims are actually backed by the retrieved context."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
    for item in eval_set:
        outcome = answer_question(item["question"], store, top_k=top_k)
        verdict, reason = judge_faithfulness(item["question"], outcome["context"], outcome["answer"], client)
        results.append({
            "question": item["question"],
            "verdict": verdict,
            "reason": reason,
            "answer": outcome["answer"],
        })
        time.sleep(0.5)  # be polite to the API

    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    return {
        "faithfulness_rate": supported / len(eval_set),
        "per_question": results,
    }


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    store = VectorStore()
    store.load(str(data_dir / "index"))
    eval_set = load_eval_set(str(data_dir / "eval_set.json"))

    print("=== Retrieval evaluation ===")
    retrieval_results = evaluate_retrieval(store, eval_set)
    print(f"Recall@{retrieval_results['k']}: {retrieval_results['recall_at_k']:.2f}")
    for q in retrieval_results["per_question"]:
        status = "HIT" if q["hit"] else "MISS"
        print(f"  [{status}] {q['question']}")

    print("\n=== Faithfulness evaluation ===")
    faithfulness_results = evaluate_faithfulness(store, eval_set)
    print(f"Faithfulness rate: {faithfulness_results['faithfulness_rate']:.2f}")
    for q in faithfulness_results["per_question"]:
        print(f"  [{q['verdict']}] {q['question']}")
        if q["reason"]:
            print(f"      reason: {q['reason']}")
