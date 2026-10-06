# Docs RAG: Question-Answering over a Codebase's Documentation

A retrieval-augmented generation (RAG) system that answers technical questions
about a software library using only its own documentation — with a real
evaluation harness, not just a demo.

## Why this project

Most RAG demos stop at "it returns an answer that looks plausible." That's not
enough to trust in production, and it's not enough to demonstrate real
understanding of the failure modes RAG systems have: irrelevant retrieval,
and answers that sound right but aren't grounded in the retrieved context
(hallucination). This project treats both as measurable, not anecdotal:

- **Retrieval quality** is measured with recall@k against a hand-verified set
  of question → correct-source-file pairs.
- **Answer faithfulness** is measured with an LLM-as-judge check: does every
  claim in the generated answer actually trace back to the retrieved context?

The system is demonstrated on [FastAPI](https://github.com/tiangolo/fastapi)'s
documentation, but `src/ingest.py` works on any folder of markdown docs — point
it at a different repo's `docs/` folder to reuse it elsewhere.

## Architecture

```
docs/*.md  →  ingest.py (heading-aware chunking)
           →  vectorstore.py (fastembed embeddings + FAISS index)
           →  rag.py (retrieve top-k chunks, generate answer with Claude)
           →  evaluate.py (recall@k, faithfulness judge)
```

- **Chunking** splits on markdown headings rather than fixed character
  windows, so each chunk stays topically coherent instead of cutting a code
  example in half.
- **Embeddings** use `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX runtime,
  no PyTorch dependency) — a deliberate choice to keep the system lightweight
  without sacrificing retrieval quality.
- **Vector search** uses FAISS with cosine similarity (via L2-normalized inner
  product).
- **Generation** uses Claude, prompted to refuse to answer rather than guess
  when the retrieved context doesn't contain the answer, and to cite sources.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

Point `data/docs/` at whatever markdown docs you want to index (the FastAPI
docs are included as the working example), then build the index:

```bash
cd src
python vectorstore.py   # embeds all chunks, saves FAISS index to data/index
```

## Usage

```bash
python cli.py
> How do I add CORS to my FastAPI app?
```

## Evaluation

```bash
python evaluate.py
```

This runs the 8-question hand-labeled eval set (`data/eval_set.json`) through
both retrieval and generation, reporting:

- **Recall@5**: fraction of questions where the correct source file appeared
  in the top 5 retrieved chunks.
- **Faithfulness rate**: fraction of generated answers an LLM judge rated as
  fully supported by the retrieved context.

Each eval question's correct source was found by manually reading the docs,
not guessed — see `data/eval_set.json`.

### Results (most recent run)

| Metric | Score |
|---|---|
| Recall@5 | 1.00 (8/8) |
| Faithfulness rate | 0.88 (7/8 fully supported) |

The one `PARTIALLY_SUPPORTED` case (the background-tasks question, in the most
consistent runs) is a genuine, interesting finding rather than a retrieval
failure: the generated answer included a code example using
`background_tasks.add_task()` that wasn't actually shown in the retrieved
context — the model filled in a plausible, correct usage pattern from its
own general knowledge of FastAPI rather than quoting the retrieved text
directly. This is a real limitation worth naming honestly: the generation
step doesn't strictly refuse to elaborate beyond the retrieved context, even
though the system prompt asks it to. It's not a hallucination in the sense
of inventing false information — the code shown was correct, idiomatic
FastAPI usage — but it also isn't strictly grounded, which is exactly the
gap this evaluation is designed to catch.

## Known limitations

- The eval set is small (8 questions) — enough to catch obvious regressions,
  not enough for statistically rigorous comparison between retrieval
  strategies. A natural next step would be expanding it to 30-50 questions.
- The faithfulness judge is itself an LLM, which has its own error rate. It's
  a useful automated proxy, not a substitute for spot-checking answers by hand.
- Chunking is heading-based only; very long sections without sub-headings
  still get paragraph-split, which can occasionally separate a code example
  from its explanation.
