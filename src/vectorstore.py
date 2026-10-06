"""Embed chunks and search them with a FAISS index."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm

from ingest import Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, good quality/size tradeoff


class VectorStore:
    def __init__(self, threads: int = 2):
        # Cap ONNX runtime threads so embedding doesn't saturate every CPU
        # core -- important on lower-end machines, where the OS can become
        # unresponsive if this is left at its default (all cores).
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL, threads=threads)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], batch_size: int = 32) -> None:
        """Embed all chunks (in batches, with progress output) and build a FAISS index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = np.array(
            list(tqdm(self.model.embed(texts, batch_size=batch_size), total=len(texts))),
            dtype="float32",
        )
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks most similar to the query, with similarity scores."""
        query_vec = np.array(list(self.model.embed([query])), dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(dir_path) / "index.faiss"))
        with open(Path(dir_path) / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, dir_path: str) -> None:
        self.index = faiss.read_index(str(Path(dir_path) / "index.faiss"))
        with open(Path(dir_path) / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)


if __name__ == "__main__":
    from ingest import build_corpus

    data_dir = Path(__file__).resolve().parent.parent / "data"
    chunks = build_corpus(str(data_dir / "docs"))
    store = VectorStore()
    print(f"Embedding {len(chunks)} chunks...")
    store.build(chunks)
    store.save(str(data_dir / "index"))
    print("Index saved to data/index")

    print("\n--- test query ---")
    results = store.search("How do I define a path parameter?", top_k=3)
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk.source}: {chunk.text[:100]}")
