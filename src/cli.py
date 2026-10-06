"""Interactive CLI: ask questions about the ingested docs."""

import sys
from pathlib import Path

from rag import answer_question
from vectorstore import VectorStore


def main():
    print("Loading index...")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    store = VectorStore()
    store.load(str(data_dir / "index"))
    print("Ready. Ask a question (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            sys.exit(0)

        if not question:
            continue

        result = answer_question(question, store)
        print(f"\n{result['answer']}\n")
        unique_sources = list(dict.fromkeys(s for s, _ in result["retrieved"]))
        print(f"(sources: {', '.join(unique_sources)})\n")


if __name__ == "__main__":
    main()
