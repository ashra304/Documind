from src.rag import build_rag_pipeline
from src.reranker import Reranker

import json


DOCUMENTS_FOLDER = r"data\documents"

QUESTIONS_FILE = (
    r"tests\evaluation_questions.json"
)


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print()
    print("=" * 70)
    print("DOCUMIND RERANKER EVALUATION")
    print("=" * 70)
    print()

    questions = load_questions()

    vector_store = build_rag_pipeline(
        DOCUMENTS_FOLDER
    )

    reranker = Reranker()

    baseline_hit_at_1 = 0
    reranked_hit_at_1 = 0

    baseline_hit_at_3 = 0
    reranked_hit_at_3 = 0

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected = (
            item["expected_document"]
        )

        # --------------------------------
        # Stage 1: FAISS candidate retrieval
        # --------------------------------

        candidates = vector_store.search(
            question,
            top_k=10
        )

        baseline_results = candidates[:3]

        baseline_documents = [
            result["document"]
            for result in baseline_results
        ]

        baseline_top1 = (
            len(baseline_documents) > 0
            and baseline_documents[0]
            == expected
        )

        baseline_top3 = (
            expected
            in baseline_documents
        )

        # --------------------------------
        # Stage 2: Cross-encoder reranking
        # --------------------------------

        reranked_results = reranker.rerank(
            question,
            candidates,
            top_k=3
        )

        reranked_documents = [
            result["document"]
            for result in reranked_results
        ]

        reranked_top1 = (
            len(reranked_documents) > 0
            and reranked_documents[0]
            == expected
        )

        reranked_top3 = (
            expected
            in reranked_documents
        )

        if baseline_top1:
            baseline_hit_at_1 += 1

        if baseline_top3:
            baseline_hit_at_3 += 1

        if reranked_top1:
            reranked_hit_at_1 += 1

        if reranked_top3:
            reranked_hit_at_3 += 1

        print(
            f"[{number}/{len(questions)}] "
            f"{question}"
        )

        print(
            f"  Expected: {expected}"
        )

        print(
            f"  FAISS:    "
            f"{baseline_documents}"
        )

        print(
            f"  Reranked: "
            f"{reranked_documents}"
        )

        print()

    total = len(questions)

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"FAISS Hit@1     : "
        f"{baseline_hit_at_1}/{total} "
        f"({baseline_hit_at_1 / total * 100:.1f}%)"
    )

    print(
        f"Reranked Hit@1  : "
        f"{reranked_hit_at_1}/{total} "
        f"({reranked_hit_at_1 / total * 100:.1f}%)"
    )

    print()

    print(
        f"FAISS Hit@3     : "
        f"{baseline_hit_at_3}/{total} "
        f"({baseline_hit_at_3 / total * 100:.1f}%)"
    )

    print(
        f"Reranked Hit@3  : "
        f"{reranked_hit_at_3}/{total} "
        f"({reranked_hit_at_3 / total * 100:.1f}%)"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()