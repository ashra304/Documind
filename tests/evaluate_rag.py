import json
import os
import sys

from src.rag import build_rag_pipeline


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DOCUMENTS_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "documents"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "tests",
    "evaluation_questions.json"
)


def load_evaluation_questions():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def evaluate_retrieval(
    vector_store,
    questions
):
    total = len(questions)

    hit_at_1 = 0
    hit_at_3 = 0

    similarities = []

    print("\n")
    print("=" * 70)
    print("DocuMind RAG Retrieval Evaluation")
    print("=" * 70)

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        expected_document = item[
            "expected_document"
        ]

        results = vector_store.search(
            question,
            top_k=3
        )

        retrieved_documents = [
            result["document"]
            for result in results
        ]

        retrieved_similarities = [
            result["similarity"]
            for result in results
        ]

        if retrieved_similarities:
            similarities.extend(
                retrieved_similarities
            )

        hit1 = (
            len(retrieved_documents) > 0
            and
            retrieved_documents[0]
            == expected_document
        )

        hit3 = (
            expected_document
            in retrieved_documents
        )

        if hit1:
            hit_at_1 += 1

        if hit3:
            hit_at_3 += 1

        status = "PASS" if hit3 else "FAIL"

        print(f"\n[{number}/{total}] {status}")
        print(f"Question: {question}")
        print(
            f"Expected: {expected_document}"
        )
        print(
            f"Retrieved: {retrieved_documents}"
        )

        if retrieved_similarities:
            print(
                "Similarities:",
                [
                    round(score, 3)
                    for score
                    in retrieved_similarities
                ]
            )

    hit_at_1_rate = (
        hit_at_1 / total * 100
        if total
        else 0
    )

    hit_at_3_rate = (
        hit_at_3 / total * 100
        if total
        else 0
    )

    average_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0
    )

    print("\n")
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print(
        f"Total Questions : {total}"
    )

    print(
        f"Hit@1           : "
        f"{hit_at_1}/{total} "
        f"({hit_at_1_rate:.1f}%)"
    )

    print(
        f"Hit@3           : "
        f"{hit_at_3}/{total} "
        f"({hit_at_3_rate:.1f}%)"
    )

    print(
        f"Average Similarity: "
        f"{average_similarity:.3f}"
    )

    print("=" * 70)

    return {
        "total_questions": total,
        "hit_at_1": hit_at_1,
        "hit_at_1_rate": hit_at_1_rate,
        "hit_at_3": hit_at_3,
        "hit_at_3_rate": hit_at_3_rate,
        "average_similarity": average_similarity
    }


def main():

    print(
        "Building DocuMind evaluation index..."
    )

    vector_store = build_rag_pipeline(
        DOCUMENTS_FOLDER
    )

    questions = load_evaluation_questions()

    evaluate_retrieval(
        vector_store,
        questions
    )


if __name__ == "__main__":
    main()