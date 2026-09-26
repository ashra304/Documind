import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.rag import build_rag_pipeline
from src.vector_store import VectorStore


load_dotenv()


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


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def load_questions():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_answer(
    question,
    answer,
    context
):

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Evaluate whether the generated answer is supported by
the retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return ONLY valid JSON in this exact format:

{{
    "grounded": true,
    "relevance": 1,
    "coverage": 1,
    "reason": "short explanation"
}}

Rules:

grounded:
- true if the answer is supported by the retrieved context.
- false if the answer contains information that is not supported.

relevance:
- 1 if the answer directly addresses the question.
- 0 if it does not.

coverage:
- 1 if the answer adequately addresses the question.
- 0 if important information is missing.

Do not use outside knowledge.
Judge only using the retrieved context.
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    # Remove markdown code fences if the model adds them
    content = content.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        return {
            "grounded": False,
            "relevance": 0,
            "coverage": 0,
            "reason":
                "Evaluator returned invalid JSON."
        }


def main():

    print(
        "\nBuilding DocuMind evaluation index..."
    )

    vector_store = build_rag_pipeline(
        DOCUMENTS_FOLDER
    )

    questions = load_questions()

    total = len(questions)

    grounded_count = 0
    relevant_count = 0
    covered_count = 0

    print("\n")
    print("=" * 70)
    print("DocuMind Answer Quality Evaluation")
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

        if not results:

            print(
                f"\n[{number}/{total}] "
                "NO RETRIEVAL"
            )

            continue

        context = "\n\n".join(
            f"""
Document: {result['document']}
Page: {result['page']}
Content:
{result['text']}
"""
            for result in results
        )

        # Generate the answer using the same
        # LLM used by the actual application.
        answer_prompt = f"""
Answer the question using ONLY the
retrieved context below.

If the context does not contain enough
information to answer the question,
say that the information is not available
in the retrieved documents.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""

        answer_response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": answer_prompt
                }
            ]
        )

        answer = (
            answer_response.choices[0]
            .message
            .content
            .strip()
        )

        evaluation = evaluate_answer(
            question,
            answer,
            context
        )

        grounded = bool(
            evaluation.get(
                "grounded",
                False
            )
        )

        relevance = (
            evaluation.get(
                "relevance",
                0
            ) == 1
        )

        coverage = (
            evaluation.get(
                "coverage",
                0
            ) == 1
        )

        if grounded:
            grounded_count += 1

        if relevance:
            relevant_count += 1

        if coverage:
            covered_count += 1

        status = (
            "PASS"
            if grounded
            and relevance
            and coverage
            else "REVIEW"
        )

        print(
            f"\n[{number}/{total}] "
            f"{status}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Expected document: "
            f"{expected_document}"
        )

        print(
            f"Answer: {answer}"
        )

        print(
            f"Grounded: {grounded}"
        )

        print(
            f"Relevant: {relevance}"
        )

        print(
            f"Coverage: {coverage}"
        )

        print(
            f"Reason: "
            f"{evaluation.get('reason', '')}"
        )

    grounded_rate = (
        grounded_count
        / total
        * 100
        if total
        else 0
    )

    relevance_rate = (
        relevant_count
        / total
        * 100
        if total
        else 0
    )

    coverage_rate = (
        covered_count
        / total
        * 100
        if total
        else 0
    )

    print("\n")
    print("=" * 70)
    print("Answer Quality Results")
    print("=" * 70)

    print(
        f"Total Questions : {total}"
    )

    print(
        f"Groundedness    : "
        f"{grounded_count}/{total} "
        f"({grounded_rate:.1f}%)"
    )

    print(
        f"Relevance       : "
        f"{relevant_count}/{total} "
        f"({relevance_rate:.1f}%)"
    )

    print(
        f"Coverage        : "
        f"{covered_count}/{total} "
        f"({coverage_rate:.1f}%)"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()