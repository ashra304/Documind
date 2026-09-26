import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from src.rag import build_rag_pipeline


load_dotenv()


DOCUMENTS_FOLDER = r"data\documents"

QUESTIONS_FILE = (
    r"tests\evaluation_questions.json"
)

REPORT_FILE = (
    r"tests\evaluation_report.json"
)


client = OpenAI(
    api_key=os.getenv(
        "OPENROUTER_API_KEY"
    ),
    base_url="https://openrouter.ai/api/v1"
)


def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def retrieve_documents(
    vector_store,
    question
):
    return vector_store.search(
        question,
        top_k=3
    )


def evaluate_retrieval(
    results,
    expected_document
):
    documents = [
        result["document"]
        for result in results
    ]

    hit_at_1 = (
        len(documents) > 0
        and documents[0] == expected_document
    )

    hit_at_3 = (
        expected_document
        in documents
    )

    return hit_at_1, hit_at_3


def generate_answer(
    vector_store,
    question,
    results
):
    context_parts = []

    for result in results:

        context_parts.append(
            f"Document: "
            f"{result['document']}\n"
            f"Page: "
            f"{result['page']}\n"
            f"Content:\n"
            f"{result['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    prompt = f"""
Answer the user's question using
ONLY the provided context.

If the context does not contain enough
information, clearly say that the
information is not available.

Do not invent facts.

Context:
{context}

Question:
{question}
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

    return response.choices[0].message.content


def evaluate_answer(
    question,
    answer,
    results
):
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    evaluation_prompt = f"""
Evaluate the following RAG answer.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Return ONLY valid JSON:

{{
  "grounded": true,
  "relevance": 1,
  "coverage": 1,
  "reason": "short explanation"
}}

Rules:

grounded:
true if the answer is supported
by the retrieved context.

relevance:
1 if the answer directly addresses
the question, otherwise 0.

coverage:
1 if the answer covers the important
information available in the context,
otherwise 0.

Do not use markdown.
Do not add any text outside JSON.
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ]
    )

    raw = response.choices[0].message.content

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "grounded": False,
            "relevance": 0,
            "coverage": 0,
            "reason": "Evaluator returned invalid JSON."
        }


def main():

    print()
    print("=" * 70)
    print("DOCUMIND RAG EVALUATION")
    print("=" * 70)
    print()

    questions = load_questions()

    vector_store = build_rag_pipeline(
        DOCUMENTS_FOLDER
    )

    results = []

    hit_at_1_count = 0
    hit_at_3_count = 0

    grounded_count = 0
    relevance_count = 0
    coverage_count = 0

    similarities = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_document = (
            item["expected_document"]
        )

        print(
            f"[{number}/{len(questions)}] "
            f"{question}"
        )

        retrieved = retrieve_documents(
            vector_store,
            question
        )

        hit_at_1, hit_at_3 = (
            evaluate_retrieval(
                retrieved,
                expected_document
            )
        )

        if hit_at_1:
            hit_at_1_count += 1

        if hit_at_3:
            hit_at_3_count += 1

        for result in retrieved:
            similarities.append(
                result["similarity"]
            )

        answer = generate_answer(
            vector_store,
            question,
            retrieved
        )

        answer_evaluation = evaluate_answer(
            question,
            answer,
            retrieved
        )

        if answer_evaluation.get(
            "grounded",
            False
        ):
            grounded_count += 1

        if answer_evaluation.get(
            "relevance",
            0
        ) == 1:
            relevance_count += 1

        if answer_evaluation.get(
            "coverage",
            0
        ) == 1:
            coverage_count += 1

        result = {
            "question": question,
            "expected_document":
                expected_document,
            "retrieved_documents": [
                {
                    "document":
                        r["document"],
                    "page":
                        r["page"],
                    "similarity":
                        r["similarity"]
                }
                for r in retrieved
            ],
            "hit_at_1":
                hit_at_1,
            "hit_at_3":
                hit_at_3,
            "answer":
                answer,
            "answer_evaluation":
                answer_evaluation
        }

        results.append(result)

        print(
            f"  Hit@1: {hit_at_1}"
        )

        print(
            f"  Hit@3: {hit_at_3}"
        )

        print(
            f"  Grounded: "
            f"{answer_evaluation.get('grounded')}"
        )

        print(
            f"  Relevance: "
            f"{answer_evaluation.get('relevance')}"
        )

        print(
            f"  Coverage: "
            f"{answer_evaluation.get('coverage')}"
        )

        print()

    total = len(questions)

    average_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0
    )

    report = {
        "timestamp":
            datetime.now().isoformat(),

        "total_questions":
            total,

        "retrieval": {
            "hit_at_1":
                hit_at_1_count / total,

            "hit_at_3":
                hit_at_3_count / total,

            "average_similarity":
                average_similarity
        },

        "answer_quality": {
            "groundedness":
                grounded_count / total,

            "relevance":
                relevance_count / total,

            "coverage":
                coverage_count / total
        },

        "results":
            results
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Hit@1           : "
        f"{hit_at_1_count}/{total} "
        f"({hit_at_1_count / total * 100:.1f}%)"
    )

    print(
        f"Hit@3           : "
        f"{hit_at_3_count}/{total} "
        f"({hit_at_3_count / total * 100:.1f}%)"
    )

    print(
        f"Avg Similarity  : "
        f"{average_similarity:.3f}"
    )

    print(
        f"Groundedness    : "
        f"{grounded_count}/{total} "
        f"({grounded_count / total * 100:.1f}%)"
    )

    print(
        f"Relevance       : "
        f"{relevance_count}/{total} "
        f"({relevance_count / total * 100:.1f}%)"
    )

    print(
        f"Coverage        : "
        f"{coverage_count}/{total} "
        f"({coverage_count / total * 100:.1f}%)"
    )

    print()
    print(
        f"Report saved to: "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()