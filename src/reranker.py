from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print(
            f"Loading reranker: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        candidates,
        top_k=3
    ):
        if not candidates:
            return []

        pairs = [
            (
                query,
                candidate["text"]
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores
        ):
            result = candidate.copy()

            result["rerank_score"] = float(
                score
            )

            reranked.append(
                result
            )

        reranked.sort(
            key=lambda item:
                item["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]