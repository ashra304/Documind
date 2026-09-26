from sentence_transformers import SentenceTransformer


# Pre-trained model used to convert text into embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":
    chunks = [
        "Distributed systems consist of multiple computers working together.",
        "Parallel computing uses multiple processors to perform tasks.",
        "Distributed shared memory provides a shared memory abstraction."
    ]

    embeddings = create_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])