from src.rag import build_rag_pipeline


vector_store = build_rag_pipeline(
    r"data\documents"
)

print("\n" + "=" * 80)
print("ARDUINO / RASPBERRY PI CHUNKS")
print("=" * 80)

for i, chunk in enumerate(vector_store.chunks):

    text = chunk["text"].lower()

    if (
        "arduino" in text
        or "raspberry" in text
    ):
        preview = (
            chunk["text"]
            .replace("\n", " ")
            [:500]
        )

        print(f"\nCHUNK {i}")
        print(f"Document: {chunk['document']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {preview}")