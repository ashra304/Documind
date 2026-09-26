def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]
        document_name = page["document"]

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text.strip(),
                    "page": page_number,
                    "document": document_name
                })

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_pages = [
        {
            "text": (
                "Distributed shared memory allows multiple computers "
                "to access memory through a shared abstraction. "
            ) * 10,
            "page": 5,
            "document": "Types_distributed_systems.pdf"
        }
    ]

    chunks = chunk_pages(sample_pages)

    for chunk in chunks:

        print("\n--------------------")
        print("Document:", chunk["document"])
        print("Page:", chunk["page"])
        print("Text:", chunk["text"])