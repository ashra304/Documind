import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from src.reranker import Reranker
from src.chunker import chunk_pages
from src.document_manager import load_all_documents
from src.vector_store import VectorStore


# Connect to OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

reranker = Reranker()

def build_rag_pipeline(
    folder_path,
    force_rebuild=False
):
    """
    Build or load the persistent RAG vector store.

    The saved index is reused only when the PDF
    collection and file metadata are unchanged.
    """

    from src.vector_store import VectorStore
    import json
    import hashlib

    index_folder = os.path.join(
        folder_path,
        "..",
        "vector_store"
    )

    index_folder = os.path.abspath(
        index_folder
    )

    os.makedirs(
        index_folder,
        exist_ok=True
    )

    index_path = os.path.join(
        index_folder,
        "index.faiss"
    )

    chunks_path = os.path.join(
        index_folder,
        "chunks.pkl"
    )

    metadata_path = os.path.join(
        index_folder,
        "documents.json"
    )

    def calculate_file_hash(file_path):
        """
        Calculate SHA-256 hash of a PDF.
        """

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb"
        ) as file:

            while True:
                data = file.read(
                    1024 * 1024
                )

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    current_documents = {}

    for filename in sorted(
        os.listdir(folder_path)
    ):

        if not filename.lower().endswith(
            ".pdf"
        ):
            continue

        file_path = os.path.join(
            folder_path,
            filename
        )

        current_documents[filename] = {
            "size": os.path.getsize(
                file_path
            ),
            "modified": os.path.getmtime(
                file_path
            ),
            "hash": calculate_file_hash(
                file_path
            )
        }

    if (
        not force_rebuild
        and os.path.exists(index_path)
        and os.path.exists(chunks_path)
        and os.path.exists(metadata_path)
    ):

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                saved_documents = json.load(
                    file
                )

            if saved_documents == current_documents:

                print(
                    "Loading existing vector index..."
                )

                vector_store = (
                    VectorStore.load(
                        index_folder
                    )
                )

                if vector_store is not None:

                    print(
                        f"Loaded "
                        f"{len(vector_store.chunks)} "
                        f"chunks from saved index."
                    )

                    return vector_store

        except Exception as error:

            print(
                "WARNING: Could not validate "
                f"saved index: {error}"
            )

    print(
        "Building new vector index..."
    )

    pages = load_all_documents(
        folder_path
    )

    print(
        f"\nLoaded {len(pages)} pages."
    )

    chunks = chunk_pages(
        pages
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    if not chunks:
        raise RuntimeError(
            "No readable content was found "
            "in the document collection."
        )

    vector_store = VectorStore(
        chunks
    )

    vector_store.save(
        index_folder
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            current_documents,
            file,
            indent=2
        )

    print(
        "Vector index saved."
    )

    return vector_store


def generate_answer(
    vector_store,
    query,
    top_k=3,
    chat_history=None,
    document_name=None
):

    if chat_history is None:
        chat_history = []

    # Build a retrieval query using recent conversation
    retrieval_query = query

    if chat_history:

        recent_history = chat_history[-4:]

        history_text = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in recent_history
            ]
        )

        retrieval_query = (
            f"Conversation context:\n"
            f"{history_text}\n\n"
            f"Current question:\n"
            f"{query}"
        )

    # Retrieve relevant chunks
    retrieved_chunks = vector_store.search(
        retrieval_query,
        top_k=max(top_k * 3, 10),
        document_name=document_name
    )

    retrieved_chunks = reranker.rerank(
        retrieval_query,
        retrieved_chunks,
        top_k=top_k 
    )

    # Build document context
    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            f"Source: {result['document']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
        )

    context = "\n\n--------------------\n\n".join(
        context_parts
    )

    # Build conversation history for the LLM
    conversation = ""

    if chat_history:

        conversation = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in chat_history[-6:]
            ]
        )

    # RAG prompt
    prompt = f"""
You are DocuMind, a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

You may use the conversation history to understand references
such as "it", "they", "this", or "those".

Do not use outside knowledge.

If the answer cannot be found in the provided document context, say:

"I could not find the answer in the provided documents."

Do not invent facts or citations.

Conversation history:

{conversation}

Document context:

{context}

Current user question:

{query}
"""

    # Generate answer
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    # Create unique sources
    sources = []

    for result in results:

        source_key = (
            result["document"],
            result["page"]
        )

        already_added = any(
            (
                source["document"],
                source["page"]
            ) == source_key
            for source in sources
        )

        if not already_added:

            sources.append({
                "document": result["document"],
                "page": result["page"],
                "similarity": result["similarity"]
            })

    return answer, sources

def summarize_document(vector_store, document_name):
    """
    Generate a summary of the entire selected document.
    Uses a map-reduce style approach:
    1. Summarize chunks in batches.
    2. Combine those summaries into a final summary.
    """

    def normalize_filename(name):
        return re.sub(r"[^a-z0-9]", "", name.lower())


    document_chunks = [
        chunk
        for chunk in vector_store.chunks
        if normalize_filename(chunk["document"])
        == normalize_filename(document_name)
    ]

    print("Requested document:", repr(document_name))
    print("Available documents:", sorted(set(
        chunk["document"] for chunk in vector_store.chunks
)))
    print("Matched chunks:", len(document_chunks))

    if not document_chunks:
        return "I could not find the selected document."

    batch_size = 10
    batch_summaries = []

    for i in range(0, len(document_chunks), batch_size):

        batch = document_chunks[i:i + batch_size]

        batch_text = "\n\n".join(
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in batch
        )

        prompt = f"""
Summarize the following section of the document.

Focus on:
- Main concepts
- Important definitions
- Key points
- Important technical details

Do not invent information.

DOCUMENT SECTION:

{batch_text}
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

        summary = response.choices[0].message.content

        batch_summaries.append(summary)

    combined_summaries = "\n\n".join(
        f"Section {i + 1}:\n{summary}"
        for i, summary in enumerate(batch_summaries)
    )

    final_prompt = f"""
Create a comprehensive summary of the document using the
section summaries below.

Include:
- Overall topic
- Major concepts
- Important definitions
- Key technical points
- Important relationships between concepts

Do not introduce information that is not present in the
section summaries.

SECTION SUMMARIES:

{combined_summaries}
"""

    final_response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ]
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":

    documents_folder = "data/documents"

    vector_store = build_rag_pipeline(
        documents_folder
    )

    query = "What is distributed shared memory?"

    answer, sources = generate_answer(
        vector_store,
        query
    )

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for source in sources:

        print(
            f"- {source['document']} "
            f"(Page {source['page']})"
        )