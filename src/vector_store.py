import os
import pickle

import faiss
import numpy as np

from src.embedder import create_embeddings


class VectorStore:

    def __init__(self, chunks):
        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(
            texts
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

    @classmethod
    def from_saved(
        cls,
        index,
        chunks
    ):
        store = cls.__new__(cls)

        store.index = index
        store.chunks = chunks

        return store

    def save(self, folder_path):

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        index_path = os.path.join(
            folder_path,
            "index.faiss"
        )

        chunks_path = os.path.join(
            folder_path,
            "chunks.pkl"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            chunks_path,
            "wb"
        ) as file:

            pickle.dump(
                self.chunks,
                file
            )

    @classmethod
    def load(cls, folder_path):

        index_path = os.path.join(
            folder_path,
            "index.faiss"
        )

        chunks_path = os.path.join(
            folder_path,
            "chunks.pkl"
        )

        if not (
            os.path.exists(index_path)
            and os.path.exists(chunks_path)
        ):
            return None

        index = faiss.read_index(
            index_path
        )

        with open(
            chunks_path,
            "rb"
        ) as file:

            chunks = pickle.load(file)

        return cls.from_saved(
            index,
            chunks
        )

    def search(
        self,
        query,
        top_k=3,
        document_name=None,
        similarity_threshold=0.25
    ):

        query_embedding = create_embeddings(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        total_chunks = len(
            self.chunks
        )

        if total_chunks == 0:
            return []

        # Retrieve a larger candidate pool.
        candidate_k = min(
            max(top_k * 8, 20),
            total_chunks
        )

        search_k = (
            total_chunks
            if document_name
            else candidate_k
        )

        similarities, indices = (
            self.index.search(
                query_embedding,
                search_k
            )
        )

        candidates = []

        for similarity, index in zip(
            similarities[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            if (
                document_name
                and chunk["document"]
                != document_name
            ):
                continue

            similarity = float(
                similarity
            )

            if similarity < similarity_threshold:
                continue

            candidates.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "document": chunk["document"],
                "similarity": similarity
            })

        # Remove exact duplicate chunks while
        # preserving similarity ranking.
        selected = []

        seen_text = set()

        for candidate in candidates:

            normalized_text = (
                candidate["text"]
                .strip()
                .lower()
            )

            if normalized_text in seen_text:
                continue

            seen_text.add(
                normalized_text
            )

            selected.append(
                candidate
            )

            if len(selected) >= top_k:
                break

        return selected