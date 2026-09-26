import os

from src.document_loader import load_pdf


def load_all_documents(folder_path):
    """
    Load every PDF from the documents folder.
    """

    all_pages = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(
                folder_path,
                filename
            )

            print(f"Loading: {filename}")

            pages = load_pdf(pdf_path)

            all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    folder = "data/documents"

    pages = load_all_documents(folder)

    print("\nTotal pages loaded:", len(pages))

    documents = set(
        page["document"]
        for page in pages
    )

    print("\nDocuments loaded:")

    for document in documents:
        print("-", document)