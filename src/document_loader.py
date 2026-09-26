from pypdf import PdfReader
import os
import pymupdf
import pytesseract


def load_pdf(pdf_path):
    document_name = os.path.basename(pdf_path)

    pages = []

    try:
        reader = PdfReader(pdf_path)
    except Exception as error:
        print(
            f"ERROR: Could not open "
            f"{document_name}: {error}"
        )
        return pages

    pdf_document = None

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        try:
            text = page.extract_text()

            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page": page_number,
                    "document": document_name
                })

            else:
                # OCR fallback for scanned/image PDFs

                if pdf_document is None:
                    pdf_document = pymupdf.open(
                        pdf_path
                    )

                pdf_page = pdf_document[
                    page_number - 1
                ]

                pixmap = pdf_page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2)
                )

                image_bytes = pixmap.tobytes(
                    "png"
                )

                from PIL import Image
                from io import BytesIO

                image = Image.open(
                    BytesIO(image_bytes)
                )

                ocr_text = pytesseract.image_to_string(
                    image
                )

                if ocr_text and ocr_text.strip():
                    pages.append({
                        "text": ocr_text.strip(),
                        "page": page_number,
                        "document": document_name
                    })

        except Exception as error:
            print(
                f"WARNING: Could not process "
                f"{document_name}, "
                f"page {page_number}: {error}"
            )

    if pdf_document is not None:
        pdf_document.close()

    return pages