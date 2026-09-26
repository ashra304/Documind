import logging
import os
import shutil
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag import (
    build_rag_pipeline,
    generate_answer,
    summarize_document,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("documind.api")


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="DocuMind API",
    description="Multi-document RAG intelligence platform.",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

# Local development is always allowed.
# A deployed frontend URL can be supplied through:
#
# FRONTEND_URL=https://your-frontend-domain.com
#
# Multiple URLs can be supplied separated by commas.

frontend_url = os.getenv("FRONTEND_URL", "").strip()

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if frontend_url:
    allowed_origins.extend(
        origin.strip()
        for origin in frontend_url.split(",")
        if origin.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

documents_folder = os.path.join(
    BASE_DIR,
    "data",
    "documents",
)

os.makedirs(
    documents_folder,
    exist_ok=True,
)


# --------------------------------------------------
# RAG initialization
# --------------------------------------------------

logger.info(
    "DocuMind documents folder: %s",
    documents_folder,
)

vector_store = build_rag_pipeline(
    documents_folder
)


# --------------------------------------------------
# Conversation memory
# --------------------------------------------------

chat_sessions = {}


# --------------------------------------------------
# Request models
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    document_name: Optional[str] = None


class SummarizeRequest(BaseModel):
    document_name: str


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "DocuMind",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    if vector_store is None:
        return {
            "status": "unhealthy",
            "vector_store": False,
        }

    return {
        "status": "healthy",
        "vector_store": True,
        "chunks": len(vector_store.chunks),
        "documents": len(
            {
                chunk["document"]
                for chunk in vector_store.chunks
            }
        ),
    }


# --------------------------------------------------
# Documents
# --------------------------------------------------

@app.get("/documents")
def get_documents():
    documents = []

    for filename in sorted(
        os.listdir(documents_folder)
    ):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(
            documents_folder,
            filename,
        )

        documents.append({
            "filename": filename,
            "size": os.path.getsize(file_path),
        })

    return {
        "documents": documents,
    }


# --------------------------------------------------
# Query
# --------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):
    global vector_store

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    session_id = (
        request.session_id
        or "default"
    )

    history = chat_sessions.get(
        session_id,
        [],
    )

    try:
        result = generate_answer(
            vector_store,
            question,
            chat_history=history,
            document_name=request.document_name,
        )

        # Store conversation for subsequent questions.
        history.append({
            "role": "user",
            "content": question,
        })

        history.append({
            "role": "assistant",
            "content": result.get(
                "answer",
                "",
            ),
        })

        # Keep memory bounded.
        # Last 10 messages = 5 exchanges.
        chat_sessions[session_id] = (
            history[-10:]
        )

        return result

    except Exception as error:
        logger.exception(
            "Query failed: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        )


# --------------------------------------------------
# Upload
# --------------------------------------------------

@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
):
    global vector_store

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = os.path.basename(
        file.filename,
    )

    if filename != file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    destination = os.path.join(
        documents_folder,
        filename,
    )

    # Prevent accidentally deleting an existing
    # document if replacement indexing fails.
    backup_path = destination + ".backup"

    try:
        if os.path.exists(destination):
            shutil.copy2(
                destination,
                backup_path,
            )

        with open(
            destination,
            "wb",
        ) as output:
            shutil.copyfileobj(
                file.file,
                output,
            )

        logger.info(
            "Uploaded document: %s",
            filename,
        )

        vector_store = build_rag_pipeline(
            documents_folder,
            force_rebuild=False,
        )

        if os.path.exists(backup_path):
            os.remove(backup_path)

        return {
            "message": "Document uploaded successfully.",
            "filename": filename,
        }

    except Exception as error:
        logger.exception(
            "Upload failed: %s",
            error,
        )

        # Restore previous version if one existed.
        if os.path.exists(backup_path):
            if os.path.exists(destination):
                os.remove(destination)

            os.replace(
                backup_path,
                destination,
            )

        elif os.path.exists(destination):
            os.remove(destination)

        raise HTTPException(
            status_code=500,
            detail="Document upload or indexing failed.",
        )


# --------------------------------------------------
# Delete
# --------------------------------------------------

@app.delete("/documents/{document_name}")
def delete_document(
    document_name: str,
):
    global vector_store

    filename = os.path.basename(
        document_name,
    )

    if filename != document_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid document name.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents can be deleted.",
        )

    file_path = os.path.join(
        documents_folder,
        filename,
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        os.remove(file_path)

        logger.info(
            "Deleted document: %s",
            filename,
        )

        vector_store = build_rag_pipeline(
            documents_folder,
            force_rebuild=False,
        )

        return {
            "message": "Document deleted successfully.",
            "filename": filename,
        }

    except Exception as error:
        logger.exception(
            "Delete failed: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document deletion or "
                "re-indexing failed."
            ),
        )


# --------------------------------------------------
# Summarization
# --------------------------------------------------

@app.post("/summarize")
def summarize(
    request: SummarizeRequest,
):
    global vector_store

    filename = os.path.basename(
        request.document_name,
    )

    if filename != request.document_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid document name.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported.",
        )

    file_path = os.path.join(
        documents_folder,
        filename,
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        summary = summarize_document(
            vector_store,
            filename,
        )

        return {
            "document": filename,
            "summary": summary,
        }

    except Exception as error:
        logger.exception(
            "Summarization failed: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to summarize document.",
        )