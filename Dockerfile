FROM python:3.13-slim

# Prevent Python from creating .pyc files
# and make logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Tesseract OCR
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first
# so Docker can cache this layer.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src ./src
COPY data/documents ./data/documents

# Create runtime directories
RUN mkdir -p /app/data/vector_store /app/logs

# Cloud platforms provide PORT.
# 8000 is used when running the container locally.
CMD sh -c "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"