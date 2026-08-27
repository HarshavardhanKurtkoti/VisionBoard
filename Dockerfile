# Multi-stage production Dockerfile for VisionBoard
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, Tesseract OCR, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY . .

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSERACT_CMD=/usr/bin/tesseract \
    PORT=8090

# Expose web service port
EXPOSE 8090

# Default command starts VisionBoard web server
CMD ["python", "app.py"]