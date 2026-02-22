FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/logs

# Install the package in development mode
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "rexi.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
