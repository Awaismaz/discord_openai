# Use a small Python image
FROM python:3.13-slim

# Prevents Python from writing .pyc files & enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages:
# - tesseract-ocr (OCR engine)
# - libtesseract-dev (OCR headers)
# - libgl1 (needed by PyMuPDF on some hosts)
# - curl/ghostscript are optional but useful for debugging PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libgl1 curl ghostscript \
 && rm -rf /var/lib/apt/lists/*

# Set Tesseract path explicitly (our code respects this)
ENV TESSERACT_CMD=/usr/bin/tesseract

# Workdir
WORKDIR /app

# Install Python deps first (better Docker layer cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Health check (optional): print tesseract version at build time
RUN tesseract --version

# Start the bot
CMD ["python", "main.py"]
