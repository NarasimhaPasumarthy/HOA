FROM python:3.12-slim

# Tesseract + Poppler are needed for OCR of scanned PDF pages
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY hoa_support ./hoa_support
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8083
CMD ["python", "-m", "hoa_support.server"]
