FROM python:3.11-slim-bookworm

# opencv-python needs these system libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker caches this layer until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

# Settings come from environment variables (docker run --env-file .env, or
# Render's environment settings), never from a .env baked into the image.
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render sets $PORT; locally it falls back to 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
