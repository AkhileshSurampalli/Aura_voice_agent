# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing (pydub/ffmpeg) and build tooling for any
# packages that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "main"]
