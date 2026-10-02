# VocalIQ server: API + web app. See DEPLOYMENT.md.
FROM python:3.12-slim

# libsndfile reads/writes WAV/FLAC/OGG/MP3; ffmpeg lets librosa decode M4A/AAC (e.g. iPhone voice memos).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# sounddevice is only used by the command-line recorder; the server doesn't need PortAudio.
RUN grep -v '^sounddevice' requirements.txt > requirements-server.txt \
 && pip install --no-cache-dir -r requirements-server.txt

COPY vocal_analyzer.py app.py ./
COPY server ./server
COPY web ./web

# Everything that must survive restarts lives on one volume.
ENV VOCALIQ_HOST=0.0.0.0 \
    VOCALIQ_PORT=8000 \
    VOCALIQ_DATA_DIR=/data \
    VOCALIQ_DB=/data/vocaliq.db \
    VOCALIQ_SECURE_COOKIES=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "app.py"]
