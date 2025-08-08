FROM python:3.10-slim

# keep image small
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf

# minimal build deps, then clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge || true

# copy app
COPY . .

# Railway provides $PORT; default 8000 for local
EXPOSE 8000
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
