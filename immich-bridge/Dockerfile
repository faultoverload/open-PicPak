# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

# Pillow's quantisation + libjpeg via system packages. Keeps the image small
# and avoids the Python wheels' C-extension surprises across architectures.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
        libpng16-16 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first so source changes don't bust the install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Non-root by default; the bridge only needs to read photos from the share.
RUN useradd --system --uid 1000 --create-home --home /home/bridge bridge \
 && chown -R bridge:bridge /app
USER bridge

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/health || exit 1

# gunicorn: 2 workers, threadsafe. The /frame.bin handler is short and CPU-
# bound (Atkinson dithering); multiple threads prevent a slow encode from
# stalling concurrent PicPak wakes. Threaded workers also let the Flask
# dev server stay viable for `docker run --rm` smoke tests.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8090", \
     "--workers", "1", \
     "--threads", "4", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]