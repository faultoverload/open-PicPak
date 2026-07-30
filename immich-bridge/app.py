"""
Immich Bridge — Phase 1

Pulls photos from Immich and serves them as 30,000-byte BWRY framebuffers
compatible with the open-PicPak firmware. Single-file Flask app.

Pipeline is an exact port of documentation/image-pipeline.md (BT.601 luma +
unclamped Atkinson dithering, vertical flip, 2 bpp MSB-first packing).

Endpoints:
    GET /frame.bin   -> application/octet-stream, exactly 30,000 bytes
    GET /health      -> {"ok": true, "pool_size": int, "served": int}

The PicPak firmware wakes, GETs /frame.bin over WiFi, displays the buffer,
and goes back to sleep. One frame per wake.

Air-gap note: every config value below is a placeholder; the .env.example file
documents the real variable names without leaking deployment-specific values.
"""
from __future__ import annotations

import io
import logging
import os
import random
import threading
import time
from collections import deque
from typing import Deque, Optional

import psycopg2
import psycopg2.extras
from flask import Flask, Response, jsonify
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; env vars still work
    pass


# ---------------------------------------------------------------------------
# Constants — these are panel/protocol properties, not policy. Do not change.
# ---------------------------------------------------------------------------

WIDTH = 400
HEIGHT = 300
BYTES_PER_PIXEL = 2  # 2 bpp
PIXELS_PER_BYTE = 4
FRAME_BYTES = WIDTH * HEIGHT // PIXELS_PER_BYTE  # 30,000

# Palette (logical anchors). Panel-actual colours are for preview only.
PALETTE = [
    (0, 0, 0),         # 0 = black
    (255, 255, 255),   # 1 = white
    (255, 255, 0),     # 2 = yellow
    (255, 0, 0),       # 3 = red
]

# BT.601 luma weights.
LW = (0.299, 0.587, 0.114)

# Atkinson diffusion kernel: distribute 1/8 of the error to six neighbours.
ATKINSON = ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))


# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables.
# ---------------------------------------------------------------------------

class Config:
    """Immutable config loaded once from the environment."""

    def __init__(self) -> None:
        # Immich Postgres
        self.db_host: str = os.environ.get("IMMICH_DB_HOST", "127.0.0.1")
        self.db_port: int = int(os.environ.get("IMMICH_DB_PORT", "5432"))
        self.db_name: str = os.environ.get("IMMICH_DB_NAME", "immich")
        self.db_user: str = os.environ.get("IMMICH_DB_USER", "immich")
        self.db_password: str = os.environ.get("IMMICH_DB_PASSWORD", "")

        # Immich storage layout
        self.upload_location: str = os.environ.get("UPLOAD_LOCATION", "/usr/src/app/upload")

        # Pool rotation
        self.pool_size: int = int(os.environ.get("IMMICH_POOL_SIZE", "100"))
        self.refresh_seconds: float = float(os.environ.get("IMMICH_POOL_REFRESH_SECONDS", "0"))

        # Server
        self.host: str = os.environ.get("BRIDGE_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("BRIDGE_PORT", "8090"))

        # Test/dev: skip the DB entirely and use a fixture file
        self.fixture_path: Optional[str] = os.environ.get("BRIDGE_FIXTURE_PATH")


CONFIG = Config()
log = logging.getLogger("immich-bridge")


# ---------------------------------------------------------------------------
# BWRY pipeline — port of documentation/image-pipeline.md §7
# ---------------------------------------------------------------------------

def to_bwry_frame(img: Image.Image) -> bytes:
    """Convert a PIL image into the 30,000-byte BWRY framebuffer.

    Steps (see documentation/image-pipeline.md):
        1. centre-crop to 4:3
        2. resize to 400x300
        3. quantise per pixel (BT.601 weighted distance)
        4. Atkinson error diffusion (unclamped)
        5. vertical flip
        6. pack 2 bpp, 4 px/byte, MSB-first
    """
    # 1. centre-crop to 4:3
    src_w, src_h = img.size
    target_ratio = WIDTH / HEIGHT  # 4/3
    if src_w / src_h > target_ratio:
        # too wide -> crop sides
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        # too tall -> crop top/bottom
        new_h = int(round(src_w / target_ratio))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    # 2. resize
    img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)

    # pull pixel data into float work buffers (error accumulates unclamped)
    n = WIDTH * HEIGHT
    r = [0.0] * n
    g = [0.0] * n
    b = [0.0] * n
    px = img.load()
    for i in range(n):
        pr, pg, pb = px[i % WIDTH, i // WIDTH]
        r[i] = pr
        g[i] = pg
        b[i] = pb

    # 3+4. quantise + Atkinson dither (unclamped, in code values)
    code = [0] * n
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i = y * WIDTH + x
            R, G, B = r[i], g[i], b[i]
            best = 0
            bd = float("inf")
            for k in range(4):
                pr, pg, pb = PALETTE[k]
                dr = R - pr
                dg = G - pg
                db = B - pb
                d = LW[0] * dr * dr + LW[1] * dg * dg + LW[2] * db * db
                if d < bd:
                    bd = d
                    best = k
            code[i] = best
            pr, pg, pb = PALETTE[best]
            eR = (R - pr) / 8.0
            eG = (G - pg) / 8.0
            eB = (B - pb) / 8.0
            for dx, dy in ATKINSON:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    j = ny * WIDTH + nx
                    r[j] += eR
                    g[j] += eG
                    b[j] += eB

    # 5+6. vertical flip + pack 2 bpp MSB-first
    out = bytearray(FRAME_BYTES)
    o = 0
    for y in range(HEIGHT - 1, -1, -1):
        row_base = y * WIDTH
        for x in range(0, WIDTH, 4):
            base = row_base + x
            out[o] = (
                (code[base] << 6)
                | (code[base + 1] << 4)
                | (code[base + 2] << 2)
                | code[base + 3]
            )
            o += 1
    return bytes(out)


def image_from_path(path: str) -> bytes:
    """Load an image from disk and produce a frame."""
    with Image.open(path) as img:
        return to_bwry_frame(img)


# ---------------------------------------------------------------------------
# Immich integration
# ---------------------------------------------------------------------------

# Query visible, non-deleted IMAGE assets. We don't filter on album/tag here
# because the upstream Immich schema's exact column set drifts between
# versions; isVisible + type + deletedAt is the minimal contract every
# version supports.
IMMICH_QUERY = """
SELECT
    a."id",
    a."originalFileName",
    a."originalPath"
FROM assets a
WHERE a."isVisible" = TRUE
  AND a."type" = 'IMAGE'
  AND a."deletedAt" IS NULL
ORDER BY random()
LIMIT %s;
"""


class ImmichSource:
    """Lazy, pooled, fault-tolerant Immich query source."""

    def __init__(self, config: Config):
        self.config = config
        self._conn = None

    def _connect(self):
        return psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            connect_timeout=5,
        )

    def ensure(self) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
            self._conn.autocommit = True

    def fetch(self, n: int) -> list[dict]:
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                self.ensure()
                with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(IMMICH_QUERY, (n,))
                    return list(cur.fetchall())
            except Exception as exc:
                log.warning("Immich query failed (attempt %d): %s", attempt + 1, exc)
                last_err = exc
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None
        raise RuntimeError(f"Immich query failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Pool rotation
# ---------------------------------------------------------------------------

class FramePool:
    """Thread-safe rotating deque of pre-encoded frames.

    Pop-and-serve semantics:
        - on startup: query N random photos, encode them all, shuffle, push
        - each GET: pop next, serve it
        - on exhaustion: re-query and re-shuffle, skipping the last-served
          photo id so we never show the same frame twice in a row
    """

    def __init__(self, source: ImmichSource, config: Config):
        self.source = source
        self.config = config
        self._lock = threading.Lock()
        self._frames: Deque[tuple[str, bytes]] = deque()  # (asset_id, frame)
        self._last_id: Optional[str] = None
        self.served = 0
        self.errors = 0
        self.last_refresh: Optional[float] = None

    def _encode(self, asset: dict) -> Optional[tuple[str, bytes]]:
        asset_id = asset["id"]
        rel_path = asset["originalPath"]
        full_path = os.path.join(self.config.upload_location, rel_path)
        try:
            frame = image_from_path(full_path)
        except FileNotFoundError:
            log.warning("Asset %s missing on disk at %s", asset_id, full_path)
            return None
        except Exception as exc:
            log.warning("Failed to encode asset %s: %s", asset_id, exc)
            return None
        if len(frame) != FRAME_BYTES:
            log.warning("Asset %s produced %d bytes, expected %d", asset_id, len(frame), FRAME_BYTES)
            return None
        return asset_id, frame

    def _refresh_locked(self) -> None:
        # Pull more than we need so we can skip the last-served one without
        # ending up with a tiny pool.
        over_fetch = max(self.config.pool_size + 8, self.config.pool_size * 2)
        rows = self.source.fetch(over_fetch)
        encoded: list[tuple[str, bytes]] = []
        for row in rows:
            item = self._encode(row)
            if item is None:
                continue
            if item[0] == self._last_id:
                # Avoid repeating the previous frame across rotations.
                continue
            encoded.append(item)
        random.shuffle(encoded)
        self._frames.clear()
        for item in encoded[: self.config.pool_size]:
            self._frames.append(item)
        self.last_refresh = time.time()
        log.info(
            "Pool refreshed: %d frames (queried %d, skipped last=%s)",
            len(self._frames),
            len(rows),
            self._last_id,
        )

    def ensure_filled(self) -> None:
        """Make sure the pool has at least one frame before serving."""
        with self._lock:
            if not self._frames:
                self._refresh_locked()

    def next(self) -> Optional[bytes]:
        with self._lock:
            if not self._frames:
                try:
                    self._refresh_locked()
                except Exception as exc:
                    self.errors += 1
                    log.error("Pool refresh failed: %s", exc)
                    return None
            if not self._frames:
                return None
            asset_id, frame = self._frames.popleft()
            self._last_id = asset_id
            self.served += 1
            return frame

    def stats(self) -> dict:
        with self._lock:
            return {
                "pool_size": len(self._frames),
                "served": self.served,
                "errors": self.errors,
                "last_id": self._last_id,
                "last_refresh": self.last_refresh,
            }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
source = ImmichSource(CONFIG)
pool = FramePool(source, CONFIG)


def _fixture_frame() -> Optional[bytes]:
    """Load a single fixture file and serve it forever. Used in tests."""
    path = CONFIG.fixture_path
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return image_from_path(path)


@app.get("/frame.bin")
def frame() -> Response:
    """Serve the next pre-encoded BWRY framebuffer (30,000 bytes)."""
    if CONFIG.fixture_path:
        fb = _fixture_frame()
        if fb is None:
            return Response(b"", status=503, mimetype="application/octet-stream")
    else:
        pool.ensure_filled()
        fb = pool.next()
        if fb is None:
            return Response(b"", status=503, mimetype="application/octet-stream")

    # Belt-and-braces: refuse to ever ship a short frame. PicPak expects
    # exactly 30,000 bytes; a partial buffer can lock the display.
    assert len(fb) == FRAME_BYTES, f"frame is {len(fb)} bytes, expected {FRAME_BYTES}"
    return Response(
        fb,
        status=200,
        mimetype="application/octet-stream",
        headers={
            "Content-Length": str(FRAME_BYTES),
            "Cache-Control": "no-store",
        },
    )


@app.get("/health")
def health() -> Response:
    return jsonify(
        {
            "ok": True,
            "fixture": CONFIG.fixture_path is not None,
            **pool.stats(),
        }
    )


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if CONFIG.fixture_path:
        log.info("Fixture mode: serving %s on every request", CONFIG.fixture_path)
    else:
        log.info(
            "Connecting to Immich at %s:%s/%s as %s",
            CONFIG.db_host, CONFIG.db_port, CONFIG.db_name, CONFIG.db_user,
        )
        try:
            pool.ensure_filled()
        except Exception as exc:
            # Don't crash on startup; /frame.bin will retry via the pool.
            log.error("Initial pool fill failed: %s", exc)
    log.info("Listening on %s:%d", CONFIG.host, CONFIG.port)
    # threaded=True so concurrent PicPak wakes don't queue behind each other.
    app.run(host=CONFIG.host, port=CONFIG.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()