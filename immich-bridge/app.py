"""
Immich Bridge — API-native (no DB)

Pulls photos from Immich via REST API and serves them as 30,000-byte BWRY
framebuffers compatible with open-PicPak firmware. Single-file Flask app.

Pipeline: /api/search/random → /api/assets/:id/original → resize → dither → serve

Endpoints:
    GET /frame.bin   → application/octet-stream, exactly 30,000 bytes
    GET /frame.png   → PNG preview (debug)
    GET /health      → {"ok": true, "served": int, ...}
    GET /info        → metadata for current frame
    GET /next        → force-advance
    GET /pool        → list pool contents
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from datetime import datetime
from typing import Deque, Optional

from flask import Flask, Response, jsonify
from PIL import Image

# Register HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAVE_HEIF = True
except ImportError:
    HAVE_HEIF = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants — panel/protocol properties. Do not change.
# ---------------------------------------------------------------------------

WIDTH = 400
HEIGHT = 300
BYTES_PER_PIXEL = 2
PIXELS_PER_BYTE = 4
FRAME_BYTES = WIDTH * HEIGHT // PIXELS_PER_BYTE  # 30,000

PALETTE = [
    (0, 0, 0),         # black
    (255, 255, 255),   # white
    (255, 255, 0),     # yellow
    (255, 0, 0),       # red
]

LW = (0.299, 0.587, 0.114)
ATKINSON = ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))
FLOYD_STEINBERG = ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1))


# ---------------------------------------------------------------------------
# Oklab perceptual colour space
# ---------------------------------------------------------------------------

def _srgb_to_linear(value: float) -> float:
    value /= 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _linear_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m, s = l_ ** (1.0 / 3.0), m ** (1.0 / 3.0), s ** (1.0 / 3.0)
    return (
        0.2104542553 * l_ + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l_ - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l_ + 0.7827717662 * m - 0.8086757660 * s,
    )


def _palette_oklab() -> list[tuple[float, float, float]]:
    return [
        _linear_to_oklab(*[_srgb_to_linear(c) for c in color])
        for color in PALETTE
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    def __init__(self) -> None:
        # Immich API
        self.api_url: str = os.environ.get(
            "IMMICH_API_URL", "https://immich.gateslab.win"
        ).rstrip("/")
        self.api_key: str = os.environ.get("IMMICH_API_KEY", "")

        # Pool rotation
        self.pool_size: int = int(os.environ.get("IMMICH_POOL_SIZE", "100"))

        # Optional album filter (requires album.read API key permission)
        self.album_id: Optional[str] = os.environ.get("IMMICH_ALBUM_ID") or None

        # Dithering mode
        self.dither_mode: str = os.environ.get("DITHER_MODE", "perceptual").lower()
        if self.dither_mode not in {"perceptual", "app"}:
            raise ValueError("DITHER_MODE must be 'perceptual' or 'app'")

        # LRU framebuffer cache
        self.cache_size: int = max(1, int(os.environ.get("CACHE_SIZE", "20")))

        # Server
        self.host: str = os.environ.get("BRIDGE_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("BRIDGE_PORT", "8090"))

        # Test/dev: skip API and use a fixture file
        self.fixture_path: Optional[str] = os.environ.get("BRIDGE_FIXTURE_PATH")


CONFIG = Config()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("immich-bridge")
log.info("immich-bridge starting (dither=%s, pool=%d)", CONFIG.dither_mode, CONFIG.pool_size)


# ---------------------------------------------------------------------------
# BWRY pipeline (unchanged from Phase 2)
# ---------------------------------------------------------------------------

def to_bwry_frame(img: Image.Image, mode: str | None = None) -> bytes:
    if mode is None:
        mode = CONFIG.dither_mode

    # Centre-crop to 4:3
    src_w, src_h = img.size
    target_ratio = WIDTH / HEIGHT
    if src_w / src_h > target_ratio:
        new_w = int(round(src_h * target_ratio))
        img = img.crop(((src_w - new_w) // 2, 0, (src_w - new_w) // 2 + new_w, src_h))
    else:
        new_h = int(round(src_w / target_ratio))
        img = img.crop((0, (src_h - new_h) // 2, src_w, (src_h - new_h) // 2 + new_h))

    # Resize
    img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)

    # Dither
    code = _dither_perceptual(img) if mode == "perceptual" else _dither_atkinson(img)

    # Vertical flip + pack 2 bpp MSB-first
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


def _dither_atkinson(img: Image.Image) -> list[int]:
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

    code = [0] * n
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i = y * WIDTH + x
            Rc, Gc, Bc = r[i], g[i], b[i]
            best = 0
            bd = float("inf")
            for k in range(4):
                pr, pg, pb = PALETTE[k]
                dr = Rc - pr
                dg = Gc - pg
                db = Bc - pb
                d = LW[0] * dr * dr + LW[1] * dg * dg + LW[2] * db * db
                if d < bd:
                    bd = d
                    best = k
            code[i] = best
            pr, pg, pb = PALETTE[best]
            eR = (Rc - pr) / 8.0
            eG = (Gc - pg) / 8.0
            eB = (Bc - pb) / 8.0
            for dx, dy in ATKINSON:
                nx, ny = x + dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    j = ny * WIDTH + nx
                    r[j] += eR
                    g[j] += eG
                    b[j] += eB
    return code


def _dither_perceptual(img: Image.Image) -> list[int]:
    palette_oklab = _palette_oklab()
    n = WIDTH * HEIGHT
    L = [0.0] * n
    A = [0.0] * n
    B = [0.0] * n
    px = img.load()
    for i in range(n):
        pr, pg, pb = px[i % WIDTH, i // WIDTH]
        L[i], A[i], B[i] = _linear_to_oklab(
            _srgb_to_linear(pr),
            _srgb_to_linear(pg),
            _srgb_to_linear(pb),
        )

    code = [0] * n
    for y in range(HEIGHT):
        reverse = y & 1
        x_range = range(WIDTH - 1, -1, -1) if reverse else range(WIDTH)
        for x in x_range:
            i = y * WIDTH + x
            l_, a, b_ = L[i], A[i], B[i]
            best = 0
            bd = float("inf")
            for k in range(4):
                pl, pa, pb = palette_oklab[k]
                dl = l_ - pl
                da = a - pa
                db = b_ - pb
                d = dl * dl + 1.35 * da * da + 1.35 * db * db
                if d < bd:
                    bd = d
                    best = k
            code[i] = best
            pl, pa, pb = palette_oklab[best]
            eL = (l_ - pl) / 16.0
            eA = (a - pa) / 16.0
            eB = (b_ - pb) / 16.0
            for dx, dy, weight in FLOYD_STEINBERG:
                actual_dx = -dx if reverse else dx
                nx, ny = x + actual_dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    j = ny * WIDTH + nx
                    L[j] += eL * weight
                    A[j] += eA * weight
                    B[j] += eB * weight
    return code


# ---------------------------------------------------------------------------
# Immich API integration
# ---------------------------------------------------------------------------

def _api_request(method: str, path: str, body: dict | None = None) -> bytes:
    """Make an authenticated request to the Immich API."""
    url = f"{CONFIG.api_url}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "x-api-key": CONFIG.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        log.error("API %s %s → HTTP %d: %s", method, path, exc.code, body_text[:200])
        raise
    except Exception:
        log.exception("API %s %s failed", method, path)
        raise


def api_health() -> dict:
    """Check API connectivity."""
    try:
        data = json.loads(_api_request("GET", "/api/server/version"))
        return {"connected": True, "version": f"{data['major']}.{data['minor']}.{data['patch']}"}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


class ImmichSource:
    """Fetches random image assets from Immich API."""

    def __init__(self, config: Config):
        self.config = config

    def fetch(self, n: int) -> list[dict]:
        """Fetch up to n random timeline-visible IMAGE assets."""
        body: dict = {"size": n, "type": "IMAGE"}
        try:
            data = json.loads(_api_request("POST", "/api/search/random", body))
            return data if isinstance(data, list) else []
        except Exception:
            log.exception("search/random failed")
            return []


def image_from_asset(asset: dict) -> bytes:
    """Download asset original from Immich API and produce a BWRY frame."""
    asset_id = asset["id"]
    img_bytes = _api_request("GET", f"/api/assets/{asset_id}/original")
    with Image.open(io.BytesIO(img_bytes)) as img:
        return to_bwry_frame(img)


# ---------------------------------------------------------------------------
# Frame Pool
# ---------------------------------------------------------------------------

class FramePool:
    """Thread-safe rotating deque of pre-encoded frames with LRU cache."""

    def __init__(self, source: ImmichSource, config: Config):
        self.source = source
        self.config = config
        self._lock = threading.RLock()
        self._frames: Deque[tuple[str, bytes]] = deque()
        self._last_id: Optional[str] = None
        self._last_metadata: Optional[dict] = None
        self._current_metadata: Optional[dict] = None
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._pool_meta: list[dict] = []
        self.served = 0
        self.errors = 0
        self.last_refresh: Optional[float] = None

    def _encode(self, asset: dict) -> Optional[tuple[str, bytes]]:
        """Encode a single asset, using LRU cache if available."""
        asset_id = asset["id"]

        with self._lock:
            if asset_id in self._cache:
                self._cache.move_to_end(asset_id)
                return asset_id, self._cache[asset_id]

        try:
            frame = image_from_asset(asset)
        except Exception as exc:
            log.warning("Failed to encode asset %s: %s", asset_id, exc)
            return None

        if len(frame) != FRAME_BYTES:
            log.warning("Asset %s produced %d bytes, expected %d", asset_id, len(frame), FRAME_BYTES)
            return None

        with self._lock:
            self._cache[asset_id] = frame
            if len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)
            self._cache.move_to_end(asset_id)

        return asset_id, frame

    def _refresh_locked(self) -> None:
        over_fetch = max(self.config.pool_size + 8, self.config.pool_size * 2)
        rows = self.source.fetch(over_fetch)
        encoded: list[tuple[str, bytes]] = []
        meta: list[dict] = []
        for row in rows:
            item = self._encode(row)
            if item is None:
                continue
            if item[0] == self._last_id:
                continue
            encoded.append(item)
            meta.append({
                "id": row.get("id"),
                "filename": row.get("originalFileName"),
                "date": row.get("fileCreatedAt"),
                "album": None,  # Album info not in search/random response
            })

        if not encoded and rows:
            # All fetched assets failed to encode — try one more batch
            rows = self.source.fetch(over_fetch)
            for row in rows:
                item = self._encode(row)
                if item is None or item[0] == self._last_id:
                    continue
                encoded.append(item)
                meta.append({
                    "id": row.get("id"),
                    "filename": row.get("originalFileName"),
                    "date": row.get("fileCreatedAt"),
                    "album": None,
                })

        random.shuffle(encoded)
        self._frames = deque(encoded)
        self._pool_meta = meta
        self.last_refresh = time.time()
        log.info("Pool refreshed: %d/%d frames", len(encoded), self.config.pool_size)

    def ensure(self) -> None:
        with self._lock:
            if not self._frames:
                self._refresh_locked()

    def next_frame(self) -> Optional[bytes]:
        with self._lock:
            if not self._frames:
                self._refresh_locked()
            if not self._frames:
                log.error("Pool empty after refresh")
                return None

            asset_id, frame = self._frames.popleft()
            self._last_id = asset_id
            # Find metadata for this asset
            for m in self._pool_meta:
                if m.get("id") == asset_id:
                    self._last_metadata = dict(m)
                    break
            self.served += 1
            return frame

    def get_info(self) -> dict:
        with self._lock:
            return {
                "last_id": self._last_id,
                "pool_size": len(self._frames),
                "dither_mode": self.config.dither_mode,
                "served": self.served,
                "errors": self.errors,
                "album_id": self.config.album_id,
            }

    def get_pool(self) -> list[dict]:
        with self._lock:
            return list(self._pool_meta)

    def get_last_frame(self) -> Optional[bytes]:
        with self._lock:
            if self._last_id and self._last_id in self._cache:
                return self._cache[self._last_id]
            return None


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


def _get_pool() -> FramePool:
    p = getattr(app, "_pool", None)
    if p is None:
        source = ImmichSource(CONFIG)
        p = FramePool(source, CONFIG)
        p.ensure()
        app._pool = p
        log.info("Pool lazily initialized")
    return p


@app.route("/frame.bin")
def frame_bin():
    pool = _get_pool()
    if pool is None:
        return Response(b"", status=503, content_type="application/octet-stream")
    frame = pool.next_frame()
    if frame is None:
        return Response(b"", status=503, content_type="application/octet-stream")
    return Response(frame, content_type="application/octet-stream")


@app.route("/frame.png")
def frame_png():
    pool = _get_pool()
    if pool is None:
        return Response(b"", status=503)
    frame = pool.get_last_frame()
    if frame is None:
        return Response(b"", status=503)
    # Reconstruct a 4-colour palette PNG from the BWRY buffer
    img = Image.new("P", (WIDTH, HEIGHT))
    img.putpalette([
        0, 0, 0,        # black
        255, 255, 255,  # white
        255, 255, 0,    # yellow
        255, 0, 0,      # red
    ] + [0] * (252 * 3))
    pixels = []
    for y in range(HEIGHT):
        row_base = (HEIGHT - 1 - y) * WIDTH  # undo vertical flip
        for x in range(0, WIDTH, 4):
            byte_idx = (y * (WIDTH // 4)) + (x // 4)
            b = frame[byte_idx]
            pixels.extend([
                (b >> 6) & 3,
                (b >> 4) & 3,
                (b >> 2) & 3,
                b & 3,
            ])
    img.putdata(pixels[:WIDTH * HEIGHT])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return Response(buf.read(), content_type="image/png")


@app.route("/health")
def health():
    pool = _get_pool()
    api_status = api_health()
    return jsonify({
        "ok": api_status["connected"] and pool is not None,
        "api": api_status,
        "pool_size": len(pool._frames) if pool else 0,
        "served": pool.served if pool else 0,
        "dither_mode": CONFIG.dither_mode,
        "fixture": CONFIG.fixture_path is not None,
    })


@app.route("/info")
def info():
    pool = _get_pool()
    if pool is None:
        return jsonify({"error": "not initialized"}), 503
    return jsonify(pool.get_info())


@app.route("/next")
def next_frame():
    pool = _get_pool()
    if pool is None:
        return jsonify({"error": "not initialized"}), 503
    pool.ensure()
    return jsonify(pool.get_info())


@app.route("/pool")
def pool_list():
    pool = _get_pool()
    if pool is None:
        return jsonify([])
    return jsonify(pool.get_pool())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if CONFIG.fixture_path:
        log.info("Fixture mode: %s", CONFIG.fixture_path)
        img = Image.open(CONFIG.fixture_path)
        frame = to_bwry_frame(img)
        with open("/tmp/fixture-frame.bin", "wb") as f:
            f.write(frame)
        log.info("Fixture frame written to /tmp/fixture-frame.bin (%d bytes)", len(frame))
        return

    if not CONFIG.api_key:
        log.fatal("IMMICH_API_KEY is required")
        raise SystemExit(1)

    source = ImmichSource(CONFIG)
    pool = FramePool(source, CONFIG)
    pool.ensure()
    app._pool = pool

    log.info("Serving on %s:%d", CONFIG.host, CONFIG.port)
    app.run(host=CONFIG.host, port=CONFIG.port, threaded=True)


if __name__ == "__main__":
    main()
