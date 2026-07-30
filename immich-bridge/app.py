"""
Immich Bridge — Phase 2

Pulls photos from Immich and serves them as 30,000-byte BWRY framebuffers
compatible with the open-PicPak firmware. Single-file Flask app.

Pipeline: resize → perceptual (Oklab) or app (BT.601) dither → vertical flip
→ 2 bpp MSB-first packing (4 px/byte).

Endpoints:
    GET /frame.bin   -> application/octet-stream, exactly 30,000 bytes
    GET /frame.png   -> PNG preview of the last-served frame (for debugging)
    GET /health      -> {"ok": true, "pool_size": int, "served": int, "db": ...}
    GET /info        -> {"last_id": "...", "pool_size": int, "dither_mode": "..."}
    GET /next        -> force-advance to the next frame (returns new metadata)
    GET /pool        -> list of {id, filename, date, album} in current pool

The PicPak firmware wakes, GETs /frame.bin over WiFi, displays the buffer,
and goes back to sleep. One frame per wake.
"""
from __future__ import annotations

import io
import logging
import os
import random
import threading
import time
from collections import OrderedDict, deque
from datetime import date, datetime
from typing import Deque, Optional

import psycopg2
import psycopg2.extras
from flask import Flask, Response, jsonify
from PIL import Image

# Register HEIC/HEIF support if available (common for iPhone photos in Immich)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAVE_HEIF = True
except ImportError:
    HAVE_HEIF = False

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

# Floyd-Steinberg kernel for perceptual dithering (serpentine scan).
# (dx, dy, weight) — weight is relative to divisor=16.
FLOYD_STEINBERG = ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1))


# ---------------------------------------------------------------------------
# Oklab perceptual colour space — used by "perceptual" dither mode
# ---------------------------------------------------------------------------

def _srgb_to_linear(value: float) -> float:
    value /= 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _linear_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Linear sRGB → Oklab (perceptually uniform)."""
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
    """Convert the BWRY palette into Oklab coordinates."""
    return [
        _linear_to_oklab(*[_srgb_to_linear(c) for c in color])
        for color in PALETTE
    ]


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
        self.upload_location: str = os.environ.get(
            "UPLOAD_LOCATION", "/usr/src/app/upload"
        )

        # Pool rotation
        self.pool_size: int = int(os.environ.get("IMMICH_POOL_SIZE", "100"))
        self.refresh_seconds: float = float(
            os.environ.get("IMMICH_POOL_REFRESH_SECONDS", "0")
        )

        # Optional filters
        self.album_id: Optional[str] = os.environ.get("IMMICH_ALBUM_ID") or None
        self.person_id: Optional[str] = os.environ.get("IMMICH_PERSON_ID") or None

        # Dithering mode: "perceptual" (Oklab Floyd-Steinberg) or "app" (BT.601 Atkinson)
        self.dither_mode: str = os.environ.get(
            "DITHER_MODE", "perceptual"
        ).lower()
        if self.dither_mode not in {"perceptual", "app"}:
            raise ValueError("DITHER_MODE must be 'perceptual' or 'app'")

        # LRU framebuffer cache size (avoids re-encoding on repeat serves)
        self.cache_size: int = max(
            1, int(os.environ.get("CACHE_SIZE", "20"))
        )

        # Server
        self.host: str = os.environ.get("BRIDGE_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("BRIDGE_PORT", "8090"))

        # Test/dev: skip the DB entirely and use a fixture file
        self.fixture_path: Optional[str] = os.environ.get("BRIDGE_FIXTURE_PATH")


CONFIG = Config()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("immich-bridge")
log.info("immich-bridge starting (dither=%s, pool=%d)", CONFIG.dither_mode, CONFIG.pool_size)


# ---------------------------------------------------------------------------
# BWRY pipeline — dual dithering mode
# ---------------------------------------------------------------------------

def to_bwry_frame(img: Image.Image, mode: str | None = None) -> bytes:
    """Convert a PIL image into the 30,000-byte BWRY framebuffer.

    mode="app": BT.601 + unclamped Atkinson (original Phase 1 pipeline)
    mode="perceptual": sRGB→Oklab + serpentine Floyd-Steinberg with chroma
                       weighting (1.35× on a/b channels)

    Steps:
        1. centre-crop to 4:3
        2. resize to 400x300
        3. dither to 4-colour palette
        4. vertical flip
        5. pack 2 bpp, 4 px/byte, MSB-first
    """
    if mode is None:
        mode = CONFIG.dither_mode

    # 1. centre-crop to 4:3
    src_w, src_h = img.size
    target_ratio = WIDTH / HEIGHT  # 4/3
    if src_w / src_h > target_ratio:
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(round(src_w / target_ratio))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    # 2. resize
    img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)

    # 3. dither
    if mode == "app":
        code = _dither_atkinson(img)
    else:
        code = _dither_perceptual(img)

    # 4+5. vertical flip + pack 2 bpp MSB-first
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
    """BT.601 luma-weighted quantisation to 4-colour palette with unclamped
    Atkinson error diffusion.  Original Phase 1 pipeline."""
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
    return code


def _dither_perceptual(img: Image.Image) -> list[int]:
    """Oklab-space quantisation with serpentine Floyd-Steinberg error diffusion.

    Uses 1.35× chroma weighting on the a/b channels to prioritise luminance
    accuracy — human vision is far more sensitive to luminance errors than
    chroma errors."""
    n = WIDTH * HEIGHT
    palette_ok = _palette_oklab()

    # Convert all pixels to Oklab
    L = [0.0] * n
    A = [0.0] * n
    B = [0.0] * n
    px = img.load()
    for i in range(n):
        r, g, b = px[i % WIDTH, i // WIDTH]
        L[i], A[i], B[i] = _linear_to_oklab(
            _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
        )

    code = [0] * n
    for y in range(HEIGHT):
        # Serpentine scan: even rows left→right, odd rows right→left
        reverse = bool(y & 1)
        xs = range(WIDTH - 1, -1, -1) if reverse else range(WIDTH)
        for x in xs:
            i = y * WIDTH + x
            lab = (L[i], A[i], B[i])

            # Find closest palette entry in Oklab space
            best = 0
            best_dist = float("inf")
            for k in range(4):
                pl, pa, pb = palette_ok[k]
                dL = lab[0] - pl
                dA = lab[1] - pa
                dB = lab[2] - pb
                # 1.35× chroma weighting: penalise chroma errors harder
                dist = dL * dL + 1.35 * dA * dA + 1.35 * dB * dB
                if dist < best_dist:
                    best_dist = dist
                    best = k

            code[i] = best

            # Distribute error to neighbours (Floyd-Steinberg kernel, divisor=16)
            pl, pa, pb = palette_ok[best]
            eL = (lab[0] - pl) / 16.0
            eA = (lab[1] - pa) / 16.0
            eB = (lab[2] - pb) / 16.0

            for dx, dy, weight in FLOYD_STEINBERG:
                # In serpentine scan, flip horizontal offsets on reverse rows
                actual_dx = -dx if reverse else dx
                nx = x + actual_dx
                ny = y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    j = ny * WIDTH + nx
                    L[j] += eL * weight
                    A[j] += eA * weight
                    B[j] += eB * weight

    return code


def image_from_path(path: str, mode: str | None = None) -> bytes:
    """Load an image from disk and produce a frame."""
    with Image.open(path) as img:
        return to_bwry_frame(img, mode)


# ---------------------------------------------------------------------------
# Immich integration
# ---------------------------------------------------------------------------

# Query with optional album and person filters.
# Uses parameterised NULL placeholders for optional filters — when album_id
# or person_id is None, the condition evaluates to TRUE and returns all matches.
IMMICH_QUERY = """
SELECT
    a."id", a."originalFileName", a."originalPath",
    a."fileCreatedAt" AS "date",
    al."albumName" AS "album"
FROM asset a
LEFT JOIN album_asset aa ON aa."assetId" = a."id"
LEFT JOIN album al ON al."id" = aa."albumId" AND al."deletedAt" IS NULL
LEFT JOIN asset_face af ON af."assetId" = a."id" AND af."deletedAt" IS NULL
LEFT JOIN person p ON p."id" = af."personId"
WHERE a."visibility" = 'timeline'
  AND a."type" = 'IMAGE'
  AND a."deletedAt" IS NULL
  AND (%s::uuid IS NULL OR al."id" = %s::uuid)
  AND (%s::uuid IS NULL OR p."id" = %s::uuid)
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

    def db_health(self) -> dict:
        """Check DB connectivity and return status info."""
        result = {
            "host": self.config.db_host,
            "port": self.config.db_port,
            "dbname": self.config.db_name,
            "connected": False,
            "error": None,
        }
        try:
            self.ensure()
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            result["connected"] = True
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def fetch(self, n: int) -> list[dict]:
        last_err: Optional[Exception] = None
        album = self.config.album_id
        person = self.config.person_id
        for attempt in range(2):
            try:
                self.ensure()
                with self._conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cur:
                    cur.execute(IMMICH_QUERY, (album, album, person, person, n))
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
    """Thread-safe rotating deque of pre-encoded frames with LRU cache.

    Pop-and-serve semantics:
        - on startup: query N random photos, encode them all, shuffle, push
        - each GET: pop next, serve it
        - on exhaustion: re-query and re-shuffle, skipping the last-served
          photo id so we never show the same frame twice in a row

    LRU framebuffer cache: encoded frames are cached in an OrderedDict
    keyed by asset_id so repeat serves (e.g. from /next force-advance
    returning to a previously-seen photo) don't re-encode.
    """

    def __init__(self, source: ImmichSource, config: Config):
        self.source = source
        self.config = config
        self._lock = threading.RLock()  # re-entrant: _encode may be called inside _refresh_locked
        self._frames: Deque[tuple[str, bytes]] = deque()  # (asset_id, frame)
        self._last_id: Optional[str] = None
        self._last_metadata: Optional[dict] = None  # metadata for /info endpoint
        self._current_metadata: Optional[dict] = None  # metadata for /info endpoint
        self._cache: OrderedDict[str, bytes] = OrderedDict()  # LRU framebuffer cache
        self._pool_meta: list[dict] = []  # metadata for all assets in current pool
        self.served = 0
        self.errors = 0
        self.last_refresh: Optional[float] = None

    def _encode(self, asset: dict) -> Optional[tuple[str, bytes]]:
        """Encode a single asset, using the LRU cache if available."""
        asset_id = asset["id"]

        # Check cache first
        with self._lock:
            if asset_id in self._cache:
                # Move to end (most-recently-used)
                self._cache.move_to_end(asset_id)
                return asset_id, self._cache[asset_id]

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
            log.warning(
                "Asset %s produced %d bytes, expected %d",
                asset_id, len(frame), FRAME_BYTES,
            )
            return None

        # Cache the encoded frame
        with self._lock:
            self._cache[asset_id] = frame
            if len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)  # evict LRU
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
                "id": row["id"],
                "filename": row["originalFileName"],
                "date": str(row["date"]) if row.get("date") else None,
                "album": row.get("album"),
            })
        random.shuffle(encoded)
        # Re-sync meta order with shuffled encoded order
        id_to_meta = {m["id"]: m for m in meta}
        meta_shuffled = [id_to_meta[aid] for aid, _ in encoded if aid in id_to_meta]
        self._frames.clear()
        self._pool_meta = []
        for i, (asset_id, frame) in enumerate(encoded[: self.config.pool_size]):
            self._frames.append((asset_id, frame))
            if i < len(meta_shuffled):
                self._pool_meta.append(meta_shuffled[i])
        self.last_refresh = time.time()
        log.info(
            "Pool refreshed: %d frames (queried %d, skipped last=%s)",
            len(self._frames), len(rows), self._last_id,
        )

    def ensure_filled(self) -> None:
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
            self._last_metadata = None  # stale
            self.served += 1
            return frame

    def force_next(self) -> Optional[dict]:
        """Force-advance to the next frame and return its metadata."""
        fb = self.next()
        if fb is None:
            return None
        return self.info()

    def info(self) -> dict:
        """Return metadata about the last-served frame."""
        with self._lock:
            return {
                "last_id": self._last_id,
                "pool_size": len(self._frames),
                "served": self.served,
                "errors": self.errors,
                "dither_mode": self.config.dither_mode,
                "album_id": self.config.album_id,
                "person_id": self.config.person_id,
            }

    def pool_list(self) -> list[dict]:
        """Return metadata for all assets currently in the pool."""
        with self._lock:
            return list(self._pool_meta)

    def stats(self) -> dict:
        with self._lock:
            return {
                "pool_size": len(self._frames),
                "served": self.served,
                "errors": self.errors,
                "last_id": self._last_id,
                "last_refresh": self.last_refresh,
                "cache_size": len(self._cache),
                "dither_mode": self.config.dither_mode,
            }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
source = ImmichSource(CONFIG)
pool = FramePool(source, CONFIG)

# Shared reference to the last-served frame bytes for the /frame.png endpoint.
_last_frame: Optional[bytes] = None


def _fixture_frame() -> Optional[bytes]:
    """Load a single fixture file and serve it forever. Used in tests."""
    path = CONFIG.fixture_path
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return image_from_path(path)


def _decode_frame_to_png(frame: bytes) -> bytes:
    """Decode a BWRY framebuffer back to a PNG for debugging/preview."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    px = img.load()
    unpacked = [0] * (WIDTH * HEIGHT)
    o = 0
    for y in range(HEIGHT - 1, -1, -1):
        row_base = y * WIDTH
        for x in range(0, WIDTH, 4):
            base = row_base + x
            b = frame[o]
            unpacked[base]     = (b >> 6) & 3
            unpacked[base + 1] = (b >> 4) & 3
            unpacked[base + 2] = (b >> 2) & 3
            unpacked[base + 3] = b & 3
            o += 1
    for i in range(WIDTH * HEIGHT):
        px[i % WIDTH, i // WIDTH] = PALETTE[unpacked[i]]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@app.get("/frame.bin")
def frame() -> Response:
    """Serve the next pre-encoded BWRY framebuffer (30,000 bytes)."""
    global _last_frame
    if CONFIG.fixture_path:
        fb = _fixture_frame()
        if fb is None:
            return Response(b"", status=503, mimetype="application/octet-stream")
    else:
        pool.ensure_filled()
        fb = pool.next()
        if fb is None:
            return Response(b"", status=503, mimetype="application/octet-stream")

    _last_frame = fb
    assert len(fb) == FRAME_BYTES, (
        f"frame is {len(fb)} bytes, expected {FRAME_BYTES}"
    )
    return Response(
        fb,
        status=200,
        mimetype="application/octet-stream",
        headers={
            "Content-Length": str(FRAME_BYTES),
            "Cache-Control": "no-store",
        },
    )


@app.get("/frame.png")
def frame_png() -> Response:
    """Return a PNG preview of the last-served frame (for debugging)."""
    if _last_frame is None:
        return Response(b"", status=404)
    png = _decode_frame_to_png(_last_frame)
    return Response(png, status=200, mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> Response:
    db_health = source.db_health() if not CONFIG.fixture_path else None
    return jsonify({
        "ok": True,
        "fixture": CONFIG.fixture_path is not None,
        "db": db_health,
        **pool.stats(),
    })


@app.get("/info")
def info() -> Response:
    """Metadata about the last-served frame and current pool state."""
    return jsonify(pool.info())


@app.get("/next")
def next_frame() -> Response:
    """Force-advance to the next frame, returning its metadata."""
    result = pool.force_next()
    if result is None:
        return jsonify({"error": "pool empty"}), 503
    return jsonify(result)


@app.get("/pool")
def pool_endpoint() -> Response:
    """List metadata for all assets in the current pool."""
    return jsonify(pool.pool_list())


def main() -> None:
    if CONFIG.fixture_path:
        log.info("Fixture mode: serving %s on every request", CONFIG.fixture_path)
    else:
        log.info(
            "Connecting to Immich at %s:%s/%s as %s (mode=%s)",
            CONFIG.db_host, CONFIG.db_port, CONFIG.db_name,
            CONFIG.db_user, CONFIG.dither_mode,
        )
        try:
            pool.ensure_filled()
        except Exception as exc:
            log.error("Initial pool fill failed: %s", exc)
    log.info("Listening on %s:%d", CONFIG.host, CONFIG.port)
    app.run(host=CONFIG.host, port=CONFIG.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
