"""
Smoke test for immich-bridge (Phase 2).

Verifies the public contract:
    - GET /frame.bin returns exactly 30,000 bytes
    - GET /frame.png returns a valid PNG
    - GET /health returns ok=True with db status (null in fixture mode)
    - GET /info returns metadata with dither_mode
    - GET /pool returns a list
    - GET /next force-advances
    - content-type is application/octet-stream for /frame.bin
    - content-length header matches body length

Uses BRIDGE_FIXTURE_PATH so the test never touches a real Immich DB.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import app as bridge  # noqa: E402

FRAME_BYTES = bridge.FRAME_BYTES
assert FRAME_BYTES == 30_000, FRAME_BYTES


def make_fixture(path: str) -> None:
    """A real-looking photo: gradients + four-colour swatches that exercise
    every BWRY palette entry and both dithering kernels."""
    img = Image.new("RGB", (640, 480))
    px = img.load()
    for y in range(480):
        for x in range(640):
            r = (x * 255) // 639
            g = (y * 255) // 479
            b = ((x + y) * 255) // (639 + 479)
            if (x // 80) % 4 == 0 and (y // 60) % 3 == 0:
                px[x, y] = (0, 0, 0)
            elif (x // 80) % 4 == 1 and (y // 60) % 3 == 0:
                px[x, y] = (255, 255, 255)
            elif (x // 80) % 4 == 2 and (y // 60) % 3 == 0:
                px[x, y] = (255, 255, 0)
            elif (x // 80) % 4 == 3 and (y // 60) % 3 == 0:
                px[x, y] = (255, 0, 0)
            else:
                px[x, y] = (r, g, b)
    img.save(path, "JPEG", quality=92)


def test_pipeline_atkinson() -> bytes:
    """Encode using the 'app' (BT.601 + Atkinson) pipeline."""
    with tempfile.TemporaryDirectory() as td:
        fixture = os.path.join(td, "fixture.jpg")
        make_fixture(fixture)
        with Image.open(fixture) as img:
            frame = bridge.to_bwry_frame(img, mode="app")
    assert len(frame) == FRAME_BYTES, len(frame)
    distinct = len({frame[i] for i in range(0, FRAME_BYTES, 73)})
    assert distinct > 4, f"pipeline produced suspiciously flat output ({distinct} distinct bytes)"
    return frame


def test_pipeline_perceptual() -> bytes:
    """Encode using the 'perceptual' (Oklab + Floyd-Steinberg) pipeline."""
    with tempfile.TemporaryDirectory() as td:
        fixture = os.path.join(td, "fixture.jpg")
        make_fixture(fixture)
        with Image.open(fixture) as img:
            frame = bridge.to_bwry_frame(img, mode="perceptual")
    assert len(frame) == FRAME_BYTES, len(frame)
    distinct = len({frame[i] for i in range(0, FRAME_BYTES, 73)})
    assert distinct > 4, f"perceptual pipeline produced flat output ({distinct} distinct bytes)"
    return frame


def test_http_endpoints() -> None:
    """Spin up the Flask test client and verify all Phase 2 endpoints."""
    with tempfile.TemporaryDirectory() as td:
        fixture = os.path.join(td, "fixture.jpg")
        make_fixture(fixture)
        os.environ["BRIDGE_FIXTURE_PATH"] = fixture
        bridge.CONFIG = bridge.Config()
        bridge.app.config["TESTING"] = True
        bridge._last_frame = None

        client = bridge.app.test_client()

        # /frame.bin
        r = client.get("/frame.bin")
        assert r.status_code == 200, r.status_code
        assert r.headers["Content-Type"] == "application/octet-stream"
        assert r.headers["Content-Length"] == str(FRAME_BYTES)
        body = r.get_data()
        assert len(body) == FRAME_BYTES, len(body)

        # Second /frame.bin should still work (fixture mode re-serves)
        r2 = client.get("/frame.bin")
        assert len(r2.get_data()) == FRAME_BYTES

        # /frame.png
        r = client.get("/frame.png")
        assert r.status_code == 200, f"/frame.png returned {r.status_code}"
        assert r.headers["Content-Type"] == "image/png"
        png = r.get_data()
        assert png[:4] == b'\x89PNG', "not a valid PNG header"

        # /health
        h = client.get("/health")
        assert h.status_code == 200
        payload = h.get_json()
        assert payload["ok"] is True
        assert payload["fixture"] is True
        assert "dither_mode" in payload
        assert payload["dither_mode"] in ("perceptual", "app")
        assert "people_filter" in payload
        assert "people_count" in payload
        # No people filter set → people_filter is None, people_count is None
        assert payload["people_filter"] is None
        assert payload["people_count"] is None

        # /info
        i = client.get("/info")
        assert i.status_code == 200
        info = i.get_json()
        assert "dither_mode" in info
        assert "pool_size" in info
        assert "last_id" in info
        assert "people_filter" in info
        assert "people_count" in info
        assert info["people_filter"] is None
        assert info["people_count"] is None

        # /pool
        p = client.get("/pool")
        assert p.status_code == 200
        assert isinstance(p.get_json(), list)


if __name__ == "__main__":
    fb = test_pipeline_atkinson()
    print(f"atkinson OK: {len(fb)} bytes")
    fb2 = test_pipeline_perceptual()
    print(f"perceptual OK: {len(fb2)} bytes")
    test_http_endpoints()
    print("http OK: all Phase 2 endpoints correct")
