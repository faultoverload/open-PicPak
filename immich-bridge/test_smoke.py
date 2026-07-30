"""
Smoke test for immich-bridge.

Verifies the public contract:
    - GET /frame.bin returns exactly 30,000 bytes
    - content-type is application/octet-stream
    - content-length header matches body length
    - /health returns ok=True

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
    every BWRY palette entry and most of the Atkinson diffusion kernel."""
    img = Image.new("RGB", (640, 480))
    px = img.load()
    for y in range(480):
        for x in range(640):
            # Two-axis gradient with palette-anchored stripes.
            r = (x * 255) // 639
            g = (y * 255) // 479
            b = ((x + y) * 255) // (639 + 479)
            # Force all four palette colours to appear.
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


def test_pipeline_directly() -> bytes:
    """Encode the fixture through the same path the HTTP server uses."""
    with tempfile.TemporaryDirectory() as td:
        fixture = os.path.join(td, "fixture.jpg")
        make_fixture(fixture)
        frame = bridge.image_from_path(fixture)
    assert len(frame) == FRAME_BYTES, len(frame)
    # First byte must be a packed 2-bpp value, i.e. only the bottom 8 bits
    # matter — but pack output is in [0, 3] per pixel so the high two bits
    # of each nibble must not exceed 0b11. We just sanity-check the value
    # is a real byte (trivially true) and that not every byte is identical
    # (which would indicate the pipeline produced a flat field).
    distinct = len({frame[i] for i in range(0, FRAME_BYTES, 73)})
    assert distinct > 4, f"pipeline produced suspiciously flat output ({distinct} distinct bytes)"
    return frame


def test_http_endpoints() -> None:
    """Spin up the Flask test client and verify the live contract."""
    with tempfile.TemporaryDirectory() as td:
        fixture = os.path.join(td, "fixture.jpg")
        make_fixture(fixture)
        os.environ["BRIDGE_FIXTURE_PATH"] = fixture
        # Force the app module to pick up the new env var (Config is read
        # at import time).
        bridge.CONFIG = bridge.Config()
        bridge.app.config["TESTING"] = True

        client = bridge.app.test_client()

        r = client.get("/frame.bin")
        assert r.status_code == 200, r.status_code
        assert r.headers["Content-Type"] == "application/octet-stream"
        assert r.headers["Content-Length"] == str(FRAME_BYTES)
        body = r.get_data()
        assert len(body) == FRAME_BYTES, len(body)

        r2 = client.get("/frame.bin")
        assert len(r2.get_data()) == FRAME_BYTES

        h = client.get("/health")
        assert h.status_code == 200
        payload = h.get_json()
        assert payload["ok"] is True
        assert payload["fixture"] is True


if __name__ == "__main__":
    fb = test_pipeline_directly()
    print(f"pipeline OK: {len(fb)} bytes")
    test_http_endpoints()
    print("http OK: /frame.bin returns 30000 bytes, /health returns ok=True")