"""Benchmark OCR latency and accuracy for the launch/gameplay pipeline.

Run from the pc-gaming-agent/ directory:

    python -m tools.ocr.ocr_benchmark [path\\to\\screenshot.png]

If no screenshot is given, the newest PNG under screenshots/ is used, or a
synthetic UI frame is generated so the script is always runnable. It reports, for
the current (optimised) engine config:

    * full-frame  read_text          latency + detected texts
    * ROI-crop    read_text_roi      latency + detected texts  (should match texts)
    * cache-hit   read_text again    latency (should be ~0 ms)
    * precomputed _match_in_frame    latency (should be ~0 ms)

Compare the ROI/cache/precomputed numbers against the full-frame baseline to
confirm the ~35-45s scans are eliminated on repeat/second-pass access while the
detected texts (accuracy) are unchanged.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _newest_screenshot() -> Path | None:
    root = Path(__file__).resolve().parent.parent.parent / "screenshots"
    if not root.exists():
        return None
    pngs = sorted(root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pngs[0] if pngs else None


def _load(path: str | None) -> np.ndarray:
    p = Path(path) if path else _newest_screenshot()
    if p is not None and p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            print(f"Loaded screenshot: {p}")
            return img
    print("No screenshot available - using a synthetic UI frame.")
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    cv2.putText(frame, "Minecraft for Windows", (380, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 2.6, (15, 15, 15), 4)
    cv2.putText(frame, "Play", (900, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, (15, 15, 15), 6)
    return frame


def _time(fn, *args, **kw) -> tuple[float, list]:
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return (time.perf_counter() - t0) * 1000.0, out


def _texts(results) -> list[str]:
    return [r.text for r in results] if results else []


def main() -> None:
    frame = _load(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    print("=" * 66)

    from tools.desktop.screen_finder import ScreenFinder
    from tools.ocr.paddle_engine import PaddleOCREngine

    eng = PaddleOCREngine(cache_ttl_seconds=30.0)
    eng.warmup()   # exclude one-time model load from the measured numbers

    ms_full, full = _time(eng.read_text, frame, use_cache=False)
    print(f"[full-frame ] {ms_full:8.0f} ms  texts={_texts(full)[:8]}")

    ms_roi, roi = _time(eng.read_text_roi, frame, use_cache=False)
    print(f"[roi-crop   ] {ms_roi:8.0f} ms  texts={_texts(roi)[:8]}")

    eng.read_text(frame, use_cache=True)             # warm cache
    ms_hit, _ = _time(eng.read_text, frame, use_cache=True)
    print(f"[cache-hit  ] {ms_hit:8.1f} ms  (repeat scan of unchanged frame)")

    sf = ScreenFinder(ocr=eng)
    ms_pre, _ = _time(sf._match_in_frame, frame, "play", 0.3, (), roi)  # noqa: SLF001
    print(f"[precomputed] {ms_pre:8.1f} ms  (_match_in_frame reusing OCR results)")

    print("=" * 66)
    if ms_full > 0:
        print(f"ROI speedup vs full : {ms_full / max(ms_roi, 1e-6):5.2f}x")
        print(f"Cache-hit speedup   : {ms_full / max(ms_hit, 1e-6):8.0f}x")
        print(f"Precomputed speedup : {ms_full / max(ms_pre, 1e-6):8.0f}x")
    same = set(_texts(full)) == set(_texts(roi))
    print(f"Accuracy preserved (full texts == roi texts): {same}")


if __name__ == "__main__":
    main()
