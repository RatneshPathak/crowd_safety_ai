"""Generate a synthetic clip for smoke-testing the pipeline.

The clip is moving dots on a textured background whose brightness and speed
ramp up part-way through. It is NOT crowd footage; it only exercises the code.
Usage: python tests/make_test_video.py [out.mp4] [seconds]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make(out="data/videos/synthetic_test.mp4", seconds=8, fps=25, size=(640, 360)):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    rng = np.random.default_rng(7)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 90
    pos = rng.uniform([0, 0], [w, h], (n, 2))
    vel = rng.normal(0, 1.0, (n, 2))
    total = int(seconds * fps)
    for i in range(total):
        phase = i / total
        base = 50 + int(90 * max(0.0, phase - 0.4) / 0.6)       # brightens late
        speed = 1.0 + 7.0 * max(0.0, phase - 0.55) / 0.45        # speeds up late
        frame = np.full((h, w, 3), base, np.uint8)
        noise = rng.integers(0, 25, (h, w, 1), dtype=np.uint8)
        frame = cv2.add(frame, np.repeat(noise, 3, axis=2))
        pos += vel * speed
        pos[:, 0] %= w
        pos[:, 1] %= h
        for x, y in pos:
            cv2.circle(frame, (int(x), int(y)), 9, (230, 230, 240), -1)
        vw.write(frame)
    vw.release()
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/videos/synthetic_test.mp4"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(make(out, secs))
