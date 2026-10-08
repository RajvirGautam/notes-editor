#!/usr/bin/env python3
"""Batch-straighten + tone-normalize already-rendered notebook JPGs.

For each page:
  1. Tone: per-channel white-point normalization (uniform brightness + neutral
     tone across pages) + gentle contrast + unsharp (crisper handwriting).
  2. Deskew: notes-editor projection-profile method levels the TEXT lines
     (imaging.detect_skew / deskew_image), rotating with expand=True and a WHITE
     fill so nothing is ever cropped.
  3. Uniform frame: paste the deskewed page, unscaled, centered onto a fixed
     white canvas -> every output is identical size, handwriting stays the same
     scale, and exposed edges become clean white margin. No content is cropped.

Usage:
  python straighten_batch.py SRC_DIR OUT_DIR
SRC_DIR must contain page-1.jpg .. page-N.jpg
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import imaging as I   # notes-editor engine (same folder)

TARGET = (247, 245, 240)     # paper white-point per channel
WHITE_PCT = 96.0
MAX_GAIN = 2.3
MAX_ANGLE = 4.0              # clamp; real skews here are < 3 deg


def tone(im: Image.Image) -> Image.Image:
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    for c in range(3):
        hi = max(np.percentile(a[:, :, c], WHITE_PCT), 1.0)
        a[:, :, c] *= min(TARGET[c] / hi, MAX_GAIN)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(1.08)
    return im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=2))


def process(im: Image.Image) -> Image.Image:
    enh = tone(im)
    angle = I.detect_skew(enh)
    angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))
    # Rotate WITHIN the page frame (expand=False) so the page fills corner-to-
    # corner -> no thick white border. The only white is the small rotation
    # sliver at each corner, which lands on the blank page margin, not writing.
    if abs(angle) < 0.01:
        return enh, angle
    dz = enh.rotate(angle, resample=Image.BICUBIC, expand=False,
                    fillcolor=TARGET)
    return dz, angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "page-*.jpg")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    for f in files:
        im = Image.open(f)
        out, ang = process(im)
        dst = os.path.join(a.out, os.path.basename(f))
        out.save(dst, format="JPEG", quality=92)
        print(f"{os.path.basename(f):12} deskew={ang:+.2f}  -> {out.size}")


if __name__ == "__main__":
    main()
