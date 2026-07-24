"""
Core image processing for the notes deskewer.

Everything here is pure Pillow + numpy (no OpenCV) so it installs cleanly.

Pipeline per page:
  1. Render the PDF page to a raster (PyMuPDF).
  2. Detect the skew angle via the projection-profile method.
  3. Rotate (deskew) the page.
  4. Auto-detect the content bounding box (to suggest margin crops).
  5. On export, apply the chosen angle + crop at high DPI.
"""

from __future__ import annotations

import io
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_page(doc: fitz.Document, page_index: int, dpi: int) -> Image.Image:
    """Render a single PDF page to a white-background RGB PIL image."""
    page = doc[page_index]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    # alpha=False -> flatten onto white, which is what we want for scans.
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


# ---------------------------------------------------------------------------
# Skew detection (projection-profile method)
# ---------------------------------------------------------------------------

def _otsu_threshold(arr: np.ndarray) -> float:
    """Classic Otsu threshold on a 0-255 array."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 127.0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            threshold = i
    return float(threshold)


def _to_ink_mask(gray: Image.Image) -> np.ndarray:
    """Return a float32 array where ink (dark) ~1.0 and paper (light) ~0.0."""
    arr = np.asarray(gray, dtype=np.float32)
    threshold = _otsu_threshold(arr)
    return (arr < threshold).astype(np.float32)


def _projection_score(mask: np.ndarray) -> float:
    """
    Score how well text lines are horizontally aligned.

    Sum ink per row -> when lines are level, the profile has sharp peaks and
    deep valleys, so the squared row-to-row differences are large.
    """
    row_sums = mask.sum(axis=1)
    diffs = np.diff(row_sums)
    return float(np.sum(diffs * diffs))


def detect_skew(img: Image.Image,
                max_angle: float = 10.0,
                coarse_step: float = 1.0,
                fine_step: float = 0.1) -> float:
    """
    Estimate the correction angle (degrees) that levels the page.

    Positive result means the export step should rotate the page counter-
    clockwise by that amount (PIL's positive rotation direction).
    """
    # Downscale for speed; detection doesn't need full resolution.
    gray = img.convert("L")
    max_dim = 1000
    scale = min(1.0, max_dim / max(gray.size))
    if scale < 1.0:
        gray = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )

    def score_at(angle: float) -> float:
        rotated = gray.rotate(angle, resample=Image.BILINEAR,
                              expand=False, fillcolor=255)
        return _projection_score(_to_ink_mask(rotated))

    # Coarse pass.
    coarse_angles = np.arange(-max_angle, max_angle + coarse_step, coarse_step)
    coarse = [(a, score_at(a)) for a in coarse_angles]
    best_angle = max(coarse, key=lambda t: t[1])[0]

    # Fine pass around the coarse winner.
    lo, hi = best_angle - coarse_step, best_angle + coarse_step
    fine_angles = np.arange(lo, hi + fine_step, fine_step)
    fine = [(a, score_at(a)) for a in fine_angles]
    best_angle = max(fine, key=lambda t: t[1])[0]

    return round(float(best_angle), 2)


# ---------------------------------------------------------------------------
# Deskew + content detection + crop
# ---------------------------------------------------------------------------

def deskew_image(img: Image.Image, angle: float) -> Image.Image:
    """Rotate the page to correct skew, filling exposed corners with white."""
    if abs(angle) < 0.01:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255))


def content_bbox_fractions(img: Image.Image, pad_frac: float = 0.01) -> dict:
    """
    Find the handwriting bounding box on an (already deskewed) page and return
    crop insets as fractions of width/height: {left, top, right, bottom}.

    `right`/`bottom` are insets from those edges (so 0 means no crop).
    A small padding is added so strokes aren't clipped.
    """
    gray = img.convert("L")
    max_dim = 1200
    scale = min(1.0, max_dim / max(gray.size))
    small = gray
    if scale < 1.0:
        small = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )
    mask = _to_ink_mask(small)

    h, w = mask.shape
    col_ink = mask.sum(axis=0)
    row_ink = mask.sum(axis=1)

    # Ignore near-empty lines to shrug off speckle/scan noise at the edges.
    col_thr = max(1.0, col_ink.max() * 0.02)
    row_thr = max(1.0, row_ink.max() * 0.02)
    cols = np.where(col_ink > col_thr)[0]
    rows = np.where(row_ink > row_thr)[0]

    if len(cols) == 0 or len(rows) == 0:
        return {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    left_px, right_px = cols[0], cols[-1]
    top_px, bottom_px = rows[0], rows[-1]

    pad_x = pad_frac * w
    pad_y = pad_frac * h
    left = max(0.0, (left_px - pad_x) / w)
    top = max(0.0, (top_px - pad_y) / h)
    right = max(0.0, (w - 1 - right_px - pad_x) / w)
    bottom = max(0.0, (h - 1 - bottom_px - pad_y) / h)

    return {
        "left": round(float(left), 4),
        "top": round(float(top), 4),
        "right": round(float(right), 4),
        "bottom": round(float(bottom), 4),
    }


def apply_crop(img: Image.Image, crop: dict) -> Image.Image:
    """Crop by fractional insets {left, top, right, bottom}."""
    w, h = img.size
    left = int(round(crop.get("left", 0.0) * w))
    top = int(round(crop.get("top", 0.0) * h))
    right = w - int(round(crop.get("right", 0.0) * w))
    bottom = h - int(round(crop.get("bottom", 0.0) * h))
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return img.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Scan / colour look
# ---------------------------------------------------------------------------

DEFAULT_COLOR = {
    "mode": "none",       # none | color | grayscale | bw
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "whiten": 0.0,        # 0..1 — pushes the paper background toward pure white
}

# One-click "make it look scanned" preset (keeps pen colour, whitens paper).
AUTO_SCAN = {
    "mode": "none",
    "brightness": 1.02,
    "contrast": 1.12,
    "saturation": 1.55,
    "whiten": 0.8,
}


def _is_identity_color(color: dict) -> bool:
    return (color.get("mode", "none") == "none"
            and abs(color.get("brightness", 1.0) - 1.0) < 1e-3
            and abs(color.get("contrast", 1.0) - 1.0) < 1e-3
            and abs(color.get("saturation", 1.0) - 1.0) < 1e-3
            and color.get("whiten", 0.0) <= 1e-3)


def apply_scan(img: Image.Image, color: dict | None) -> Image.Image:
    """
    Apply a 'scanned notes' look: whiten the paper, boost contrast, keep or
    drop colour. All controls are optional and default to no-ops.
    """
    if not color or _is_identity_color(color):
        return img

    out = img.convert("RGB")
    mode = color.get("mode", "none")
    b = float(color.get("brightness", 1.0))
    c = float(color.get("contrast", 1.0))
    sat = float(color.get("saturation", 1.0))
    whiten = float(color.get("whiten", 0.0))

    if abs(b - 1.0) > 1e-3:
        out = ImageEnhance.Brightness(out).enhance(b)

    if whiten > 1e-3:
        # Levels stretch: lift the paper to pure white AND pull the ink toward
        # black at the same time. Unlike a plain brightness multiply this keeps
        # (actually deepens) the ink instead of washing it out, and the wider
        # channel spread makes coloured pens more vivid — the clean-scan look.
        s = min(max(whiten, 0.0), 1.0)
        black_point = s * 42.0            # 0   -> 42
        white_point = 255.0 - s * 92.0    # 255 -> 163
        span = max(white_point - black_point, 1.0)
        arr = np.asarray(out, dtype=np.float32)
        arr = (arr - black_point) * (255.0 / span)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if abs(c - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(c)
    if abs(sat - 1.0) > 1e-3:
        out = ImageEnhance.Color(out).enhance(sat)

    if mode == "grayscale":
        out = out.convert("L").convert("RGB")
    elif mode == "bw":
        gray = np.asarray(out.convert("L"), dtype=np.float32)
        thr = _otsu_threshold(gray)
        bw = np.where(gray >= thr, 255, 0).astype(np.uint8)
        out = Image.fromarray(bw).convert("RGB")

    return out


# ---------------------------------------------------------------------------
# Palette learning from a sample page, and colour transfer to other pages
# ---------------------------------------------------------------------------

def _rgb_to_hsv_np(rgb01: np.ndarray):
    """Vectorised RGB->HSV. rgb01 is (...,3) in 0..1. Returns h,s,v each (...,)."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mx = rgb01.max(-1)
    mn = rgb01.min(-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    m = diff > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & (mx == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / diff[rm]) % 6
    h[gm] = ((b[gm] - r[gm]) / diff[gm]) + 2
    h[bm] = ((r[bm] - g[bm]) / diff[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def _lum(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _ink_alpha(s: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """
    Per-pixel ink coverage (0 paper .. 1 solid stroke), robust to uneven
    lighting: each pixel is compared to its *local* paper brightness (a heavy
    blur), so shadows and gradients don't get mistaken for ink. Colourful
    pixels (highlighters) also count regardless of brightness.
    """
    h, w = lum.shape
    radius = max(18, int(max(h, w) / 12))
    lum_img = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8))
    local_bg = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(radius)),
                          dtype=np.float32)
    local_bg = np.maximum(local_bg, 1.0)
    # how much darker than the local paper (as a ratio, so gradients cancel);
    # the 0.10 floor keeps scan noise on the paper from registering as ink.
    darkness = np.clip((1.0 - lum / local_bg - 0.10) / 0.30, 0.0, 1.0)
    colour = np.clip((s - 0.20) / 0.45, 0.0, 1.0)
    alpha = np.clip(np.maximum(darkness, colour * 0.9), 0.0, 1.0)
    return alpha * alpha * (3 - 2 * alpha)   # smoothstep: crush faint, keep solid


def extract_palette(img: Image.Image) -> dict:
    """
    Learn a colour palette from a sample page: the paper (background) colour
    plus the distinct pen/highlight colours.

    Pens are extracted from the ink pixels only, and each pen's colour is taken
    from that hue's *most saturated* pixels — so thin strokes that quantisation
    would otherwise blend with the paper come back as their true vivid colour.
    """
    small = img.convert("RGB")
    small.thumbnail((520, 520))
    rgb = np.asarray(small, dtype=np.float32)
    rgb01 = rgb / 255.0
    h, s, v = _rgb_to_hsv_np(rgb01)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # --- background (paper): median of the bright, low-chroma pixels ----------
    paper = (lum > np.percentile(lum, 60)) & (s < 0.18)
    if paper.sum() > 50:
        bg = np.median(rgb[paper], axis=0)
    else:
        bg = np.array([np.percentile(rgb[..., c], 90) for c in range(3)])
    bg_lum = float(_lum(bg))

    # --- ink mask (illumination-robust) ---------------------------------------
    alpha = _ink_alpha(s, lum)
    ink = alpha > 0.28

    hf, sf, vf = h[ink], s[ink], v[ink]
    rf = rgb[ink]
    inks = []

    # --- coloured pens: find dominant hues, vivid representative each ----------
    colourful = sf > 0.22
    if colourful.sum() > 30:
        hc, sc, rc = hf[colourful], sf[colourful], rf[colourful]
        nb = 36
        bins = np.minimum((hc * nb).astype(int), nb - 1)
        weight = np.bincount(bins, weights=sc, minlength=nb)
        # circular smoothing
        sm = weight.copy()
        for _ in range(2):
            sm = (np.roll(sm, 1) + 2 * sm + np.roll(sm, -1)) / 4.0
        peak = sm.max()
        for b in range(nb):
            if sm[b] < peak * 0.16:
                continue
            if sm[b] < sm[(b - 1) % nb] or sm[b] < sm[(b + 1) % nb]:
                continue  # keep local maxima only
            # pixels within +/- 1 bin of this hue peak
            near = (np.abs(bins - b) <= 1) | (np.abs(bins - b) >= nb - 1)
            if near.sum() < 12:
                continue
            sub_r, sub_s = rc[near], sc[near]
            # vivid representative = mean of the top-saturation third
            cut = np.percentile(sub_s, 66)
            vivid = sub_r[sub_s >= cut]
            rep = vivid.mean(axis=0) if len(vivid) else sub_r.mean(axis=0)
            if not any(np.abs(rep - e).sum() < 40 for e in inks):
                inks.append(rep)

    # --- neutral / black pen (only when there's a genuinely dark one) ---------
    neutral = (sf < 0.22) & (vf < (bg_lum / 255.0) * 0.75)
    if neutral.sum() > 20:
        nrep = rf[neutral].mean(axis=0)
        if _lum(nrep) < 110:
            inks.insert(0, nrep)

    if not inks:
        inks = [np.array([30, 30, 40], dtype=np.float32)]

    return {"background": [int(round(x)) for x in bg],
            "inks": [[int(round(v)) for v in c] for c in inks[:6]]}


def transfer_palette(img: Image.Image, palette: dict) -> Image.Image:
    """
    Repaint a page into a learned palette: clean the paper to the sample's
    background colour and recolour ink strokes to the nearest sample pen colour,
    keeping each stroke's coverage (anti-aliasing) intact.
    """
    if not palette or not palette.get("inks"):
        return img
    bg = np.array(palette["background"], dtype=np.float32)
    inks = np.array(palette["inks"], dtype=np.float32)      # K x 3

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    rgb01 = rgb / 255.0
    h, s, _ = _rgb_to_hsv_np(rgb01)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Ink coverage per pixel, judged against the local paper brightness so
    # shadows/gradients don't get painted as ink.
    alpha = _ink_alpha(s, lum)

    # Match each ink pixel to a sample pen: by hue when colourful, else (only if
    # the sample actually has a dark/neutral pen) by that neutral pen.
    ink_h, _, _ = _rgb_to_hsv_np((inks / 255.0)[None, :, :])
    ink_h = ink_h.reshape(-1)
    ink_lum = 0.299 * inks[:, 0] + 0.587 * inks[:, 1] + 0.114 * inks[:, 2]
    neutral_idx = int(np.argmin(ink_lum))

    hue_d = np.abs(h[..., None] - ink_h[None, None, :])
    hue_d = np.minimum(hue_d, 1.0 - hue_d)
    idx = np.argmin(hue_d, axis=-1)
    if ink_lum[neutral_idx] < 110:
        idx = np.where(s < 0.18, neutral_idx, idx)
    chosen = inks[idx]                                       # H x W x 3

    a = alpha[..., None]
    out = bg[None, None, :] * (1.0 - a) + chosen * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def finish_color(img: Image.Image, color: dict | None,
                 palette: dict | None = None) -> Image.Image:
    """Apply either palette transfer (mode 'sample') or the slider scan look."""
    if color and color.get("mode") == "sample" and palette:
        return transfer_palette(img, palette)
    return apply_scan(img, color)


# ---------------------------------------------------------------------------
# Watermark overlay
# ---------------------------------------------------------------------------

# A full-page RGBA overlay (the "Arivihan" branding), mostly transparent with a
# faint diagonal logo. It's composited on top of the finished page.
WATERMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "watermark.png")

# Native opacity is quite faint (~226 grey over white); a small boost lands it
# closer to the intended look. Callers pass a multiplier where 1.0 == native.
DEFAULT_WATERMARK_OPACITY = 1.5

_watermark_cache: dict = {}


def _load_watermark() -> Image.Image | None:
    """Load (and cache) the watermark PNG as RGBA, or None if it's missing."""
    if "img" not in _watermark_cache:
        try:
            _watermark_cache["img"] = Image.open(WATERMARK_PATH).convert("RGBA")
        except (FileNotFoundError, OSError):
            _watermark_cache["img"] = None
    return _watermark_cache["img"]


def apply_watermark(img: Image.Image, opacity: float = DEFAULT_WATERMARK_OPACITY
                    ) -> Image.Image:
    """
    Composite the branding watermark over a finished page.

    The overlay is scaled to *cover* the page (preserving its aspect so the logo
    isn't distorted) and centre-cropped, then alpha-composited. `opacity` scales
    the overlay's alpha (1.0 == the PNG's native, faint opacity).
    """
    if opacity <= 0:
        return img
    wm = _load_watermark()
    if wm is None:
        return img

    W, H = img.size
    ww, wh = wm.size
    scale = max(W / ww, H / wh)
    nw, nh = max(1, round(ww * scale)), max(1, round(wh * scale))
    layer = wm.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - W) // 2, (nh - H) // 2
    layer = layer.crop((left, top, left + W, top + H))

    if abs(opacity - 1.0) > 1e-3:
        arr = np.asarray(layer, dtype=np.float32)
        arr[..., 3] = np.clip(arr[..., 3] * opacity, 0.0, 255.0)
        layer = Image.fromarray(arr.astype(np.uint8), "RGBA")

    base = img.convert("RGBA")
    return Image.alpha_composite(base, layer).convert("RGB")


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_page(img: Image.Image, angle: float, crop: dict,
                 color: dict | None = None,
                 palette: dict | None = None) -> Image.Image:
    """Full transform for one page: deskew -> crop -> scan/colour/palette."""
    dcrop = apply_crop(deskew_image(img, angle), crop)
    return finish_color(dcrop, color, palette)
