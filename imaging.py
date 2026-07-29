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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
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
    return (arr <= threshold).astype(np.float32)


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


_QUARTER_TURNS = {90: Image.ROTATE_270, 180: Image.ROTATE_180,
                  270: Image.ROTATE_90}


def rotate_page(img: Image.Image, rot: float) -> Image.Image:
    """
    Coarse page rotation in degrees *clockwise* (the UI convention).

    Exact multiples of 90 use lossless transposes; anything else falls back
    to a resampled rotate with white corner fill (like deskew).
    """
    r = float(rot) % 360.0
    if r < 0.01 or r > 359.99:
        return img
    snapped = round(r)
    if abs(r - snapped) < 1e-6 and snapped in _QUARTER_TURNS:
        return img.transpose(_QUARTER_TURNS[snapped])
    return img.rotate(-r, resample=Image.BICUBIC, expand=True,
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

def apply_removals(img: Image.Image, removals: list[dict] | None) -> Image.Image:
    """
    Remove horizontal blank ranges (gaps) from an image.
    Each item in `removals` is a dict {"top": float, "bottom": float} (fraction of height 0..1).
    Content below each gap is shifted upward, and the space left at the bottom of the page
    is filled with the page's bottom blank paper portion (preserving page height).
    """
    if not removals:
        return img

    w, h = img.size
    if h <= 1:
        return img

    # Parse and clean intervals
    cleaned = []
    for r in removals:
        t = float(r.get("top", 0.0))
        b = float(r.get("bottom", 0.0))
        if t > b:
            t, b = b, t
        t = max(0.0, min(1.0, t))
        b = max(0.0, min(1.0, b))
        if b - t > 0.001:
            cleaned.append((t, b))

    if not cleaned:
        return img

    # Sort intervals by top
    cleaned.sort(key=lambda x: x[0])

    # Merge overlapping or adjacent intervals
    merged = []
    for t, b in cleaned:
        if not merged:
            merged.append((t, b))
        else:
            prev_t, prev_b = merged[-1]
            if t <= prev_b:
                merged[-1] = (prev_t, max(prev_b, b))
            else:
                merged.append((t, b))

    # Convert to pixel bounds
    intervals_px = [(int(round(t * h)), int(round(b * h))) for t, b in merged]

    # Find row segments to KEEP
    keep_segments = []
    curr = 0
    for y1, y2 in intervals_px:
        if y1 > curr:
            keep_segments.append((curr, y1))
        curr = max(curr, y2)
    if curr < h:
        keep_segments.append((curr, h))

    # Crop kept segments
    slices = [img.crop((0, y_start, w, y_end)) for y_start, y_end in keep_segments if y_end > y_start]
    kept_h = sum(s.height for s in slices)
    removed_h = h - kept_h

    if removed_h <= 0 or not slices:
        return img

    # Sample background color for default fill or tile from bottom portion
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    bright_pixels = arr[lum > np.percentile(lum, 70)]
    if len(bright_pixels) > 0:
        bg_rgb = tuple(int(round(x)) for x in np.median(bright_pixels, axis=0))
    else:
        bg_rgb = (255, 255, 255)

    out = Image.new(img.mode, (w, h), bg_rgb)

    # Fill freed space [kept_h:h] with paper background / bottom margin tile
    bottom_margin_h = min(30, h)
    bottom_slice = img.crop((0, h - bottom_margin_h, w, h))
    for y_pos in range(kept_h, h, bottom_margin_h):
        out.paste(bottom_slice, (0, y_pos))

    # Paste kept content slices sequentially from top downwards
    curr_y = 0
    for s in slices:
        out.paste(s, (0, curr_y))
        curr_y += s.height

    return out


def auto_detect_blank_gaps(img: Image.Image, min_gap_frac: float = 0.025) -> list[dict]:
    """
    Auto-detect horizontal blank bands (gaps) between handwritten/printed content blocks.
    Returns list of dicts: [{"top": float, "bottom": float}, ...]
    """
    gray = img.convert("L")
    max_dim = 1000
    scale = min(1.0, max_dim / max(gray.size))
    small = gray
    if scale < 1.0:
        small = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )
    mask = _to_ink_mask(small)
    sh, sw = mask.shape
    row_ink = mask.sum(axis=1)

    col_ink = mask.sum(axis=0)
    row_thr = max(1.0, row_ink.max() * 0.015)
    rows = np.where(row_ink > row_thr)[0]

    if len(rows) < 2:
        return []

    first_content_row, last_content_row = rows[0], rows[-1]

    is_blank = row_ink <= row_thr
    gaps = []
    in_gap = False
    gap_start = 0

    for r in range(first_content_row, last_content_row + 1):
        if is_blank[r]:
            if not in_gap:
                in_gap = True
                gap_start = r
        else:
            if in_gap:
                in_gap = False
                gap_end = r - 1
                gap_h = gap_end - gap_start + 1
                if gap_h / float(sh) >= min_gap_frac:
                    gaps.append({
                        "top": round(gap_start / float(sh), 4),
                        "bottom": round((gap_end + 1) / float(sh), 4),
                    })

    if in_gap:
        gap_end = last_content_row
        gap_h = gap_end - gap_start + 1
        if gap_h / float(sh) >= min_gap_frac:
            gaps.append({
                "top": round(gap_start / float(sh), 4),
                "bottom": round((gap_end + 1) / float(sh), 4),
            })

    return gaps


# ---------------------------------------------------------------------------
# Scan / colour look
# ---------------------------------------------------------------------------


DEFAULT_COLOR = {
    "mode": "none",       # none | color | grayscale | bw
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "whiten": 0.0,        # 0..1 — pushes the paper background toward pure white
    "deyellow": 0.0,      # 0..1 — neutralises yellow paper cast toward white
}

# One-click "make it look scanned" preset (keeps pen colour, whitens paper).
AUTO_SCAN = {
    "mode": "none",
    "brightness": 1.02,
    "contrast": 1.12,
    "saturation": 1.55,
    "whiten": 0.8,
    "deyellow": 0.85,
}


def _is_identity_color(color: dict) -> bool:
    return (color.get("mode", "none") == "none"
            and abs(color.get("brightness", 1.0) - 1.0) < 1e-3
            and abs(color.get("contrast", 1.0) - 1.0) < 1e-3
            and abs(color.get("saturation", 1.0) - 1.0) < 1e-3
            and color.get("whiten", 0.0) <= 1e-3
            and color.get("deyellow", 0.0) <= 1e-3)


def apply_deyellow(img: Image.Image, strength: float) -> Image.Image:
    """
    Anti-yellowing: turn yellow/aged/warm-lit paper into white paper.

    A brightness/levels stretch can't fix a colour cast — yellow paper stays
    yellow, just brighter. Instead this estimates the paper's colour *per
    channel and per region* (max-filter a small copy to erase ink strokes,
    then blur), and divides the page by that background. Whatever colour the
    paper is — yellow, warm-lit, unevenly shadowed — it maps to pure white,
    while ink keeps its contrast against the paper.

    Bright, strongly-saturated fills (highlighter strokes) are protected,
    otherwise the division would bleach them to white along with the paper.
    """
    s = min(max(float(strength), 0.0), 1.0)
    if s <= 1e-3:
        return img

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)

    # Per-channel paper background on a small copy: MaxFilter removes the
    # (darker) ink strokes so only paper survives, blur smooths the estimate.
    small = rgb.copy()
    small.thumbnail((360, 360), Image.BILINEAR)
    small = small.filter(ImageFilter.MaxFilter(7))
    small = small.filter(ImageFilter.GaussianBlur(10))
    bg = np.asarray(small.resize(rgb.size, Image.BILINEAR), dtype=np.float32)
    bg = np.maximum(bg, 64.0)

    flat = np.clip(arr * (255.0 / bg), 0.0, 255.0)

    # Highlighter guard: paper (even yellow paper) is only mildly saturated;
    # a highlighter fill is bright AND vivid. Fade the correction out there.
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    vivid = (np.clip((sat - 0.35) / 0.15, 0.0, 1.0)
             * np.clip((mx - 170.0) / 50.0, 0.0, 1.0))
    w = (s * (1.0 - vivid))[..., None]

    out = arr * (1.0 - w) + flat * w
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
    deyellow = float(color.get("deyellow", 0.0))

    # Anti-yellowing runs first so the whiten levels-stretch and contrast work
    # on already-neutral (white) paper instead of fighting the yellow cast.
    if deyellow > 1e-3:
        out = apply_deyellow(out, deyellow)

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

    # Drop highlighter-like colours from the *pen* set. A highlighter is a
    # bright, translucent fill sitting over white paper, so it reads as high
    # luminance; real pens (even vivid red/blue) are darker strokes. If we kept
    # highlighters here they'd be used to recolour strokes on every other page
    # — i.e. the sample's highlight would get "pasted" everywhere. We report
    # them separately so the UI can still show what was found.
    HIGHLIGHT_LUM = 150.0
    pens = [c for c in inks if _lum(c) <= HIGHLIGHT_LUM]
    highlights = [c for c in inks if _lum(c) > HIGHLIGHT_LUM]
    if not pens:                                   # keep at least the darkest
        pens = [min(inks, key=_lum)]

    return {"background": [int(round(x)) for x in bg],
            "inks": [[int(round(v)) for v in c] for c in pens[:6]],
            "highlights": [[int(round(v)) for v in c] for c in highlights[:6]]}


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
    # Only recolour a pixel to a *coloured* pen when it's genuinely colourful;
    # otherwise fall back to the neutral/dark pen. This stops faintly-tinted
    # black text from being tinted toward the nearest coloured pen.
    if ink_lum[neutral_idx] < 110:
        idx = np.where(s < 0.25, neutral_idx, idx)
    chosen = inks[idx]                                       # H x W x 3

    a = alpha[..., None]
    out = bg[None, None, :] * (1.0 - a) + chosen * a

    # Preserve genuine highlighter regions on *this* page rather than flattening
    # them into a solid pen colour: a highlighter is bright + colourful, so keep
    # its original pixels. The dark text under the highlight isn't bright, so it
    # still gets recoloured to the matched pen — we keep the highlight, clean the
    # ink, and never invent a highlight that wasn't there.
    highlight = (s > 0.25) & (lum > 160.0)
    out = np.where(highlight[..., None], rgb, out)

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


def watermark_geometry(size: tuple[int, int], crop: dict | None,
                       center_margins: bool, cropped: bool) -> tuple:
    """
    Work out where the watermark goes on one page.

    Returns (sheet_rect, centre_xy), both in pixels of the image being
    watermarked:

      sheet_rect  the whole sheet (x0, y0, x1, y1). The overlay's size and its
                  dx/dy offsets are *always* measured against this, so flipping
                  the anchor moves the mark without resizing it.
      centre_xy   the point the mark is centred on — the sheet's centre, or the
                  centre of the margin box (the region the crop keeps) when
                  `center_margins` is set.

    `cropped` says whether the image has already had the crop applied. When it
    has, the sheet is reconstructed by scaling the kept region back up, so it
    extends past the image edges and gets clipped — exactly as it would if the
    page had been watermarked before cropping. When it hasn't (the main
    on-screen preview shows the uncropped page under draggable guides), the
    margin box is the crop inset rect instead.
    """
    W, H = size
    c = crop or {}
    left = float(c.get("left", 0) or 0)
    top = float(c.get("top", 0) or 0)
    right = float(c.get("right", 0) or 0)
    bottom = float(c.get("bottom", 0) or 0)

    if cropped:
        kw = max(1e-6, 1.0 - left - right)
        kh = max(1e-6, 1.0 - top - bottom)
        full_w, full_h = W / kw, H / kh
        x0, y0 = -left * full_w, -top * full_h
        sheet = (x0, y0, x0 + full_w, y0 + full_h)
        margin = (0.0, 0.0, float(W), float(H))
    else:
        sheet = (0.0, 0.0, float(W), float(H))
        margin = (left * W, top * H, (1.0 - right) * W, (1.0 - bottom) * H)

    box = margin if center_margins else sheet
    return sheet, ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def apply_watermark(img: Image.Image,
                    opacity: float = DEFAULT_WATERMARK_OPACITY,
                    scale: float = 1.0,
                    rotate: float = 0.0,
                    dx: float = 0.0,
                    dy: float = 0.0,
                    sheet: tuple | None = None,
                    centre: tuple | None = None,
                    align_ink: bool = False) -> Image.Image:
    """
    Composite the branding watermark over a finished page.

    The overlay is sized relative to a "cover the sheet" baseline (preserving its
    aspect so the logo isn't distorted), optionally rotated, then placed on a
    transparent full-page canvas and alpha-composited.

    Controls (all optional, defaults reproduce the classic full-page look):
      opacity   : alpha multiplier (1.0 == the PNG's native, faint opacity).
      scale     : size relative to sheet-cover (1.0 == exactly covers the sheet;
                  <1 shrinks it into a movable stamp, >1 enlarges it).
      rotate    : rotation in degrees (counter-clockwise positive).
      dx, dy    : offset as a fraction of the sheet's width/height (+dx right,
                  +dy down).
      sheet     : the sheet rect (x0, y0, x1, y1) in pixels — see
                  `watermark_geometry`. Defaults to the whole image.
      centre    : the point to centre on. Defaults to the sheet's centre.
      align_ink : centre the mark's *visible* pixels rather than the PNG canvas.
                  The artwork carries a lot of transparent padding, so without
                  this the logo lands well off the point it was aimed at.
    """
    if opacity <= 0:
        return img
    wm = _load_watermark()
    if wm is None:
        return img

    W, H = img.size
    ww, wh = wm.size

    if sheet is None:
        sheet = (0.0, 0.0, float(W), float(H))
    x0, y0, x1, y1 = (float(v) for v in sheet)
    sw = max(1.0, x1 - x0)
    sh = max(1.0, y1 - y0)
    cx0, cy0 = ((x0 + x1) / 2.0, (y0 + y1) / 2.0) if centre is None \
        else (float(centre[0]), float(centre[1]))

    # Baseline: the scale that makes the overlay just cover the sheet. Everything
    # is expressed as a multiple of that, so `scale=1` == "covers the page" — and
    # re-anchoring the centre never changes the mark's size.
    cover = max(sw / ww, sh / wh)
    s = cover * max(scale, 0.01)
    nw, nh = max(1, round(ww * s)), max(1, round(wh * s))
    layer = wm.resize((nw, nh), Image.LANCZOS)

    if abs(rotate) > 1e-3:
        layer = layer.rotate(rotate, resample=Image.BICUBIC, expand=True,
                             fillcolor=(0, 0, 0, 0))

    if abs(opacity - 1.0) > 1e-3:
        arr = np.asarray(layer, dtype=np.float32)
        arr[..., 3] = np.clip(arr[..., 3] * opacity, 0.0, 255.0)
        layer = Image.fromarray(arr.astype(np.uint8), "RGBA")

    # Place the (possibly smaller/rotated) overlay onto a transparent page-sized
    # canvas, centred on the anchor point plus the requested offset. The layer
    # can hang off the page (enlarged, rotated, or anchored to the whole sheet
    # while the image is already cropped), so clip to the visible part.
    lw, lh = layer.size
    ax, ay = lw / 2.0, lh / 2.0
    if align_ink:
        ink = layer.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
        if ink:
            ax, ay = (ink[0] + ink[2]) / 2.0, (ink[1] + ink[3]) / 2.0
    cx = int(round(cx0 - ax + dx * sw))
    cy = int(round(cy0 - ay + dy * sh))
    sx, sy = max(0, -cx), max(0, -cy)
    px, py = max(0, cx), max(0, cy)
    vw, vh = min(lw - sx, W - px), min(lh - sy, H - py)
    if vw <= 0 or vh <= 0:
        return img.convert("RGB")

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.alpha_composite(layer.crop((sx, sy, sx + vw, sy + vh)), (px, py))

    base = img.convert("RGBA")
    return Image.alpha_composite(base, canvas).convert("RGB")


def paste_overlays(img: Image.Image, overlays: list[dict], crop: dict | None,
                   palette: dict | None, src_render) -> Image.Image:
    """
    Composite content pulled from other pages ("merge pages") onto a page.

    Each overlay dict carries the source page's full transform (angle, rotate,
    crop, removals, color) plus its placement on the destination: x, y are the
    top-left corner and w the width, all fractions of the destination's
    *uncropped* deskewed page — the same space the editor's guides live in.
    `crop` is the destination crop already applied to `img` (None if it wasn't)
    so placements map through it. `src_render(ov)` returns the source page's
    raw raster (or None to skip that overlay). A bad overlay never kills the
    page — it's just skipped.
    """
    if not overlays:
        return img
    c = crop or {}
    cl = float(c.get("left", 0) or 0)
    ct = float(c.get("top", 0) or 0)
    cr = float(c.get("right", 0) or 0)
    cb = float(c.get("bottom", 0) or 0)
    kw = max(1e-6, 1.0 - cl - cr)
    kh = max(1e-6, 1.0 - ct - cb)
    W, H = img.size
    full_w, full_h = W / kw, H / kh          # uncropped-equivalent sheet size
    for ov in overlays:
        try:
            base = src_render(ov)
            if base is None:
                continue
            src = process_page(base, float(ov.get("angle", 0) or 0),
                               ov.get("crop") or {}, ov.get("color"),
                               palette, ov.get("removals"),
                               float(ov.get("rotate", 0) or 0))
            if ov.get("shapes"):
                src = draw_shapes(src, ov["shapes"], ov.get("crop"))
            ow = max(1, int(round(float(ov.get("w", 0.5)) * full_w)))
            oh = max(1, int(round(ow * src.height / max(1, src.width))))
            src = src.resize((ow, oh), Image.LANCZOS)
            x = int(round((float(ov.get("x", 0)) - cl) * full_w))
            y = int(round((float(ov.get("y", 0)) - ct) * full_h))
            img.paste(src, (x, y))           # paste clips at the page edges
        except Exception:
            continue
    return img


def _parse_hex_color(s) -> tuple[int, int, int]:
    """'#rrggbb' (or '#rgb') -> RGB tuple; anything malformed -> white."""
    try:
        s = str(s or "").strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            return (255, 255, 255)
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def draw_shapes(img: Image.Image, shapes: list[dict] | None,
                crop: dict | None = None) -> Image.Image:
    """
    Burn user-drawn filled polygons ("shape filler") into a page.

    Each shape is {"pts": [[x, y], ...], "color": "#rrggbb"} with points as
    fractions of the *uncropped* deskewed page (post-removals) — the space the
    editor's main preview shows. `crop` is the crop already applied to `img`
    (None if it wasn't), so points map through the insets exactly like
    paste_overlays. Edges are anti-aliased via a 2x supersampled mask at
    preview sizes (skipped at large export sizes, where aliasing is invisible).
    A malformed shape never kills the page — it's just skipped.
    """
    if not shapes:
        return img
    c = crop or {}
    cl = float(c.get("left", 0) or 0)
    ct = float(c.get("top", 0) or 0)
    cr = float(c.get("right", 0) or 0)
    cb = float(c.get("bottom", 0) or 0)
    kw = max(1e-6, 1.0 - cl - cr)
    kh = max(1e-6, 1.0 - ct - cb)
    W, H = img.size
    full_w, full_h = W / kw, H / kh          # uncropped-equivalent sheet size
    ss = 2 if max(W, H) <= 1600 else 1       # supersample small renders only
    draw = ImageDraw.Draw(img) if ss == 1 else None
    for sh in shapes:
        try:
            pts = sh.get("pts") or []
            if len(pts) < 3:
                continue
            xy = [((float(p[0]) - cl) * full_w, (float(p[1]) - ct) * full_h)
                  for p in pts]
            col = _parse_hex_color(sh.get("color"))
            if ss == 1:
                draw.polygon(xy, fill=col)
            else:
                mask = Image.new("L", (W * ss, H * ss), 0)
                ImageDraw.Draw(mask).polygon(
                    [(x * ss, y * ss) for x, y in xy], fill=255)
                mask = mask.resize((W, H), Image.BILINEAR)
                img.paste(Image.new("RGB", (W, H), col), (0, 0), mask)
        except Exception:
            continue
    return img


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_page(img: Image.Image, angle: float, crop: dict,
                 color: dict | None = None,
                 palette: dict | None = None,
                 removals: list[dict] | None = None,
                 rotate: float = 0.0) -> Image.Image:
    """Full transform for one page: rotate -> deskew -> removals -> crop ->
    scan/colour/palette.

    Rotation runs first: crop and removal fractions are recorded against the
    rotated preview, so they must be applied in that same space. Removals run
    before the crop because their fractions are recorded on the uncropped
    deskewed preview — applying them after the crop would shift the removed
    band by the top-crop inset.
    """
    drem = apply_removals(deskew_image(rotate_page(img, rotate), angle), removals)
    dcrop = apply_crop(drem, crop)
    return finish_color(dcrop, color, palette)

