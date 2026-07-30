"""
Branded border frames for the "Borders" tab — integrated from the
arivihan-notes-composer repo (compositor.py / front_pages.py / html_header.py).

A border is a full-page branded frame PNG with a transparent slot where the
notes content shows through, plus a header pill at the top where the chapter
number + name are stamped ("CHAPTER - 05 | Name").

Assets live per class so 10th/12th can carry different designs:

    borders/
      12th/
        config.json     # template size, slot rect, header layout — TUNE HERE
        templates/      # frame PNGs, cycled: page 1 -> tpl 1, page 2 -> tpl 2 …
        front_pages/    # cover images named "<Board>__<Stream>.<ext>"
      10th/             # same structure; covers are "<Board>.<ext>" (no
                        # streams in 10th). A config "templates_from": "<cls>"
                        # key can point a class at another class's templates/.
    fonts/Urbanist-var.ttf

The frame is NEVER resized or distorted: every composed page is exactly the
template's native size. The cropped/rotated notes content is contain-fitted
into the fixed slot — scaled to fit whole (no ink trimmed) and centred, with
white space filling whatever the aspect difference leaves over. Adjusting the
margins only changes how the content sits inside the slot; the live preview
mirrors this around the guides.

The chapter header is real HTML/CSS rasterised with headless Chromium
(Playwright), exactly like the composer. When Playwright or its browser is
missing, the same layout is drawn with Pillow instead so exports still work.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import imaging

HERE = Path(__file__).parent
BORDERS_DIR = HERE / "borders"
FONT_PATH = HERE / "fonts" / "Urbanist-var.ttf"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SEP = "__"                      # "<Board>__<Stream>.<ext>" cover filenames
                                # ("<Board>.<ext>" for streamless classes)
_SAFE_CLS = re.compile(r"^[A-Za-z0-9 _-]+$")

try:
    from playwright.sync_api import sync_playwright
except Exception:               # not installed — PIL fallback takes over
    sync_playwright = None


def natural_key(name: str) -> list:
    """Sort page-2 before page-10, template-2 before template-10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------
# Classes / config / templates / covers
# --------------------------------------------------------------------------

def list_classes() -> list[str]:
    """Class folders (e.g. 10th, 12th) that carry a border config."""
    if not BORDERS_DIR.is_dir():
        return []
    out = [p.name for p in BORDERS_DIR.iterdir()
           if p.is_dir() and (p / "config.json").is_file()]
    return sorted(out, key=natural_key)


def class_dir(cls: str) -> Path:
    if not cls or not _SAFE_CLS.match(cls):
        raise LookupError("bad class name")
    path = BORDERS_DIR / cls
    if not path.is_dir() or not (path / "config.json").is_file():
        raise LookupError(f"no border assets for class {cls!r}")
    return path


def load_config(cls: str) -> dict:
    with open(class_dir(cls) / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


def list_templates(cls: str) -> list[Path]:
    src = load_config(cls).get("templates_from") or cls
    tdir = class_dir(src) / "templates"
    if not tdir.is_dir():
        return []
    files = [p for p in tdir.iterdir()
             if p.suffix.lower() in IMG_EXTS and not p.name.startswith(("_", "."))]
    return sorted(files, key=lambda p: natural_key(p.name))


def load_template(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def scan_front_pages(cls: str) -> list[dict]:
    """[{board, stream, file}] for every valid cover in the class folder."""
    fdir = class_dir(cls) / "front_pages"
    if not fdir.is_dir():
        return []
    out = []
    for p in sorted(fdir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS or p.name.startswith(("_", ".")):
            continue
        stem = p.stem
        if SEP in stem:
            board, stream = (s.strip() for s in stem.split(SEP, 1))
        else:                   # board-only cover (classes without streams)
            board, stream = stem.strip(), ""
        if board:
            out.append({"board": board, "stream": stream, "file": p.name})
    return out


def find_cover(cls: str, board: str, stream: str) -> Path | None:
    for c in scan_front_pages(cls):
        if c["board"] == board and c["stream"] == stream:
            return class_dir(cls) / "front_pages" / c["file"]
    return None


def boards_with_covers(cls: str, stream: str = "") -> list[tuple[str, Path]]:
    """
    [(board, cover path)] for every board that has a cover for `stream`.

    Backs the Borders tab's "All boards" export: the notes are rendered once
    and each board gets its own PDF with its own cover as page 1. Streamless
    classes (10th) pass stream="" and match every board-only cover.
    """
    fdir = class_dir(cls) / "front_pages"
    out = [(c["board"], fdir / c["file"])
           for c in scan_front_pages(cls) if c["stream"] == (stream or "")]
    return sorted(out, key=lambda bc: natural_key(bc[0]))


def load_cover(path: Path, size, bg=(255, 255, 255, 255)) -> Image.Image:
    """
    Cover image scaled to exactly `size` (the frame page size).

    Near-matching aspects get the classic cover-fit (fill, minimal crop).
    A big mismatch — e.g. the 9:16 cover art on an A4 page — would crop real
    artwork away, so instead the whole cover is fitted inside and the slack
    is filled with a blurred stretch of the cover's own edges (the standard
    letterbox-fill look), keeping every element of the design visible.
    """
    size = tuple(int(v) for v in size)
    cover = Image.open(path).convert("RGBA")
    if cover.size == size:
        return cover
    tw, th = size
    ca, ta = cover.width / cover.height, tw / th
    if abs(ca - ta) / ta < 0.04:
        scale = max(tw / cover.width, th / cover.height)
        nw, nh = round(cover.width * scale), round(cover.height * scale)
        cover = cover.resize((nw, nh), Image.Resampling.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        return cover.crop((left, top, left + tw, top + th))

    scale = min(tw / cover.width, th / cover.height)
    nw = max(1, round(cover.width * scale))
    nh = max(1, round(cover.height * scale))
    fit = cover.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, tuple(bg))
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    blur = ImageFilter.GaussianBlur(30)
    if nw < tw:                                   # side bars (full height)
        strip = max(8, nw // 20)
        if x0 > 0:
            canvas.paste(fit.crop((0, 0, strip, nh))
                         .resize((x0, nh), Image.LANCZOS).filter(blur), (0, y0))
        rpad = tw - nw - x0
        if rpad > 0:
            canvas.paste(fit.crop((nw - strip, 0, nw, nh))
                         .resize((rpad, nh), Image.LANCZOS).filter(blur),
                         (x0 + nw, y0))
    if nh < th:                                   # top/bottom bars (full width)
        strip = max(8, nh // 20)
        if y0 > 0:
            canvas.paste(fit.crop((0, 0, nw, strip))
                         .resize((nw, y0), Image.LANCZOS).filter(blur), (x0, 0))
        bpad = th - nh - y0
        if bpad > 0:
            canvas.paste(fit.crop((0, nh - strip, nw, nh))
                         .resize((nw, bpad), Image.LANCZOS).filter(blur),
                         (x0, y0 + nh))
    canvas.paste(fit, (x0, y0))
    return canvas


def options() -> dict:
    """Everything the Borders tab needs, for every class, in one payload."""
    classes = []
    for cls in list_classes():
        cfg = load_config(cls)
        boards: list[str] = []
        streams_by_board: dict[str, list[str]] = {}
        for c in scan_front_pages(cls):
            if c["board"] not in streams_by_board:
                streams_by_board[c["board"]] = []
                boards.append(c["board"])
            if c["stream"] and c["stream"] not in streams_by_board[c["board"]]:
                streams_by_board[c["board"]].append(c["stream"])
        tpls = list_templates(cls)
        classes.append({
            "name": cls,
            "template_count": len(tpls),
            # bumps when template files change -> busts the browser's cache
            "version": int(max((p.stat().st_mtime for p in tpls), default=0)),
            "boards": boards,
            "streams_by_board": streams_by_board,
            "config": {
                "template_size": cfg["template_size"],
                "slot": cfg["slot"],
                "header": cfg.get("header", {}),
                "background_color": cfg.get("background_color",
                                            [255, 255, 255, 255]),
            },
        })
    return {"classes": classes}


# --------------------------------------------------------------------------
# Downscaled template previews (for the live in-browser overlay)
# --------------------------------------------------------------------------

_PREVIEW_CACHE: dict = {}


def template_preview_png(cls: str, index: int, width: int = 1000) -> bytes:
    paths = list_templates(cls)
    if not paths:
        raise LookupError(f"no border templates installed for class {cls!r}")
    p = paths[index % len(paths)]
    width = max(200, min(1600, int(width)))
    key = (cls, p.name, p.stat().st_mtime, width)
    hit = _PREVIEW_CACHE.get(key)
    if hit is not None:
        return hit
    im = Image.open(p).convert("RGBA")
    scale = width / im.width
    im = im.resize((width, max(1, round(im.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    data = buf.getvalue()
    if len(_PREVIEW_CACHE) > 60:      # tiny LRU-ish cap; previews rebuild fast
        _PREVIEW_CACHE.clear()
    _PREVIEW_CACHE[key] = data
    return data


# --------------------------------------------------------------------------
# Composition: contain-fit the page into the FIXED frame slot
# --------------------------------------------------------------------------

def _paper_color(img: Image.Image) -> tuple:
    """
    Median colour of the page's bright (paper) pixels.

    The slot's slack is filled with this instead of hard white, so the added
    space blends seamlessly with the sheet — the content reads as one full
    page inside the frame rather than a smaller page floating on white. Shared
    with the grown-sheet padding in imaging.apply_crop.
    """
    return imaging.paper_color(img)


def _clampf(v, lo=0.2, hi=4.0, dflt=1.0) -> float:
    try:
        return min(hi, max(lo, float(v or dflt)))
    except (TypeError, ValueError):
        return dflt


def _clampa(v) -> float:
    """Content alignment in the slot: -1 (top/left) .. 0 (centred) .. 1."""
    try:
        return min(1.0, max(-1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


# "fill" mode stretches at most this much toward the slot's shape — enough to
# absorb small aspect mismatches invisibly; anything beyond is cover-cropped.
FILL_MAX_STRETCH = 1.15


def fit_scales(cw: float, ch: float, sw: float, sh: float,
               fit_mode: str) -> tuple[float, float]:
    """Per-axis scale factors that place a cw x ch page into a sw x sh slot.

    page  -- contain: whole page visible, slack padded.
    width -- composer's "not A4": width fills exactly, vertical pads/crops.
    fill  -- fill the slot completely: stretch up to FILL_MAX_STRETCH toward
             the slot's aspect (barely visible on notes), then cover-crop
             whatever mismatch remains. No padding ever.
    """
    if fit_mode == "width":
        k = sw / cw
        return k, k
    if fit_mode == "fill":
        need = (sw / sh) / (cw / ch)      # >1: slot is wider than the page
        sxs = min(need, FILL_MAX_STRETCH) if need > 1 else 1.0
        sys_ = min(1.0 / need, FILL_MAX_STRETCH) if need < 1 else 1.0
        k = max(sw / (cw * sxs), sh / (ch * sys_))
        return k * sxs, k * sys_
    k = min(sw / cw, sh / ch)             # page (contain)
    return k, k


def compose_bordered(content: Image.Image, tpl: Image.Image, cfg: dict,
                     header: Image.Image | None = None,
                     zoom: float = 1.0,
                     stretch_w: float = 1.0,
                     stretch_h: float = 1.0,
                     fit_mode: str = "page",
                     align_x: float = 0.0,
                     align_y: float = 0.0) -> Image.Image:
    """
    Put `content` (the processed, cropped page) behind the border frame.

    The frame keeps its native size — every output page is exactly
    `template_size`. `fit_mode` picks how the page meets the slot (see
    fit_scales); slack is padded in the page's own paper tone and overflow
    is cropped under the frame. `zoom` multiplies the fit uniformly and
    `stretch_w`/`stretch_h` scale each axis independently on top.
    `align_x`/`align_y` place the content inside the slot (-1 = top/left
    edge, 0 = centred, 1 = bottom/right edge — the editor's Move tab).
    The template's transparent slot shows the content through it; the
    header goes on top.
    """
    slot = cfg["slot"]
    bg = tuple(cfg.get("background_color", [255, 255, 255, 255]))
    canvas = Image.new("RGBA", tpl.size, bg)

    zoom = _clampf(zoom)
    stretch_w = _clampf(stretch_w)
    stretch_h = _clampf(stretch_h)
    kx, ky = fit_scales(content.width, content.height,
                        slot["w"], slot["h"], fit_mode)
    nw = max(1, round(content.width * kx * zoom * stretch_w))
    nh = max(1, round(content.height * ky * zoom * stretch_h))
    fitted = content.convert("RGB").resize((nw, nh), Image.LANCZOS)
    pane = Image.new("RGB", (slot["w"], slot["h"]), _paper_color(content))
    # aligned paste (centred by default); PIL clips whatever the zoom or the
    # alignment pushes past the pane edges
    pane.paste(fitted, (round((slot["w"] - nw) / 2 * (1.0 + _clampa(align_x))),
                        round((slot["h"] - nh) / 2 * (1.0 + _clampa(align_y)))))
    canvas.paste(pane, (slot["x"], slot["y"]))

    canvas = Image.alpha_composite(canvas, tpl)

    if header is not None:
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(header, (0, 0))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


# --------------------------------------------------------------------------
# Header geometry scaling
# --------------------------------------------------------------------------

# x-positions and type metrics follow the horizontal stretch so the text keeps
# its designed relationship to the pill, divider and stamp; y-positions follow
# the vertical stretch. Glyphs themselves are never distorted.
_H_XKEYS = ("divider_x", "label_left", "label_right_gap", "text_left",
            "name_gap", "name_max_right")
_H_FONTKEYS = ("font_size", "letter_spacing", "label_min_font_size",
               "name_min_font_size", "name_one_line_min")
_H_YKEYS = ("center_y", "canvas_height", "name_max_height")


def scaled_header(h: dict, sx: float, sy: float) -> dict:
    out = dict(h)
    for k in _H_XKEYS:
        if k in out:
            out[k] = out[k] * sx
    for k in _H_FONTKEYS:
        if k in out:
            out[k] = out[k] * sx
    for k in _H_YKEYS:
        if k in out:
            out[k] = out[k] * sy
    dv = out.get("divider")
    if isinstance(dv, dict):
        dv = dict(dv)
        dv["width"] = dv.get("width", 3) * sx
        dv["height"] = dv.get("height", 54) * sy
        dv["radius"] = dv.get("radius", 2) * min(sx, sy)
        out["divider"] = dv
    return out


# --------------------------------------------------------------------------
# Header rendering — HTML/CSS via Playwright, PIL fallback
# --------------------------------------------------------------------------

_font_uri_cache: dict = {}


def _font_data_uri() -> str:
    key = str(FONT_PATH)
    if key not in _font_uri_cache:
        b64 = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
        _font_uri_cache[key] = f"data:font/ttf;base64,{b64}"
    return _font_uri_cache[key]


def _common_css(h: dict, tw: int, ch: int, font_uri: str) -> str:
    return f"""
  @font-face {{
    font-family:'Urbanist';
    src:url('{font_uri}') format('truetype');
    font-weight:100 900;
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:{tw}px;height:{ch}px;background:transparent}}
  .txt{{
    position:absolute;
    font-family:'Urbanist',sans-serif;
    font-weight:{h['font_weight']};
    font-size:{h['font_size']}px;
    letter-spacing:{h['letter_spacing']}px;
    line-height:1;white-space:nowrap;
  }}
  .label{{color:{h['label_color']}}}
  .name{{color:{h['name_color']}}}
"""


def _html_anchored(number: str, name: str, h: dict, tw: int, ch: int,
                   font_uri: str) -> str:
    """Template already contains the "|" divider: label left-anchored at a
    fixed x, name left-anchored right of the divider."""
    number = _html.escape(number.strip())
    name = _html.escape(name.strip())
    dv = h["divider_x"]
    label_left = h.get("label_left", h.get("text_left", 180))
    name_left = dv + h["name_gap"]
    lh = h.get("name_line_height", 1.06)
    label = f"{h['label_prefix']}{number}" if number else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{_common_css(h, tw, ch, font_uri)}
  #label{{left:{label_left}px;top:{h['center_y']}px;transform:translateY(-50%);
         text-align:left;white-space:nowrap}}
  #name{{left:{name_left}px;top:{h['center_y']}px;transform:translateY(-50%);
        text-align:left;white-space:nowrap;line-height:{lh};
        overflow-wrap:break-word}}
</style></head><body>
  <div id="label" class="txt label">{label}</div>
  <div id="name" class="txt name">{name}</div>
</body></html>"""


def _html_flex(number: str, name: str, h: dict, tw: int, ch: int,
               font_uri: str) -> str:
    """Whole unit incl. a drawn divider as one flex row (no divider in art)."""
    number = _html.escape(number.strip())
    name = _html.escape(name.strip())
    dv = h.get("divider", {"width": 3, "height": 54, "color": "#12303a", "radius": 2})
    label = f'<span class="label">{h["label_prefix"]}{number}</span>' if number else ""
    divider = ('<span class="divider"></span>'
               if number and name and dv.get("width", 0) > 0 else "")
    name_span = f'<span class="name">{name}</span>' if name else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{_common_css(h, tw, ch, font_uri)}
  #bar{{left:{h['text_left']}px;top:{h['center_y']}px;transform:translateY(-50%);
       display:flex;align-items:center;
       max-width:{h['name_max_right'] - h['text_left']}px;}}
  #bar.txt{{white-space:nowrap}}
  .divider{{display:inline-block;width:{dv['width']}px;height:{dv['height']}px;
    background:{dv['color']};border-radius:{dv['radius']}px;
    margin:0 {h['name_gap']}px;flex:0 0 auto;}}
</style></head><body>
  <div id="bar" class="txt">{label}{divider}{name_span}</div>
</body></html>"""


_LABEL_FIT_JS = """([avail, minFs]) => {
    const el = document.getElementById('label');
    if (!el || !el.textContent) return;
    let fs = parseFloat(getComputedStyle(el).fontSize);
    while (el.scrollWidth > avail && fs > minFs) {
        fs -= 1; el.style.fontSize = fs + 'px';
    }
}"""

_NAME_FIT_JS = """([zoneW, oneLineMin, minFs, maxLines, maxH, lineH, startFs]) => {
    const el = document.getElementById('name');
    if (!el) return;
    el.style.whiteSpace = 'nowrap';
    el.style.width = 'auto';
    el.style.textWrap = '';
    let fs = startFs; el.style.fontSize = fs + 'px';
    while (el.scrollWidth > zoneW && fs > oneLineMin) {
        fs -= 1; el.style.fontSize = fs + 'px';
    }
    if (el.scrollWidth <= zoneW) return;
    el.style.whiteSpace = 'normal';
    el.style.width = zoneW + 'px';
    el.style.textWrap = 'balance';
    fs = startFs; el.style.fontSize = fs + 'px';
    const overflow = () => {
        const lines = Math.round(el.offsetHeight / (fs * lineH));
        return lines > maxLines
            || el.offsetHeight > maxH
            || el.scrollWidth > el.clientWidth + 1;
    };
    while (overflow() && fs > minFs) {
        fs -= 1; el.style.fontSize = fs + 'px';
    }
}"""

_BAR_FIT_JS = """([maxW, minFs]) => {
    const el = document.getElementById('bar');
    if (!el) return;
    let fs = parseFloat(getComputedStyle(el).fontSize);
    while (el.scrollWidth > maxW && fs > minFs) {
        fs -= 1; el.style.fontSize = fs + 'px';
    }
}"""


class HeaderRenderer:
    """
    Renders the chapter-header overlay for each page geometry of an export.

    Use as a context manager so one headless browser serves the whole job.
    render() caches by rounded geometry, so pages that share a crop reuse the
    same overlay. Falls back to Pillow drawing when Chromium is unavailable.
    """

    def __init__(self, cfg: dict, number: str, name: str):
        self.cfg = cfg
        self.number = (number or "").strip()
        self.name = (name or "").strip()
        self._cache: dict = {}
        self._pw = None
        self._browser = None
        self._page = None
        self._pw_broken = False

    # -- context management -------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._browser = self._pw = self._page = None
        return False

    # -- public -------------------------------------------------------------
    def render(self, out_w: int, sx: float, sy: float) -> Image.Image | None:
        """Overlay for one composed page (full width, header-band height)."""
        if not self.number and not self.name:
            return None
        key = (int(round(out_w)), round(sx, 3), round(sy, 3))
        if key in self._cache:
            return self._cache[key]
        h = scaled_header(self.cfg["header"], sx, sy)
        ch = max(1, int(round(h.get("canvas_height", 360))))
        tw = max(1, int(round(out_w)))
        img = None
        if sync_playwright is not None and not self._pw_broken:
            try:
                img = self._render_playwright(h, tw, ch)
            except Exception:
                self._pw_broken = True      # e.g. chromium not installed
        if img is None:
            img = self._render_pil(h, tw, ch)
        self._cache[key] = img
        return img

    # -- playwright path ----------------------------------------------------
    def _ensure_page(self, tw: int, ch: int):
        if self._pw is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                args=["--force-color-profile=srgb"])
            self._page = self._browser.new_page(
                viewport={"width": tw, "height": ch}, device_scale_factor=1)
        else:
            self._page.set_viewport_size({"width": tw, "height": ch})
        return self._page

    def _render_playwright(self, h: dict, tw: int, ch: int) -> Image.Image:
        anchored = h.get("use_template_divider", False)
        font_uri = _font_data_uri()
        if anchored:
            html = _html_anchored(self.number, self.name, h, tw, ch, font_uri)
        else:
            html = _html_flex(self.number, self.name, h, tw, ch, font_uri)

        page = self._ensure_page(tw, ch)
        page.set_content(html, wait_until="networkidle")

        min_fs = h.get("name_min_font_size", 16)
        max_lines = h.get("name_max_lines", 2)
        line_h = h.get("name_line_height", 1.08)
        if anchored:
            label_left = h.get("label_left", h.get("text_left", 180))
            label_avail = h["divider_x"] - label_left - h.get("label_right_gap", 20)
            page.evaluate(_LABEL_FIT_JS,
                          [label_avail, h.get("label_min_font_size", 24)])
            zone_w = h["name_max_right"] - (h["divider_x"] + h["name_gap"])
            page.evaluate(_NAME_FIT_JS,
                          [zone_w, h.get("name_one_line_min", 26), min_fs,
                           max_lines, h.get("name_max_height", 92), line_h,
                           h["font_size"]])
        else:
            page.evaluate(_BAR_FIT_JS,
                          [h["name_max_right"] - h["text_left"], min_fs])

        png = page.screenshot(omit_background=True)
        return Image.open(io.BytesIO(png)).convert("RGBA")

    # -- PIL fallback path ---------------------------------------------------
    def _render_pil(self, h: dict, tw: int, ch: int) -> Image.Image:
        img = Image.new("RGBA", (tw, ch), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        weight = int(h.get("font_weight", 700))
        ls = float(h.get("letter_spacing", 0))
        cy = h.get("center_y", ch / 2)
        anchored = h.get("use_template_divider", False)

        label = f"{h['label_prefix']}{self.number}" if self.number else ""
        name = self.name

        if anchored:
            label_left = h.get("label_left", h.get("text_left", 180))
            if label:
                avail = h["divider_x"] - label_left - h.get("label_right_gap", 20)
                fs = self._fit_line(label, h["font_size"], ls, weight, avail,
                                    h.get("label_min_font_size", 24))
                self._draw_ls(draw, (label_left, cy), label,
                              self._font(fs, weight), h["label_color"], ls)
            if name:
                name_left = h["divider_x"] + h["name_gap"]
                zone_w = h["name_max_right"] - name_left
                self._draw_name(draw, name, name_left, cy, zone_w, h, ls, weight)
        else:
            # single flex row: LABEL | NAME with a drawn divider
            dv = h.get("divider", {"width": 3, "height": 54,
                                   "color": "#12303a", "radius": 2})
            gap = h.get("name_gap", 24)
            max_w = h["name_max_right"] - h["text_left"]
            fs = h["font_size"]
            min_fs = h.get("name_min_font_size", 16)

            def row_w(f):
                w = 0.0
                if label:
                    w += self._text_w(label, f, ls, weight)
                if label and name and dv.get("width", 0) > 0:
                    w += 2 * gap + dv["width"]
                if name:
                    w += self._text_w(name, f, ls, weight)
                return w

            while row_w(fs) > max_w and fs > min_fs:
                fs -= 1
            x = h["text_left"]
            if label:
                self._draw_ls(draw, (x, cy), label, self._font(fs, weight),
                              h["label_color"], ls)
                x += self._text_w(label, fs, ls, weight)
            if label and name and dv.get("width", 0) > 0:
                x += gap
                x0, x1 = x, x + dv["width"]
                y0, y1 = cy - dv["height"] / 2, cy + dv["height"] / 2
                draw.rounded_rectangle([x0, y0, x1, y1],
                                       radius=dv.get("radius", 2),
                                       fill=dv.get("color", "#12303a"))
                x = x1 + gap
            if name:
                self._draw_ls(draw, (x, cy), name, self._font(fs, weight),
                              h["name_color"], ls)
        return img

    def _draw_name(self, draw, name, left, cy, zone_w, h, ls, weight):
        """One line if it fits (shrinking a little), else wrap to <=N lines."""
        fs = self._fit_line(name, h["font_size"], ls, weight, zone_w,
                            h.get("name_one_line_min", 26))
        if self._text_w(name, fs, ls, weight) <= zone_w:
            self._draw_ls(draw, (left, cy), name, self._font(fs, weight),
                          h["name_color"], ls)
            return
        max_lines = int(h.get("name_max_lines", 2))
        min_fs = int(round(h.get("name_min_font_size", 16)))
        line_h = h.get("name_line_height", 1.06)
        max_h = h.get("name_max_height", 92)
        fs = int(round(h["font_size"]))
        lines = [name]
        while fs > min_fs:
            lines = self._wrap(name, fs, ls, weight, zone_w)
            if (len(lines) <= max_lines and len(lines) * fs * line_h <= max_h
                    and all(self._text_w(l, fs, ls, weight) <= zone_w
                            for l in lines)):
                break
            fs -= 1
        top = cy - (len(lines) - 1) * fs * line_h / 2
        for i, line in enumerate(lines):
            self._draw_ls(draw, (left, top + i * fs * line_h), line,
                          self._font(fs, weight), h["name_color"], ls)

    def _wrap(self, text, fs, ls, weight, zone_w):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if cur and self._text_w(trial, fs, ls, weight) > zone_w:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines or [text]

    def _fit_line(self, text, start_fs, ls, weight, avail, min_fs):
        fs = int(round(start_fs))
        min_fs = int(round(min_fs))
        while self._text_w(text, fs, ls, weight) > avail and fs > min_fs:
            fs -= 1
        return fs

    @staticmethod
    def _font(size, weight) -> ImageFont.FreeTypeFont:
        font = ImageFont.truetype(str(FONT_PATH), max(8, int(round(size))))
        try:
            font.set_variation_by_axes([weight])
        except Exception:
            pass
        return font

    def _text_w(self, text, fs, ls, weight) -> float:
        font = self._font(fs, weight)
        return font.getlength(text) + ls * max(0, len(text) - 1)

    def _draw_ls(self, draw, xy, text, font, fill, ls):
        """draw.text with CSS-style letter-spacing, anchored left-middle."""
        x, y = xy
        if ls <= 0.01:
            draw.text((x, y), text, font=font, fill=fill, anchor="lm")
            return
        for chch in text:
            draw.text((x, y), chch, font=font, fill=fill, anchor="lm")
            x += font.getlength(chch) + ls
