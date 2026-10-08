"""
Patchwork — typed text becomes a styled image you drop on a page.

The user types plain text in the Patchwork panel; a headless Chromium lays it
out with the Arivihan `template.css` (Kalam, teal answer boxes, the red double
margin rule) under the "Topic missed by toppers" banner, and the result is
rasterised to a PNG. server.py then hands that PNG to the editor as an ordinary
overlay block, so it moves, resizes and stretches like pulled page content.

Everything it draws with is vendored under patchwork_assets/ — fonts, KaTeX +
mhchem, the banner artwork and template.css itself — so rendering never touches
the network. The page is loaded from a real file:// URL (not set_content) so
those relative asset paths resolve.

Rendered patches live in a registry keyed by id, exactly like spellfix patches:
imaging.py composites by id and never sees the filesystem.
"""

from __future__ import annotations

import html as _html
import re
import threading
import uuid
from pathlib import Path

from PIL import Image

try:
    from playwright.sync_api import sync_playwright
except Exception:                                    # chromium not installed
    sync_playwright = None

HERE = Path(__file__).parent
ASSETS = HERE / "patchwork_assets"
WORK = HERE / "work" / "patchwork"
WORK.mkdir(parents=True, exist_ok=True)

# Playwright's sync API is not thread-safe and Flask serves threaded, so one
# render at a time. A patch takes ~1s, and this is a manual button press.
_LOCK = threading.Lock()

_CHROME_ARGS = [
    "--allow-file-access-from-files",
    "--force-color-profile=srgb",
    "--font-render-hinting=none",
    "--disable-lcd-text",
]

# Layout is authored at this width and rasterised at `scale`, so the PNG is
# 1800px wide by default — plenty for a 300-DPI export at any block size.
DEFAULT_WIDTH = 900
DEFAULT_SCALE = 2

LABELS = {
    "en": {"title": "Topic missed by toppers", "image": "label/topic-missed-en.png"},
    "hi": {"title": "टॉपर्स से छूटे हुए टॉपिक", "image": "label/topic-missed-hi.png"},
    "none": {"title": "No banner", "image": None},
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def png_path(pid: str) -> Path | None:
    """Path of a rendered patch, or None if the id is unknown or unsafe."""
    if not pid or not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", pid):
        return None
    p = WORK / f"{pid}.png"
    return p if p.is_file() else None


def get(pid: str) -> Image.Image | None:
    """RGBA image for a patch id — the resolver imaging.py composites with."""
    p = png_path(pid)
    if p is None:
        return None
    try:
        with Image.open(p) as im:
            return im.convert("RGBA")
    except Exception:
        return None


def available() -> bool:
    """False when Chromium is missing — the UI greys the panel out."""
    return sync_playwright is not None


# ---------------------------------------------------------------------------
# Markup: plain text -> HTML in the template's own classes
# ---------------------------------------------------------------------------
_MATH_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_MATH_INLINE = re.compile(r"\$(.+?)\$", re.S)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def _inline(text: str) -> str:
    """Escape a line, then re-enable $math$ and **bold**."""
    out = _html.escape(text)

    def _disp(m):
        return f'<span class="tex" data-display="1">{m.group(1)}</span>'

    def _inl(m):
        return f'<span class="tex" data-display="0">{m.group(1)}</span>'

    out = _MATH_DISPLAY.sub(_disp, out)
    out = _MATH_INLINE.sub(_inl, out)
    out = _BOLD.sub(lambda m: f"<b>{m.group(1)}</b>", out)
    return out


def to_html(text: str) -> str:
    """
    Turn the panel's plain text into template-styled HTML.

        # Heading                 -> section heading
        - bullet                  -> bulleted list
        | a | b |                 -> comparison table (first row = header)
        ANSWER: x                 -> the teal answer box (rest of the block
                                     joins it until a blank line)
        $x$ / $$x$$               -> KaTeX inline / display
        **bold**                  -> bold
        blank line                -> paragraph break

    Anything else is a plain line. Unknown syntax is never an error — it just
    comes out as text, so a typo degrades to something the user can see.
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        # --- answer box: everything up to a blank line ---------------------
        m = re.match(r"^(?:ANSWER|ANS)\s*[:：]\s*(.*)$", line, re.I)
        if m:
            body = [m.group(1)] if m.group(1) else []
            i += 1
            while i < n and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            inner = "".join(f"<div>{_inline(b)}</div>" for b in body)
            out.append('<div class="pw-ans">'
                       '<div class="pw-ans-label">Answer</div>'
                       f'<div class="pw-ans-body">{inner}</div></div>')
            continue

        # --- heading -------------------------------------------------------
        if line.startswith("#"):
            out.append(f'<div class="pw-hd">{_inline(line.lstrip("#").strip())}</div>')
            i += 1
            continue

        # --- table ---------------------------------------------------------
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 3:
            rows = []
            while i < n:
                s = lines[i].strip()
                if not (s.startswith("|") and s.endswith("|")):
                    break
                cells = [c.strip() for c in s.strip("|").split("|")]
                # A markdown separator row (|---|---|) just marks the header.
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    i += 1
                    continue
                rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                    for r in rows[1:])
                out.append(f'<table class="diff-table"><tr>{head}</tr>{body}</table>')
            continue

        # --- bullet list ---------------------------------------------------
        if re.match(r"^[-*•]\s+", line):
            items = []
            while i < n and re.match(r"^[-*•]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*•]\s+", "", lines[i].strip()))
                i += 1
            lis = "".join(f"<li>{_inline(t)}</li>" for t in items)
            out.append(f"<ul>{lis}</ul>")
            continue

        # --- plain line ----------------------------------------------------
        out.append(f"<p>{_inline(line)}</p>")
        i += 1

    return "".join(out) or "<p></p>"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def _font_face(rel: str) -> str:
    return f"""
@font-face{{font-family:'Kalam';src:url('{rel}/fonts/Kalam-Regular.ttf') format('truetype');
  font-weight:400;font-style:normal;font-display:block;}}
@font-face{{font-family:'Kalam';src:url('{rel}/fonts/Kalam-Bold.ttf') format('truetype');
  font-weight:700;font-style:normal;font-display:block;}}
@font-face{{font-family:'Kalam';src:url('{rel}/fonts/Kalam-Light.ttf') format('truetype');
  font-weight:300;font-style:normal;font-display:block;}}
"""


# Patch-specific styling. template.css supplies the design tokens (teal family,
# --margin-line-color, Kalam) and the content classes (.diff-table,
# .answer-text); this only adds the patch shell: the banner, the red double
# margin rule and the answer box, all built from those same tokens.
_PATCH_CSS = """
html,body{background:transparent;margin:0;padding:0;}
#pw-root{
  width:var(--pw-w);
  background:var(--pw-bg);
  font-family:var(--font-devanagari);
  color:var(--text-dark);
  padding:var(--pw-pad-y) var(--pw-pad-x) calc(var(--pw-pad-y) + 4px);
  box-sizing:border-box;
}
/* Banner rides in the flow above the content and pushes it down, so the
   artwork never sits on top of a line of text. */
#pw-banner{display:block;margin:0 0 14px 0;}
#pw-banner img{display:block;width:var(--pw-banner-w);height:auto;}
#pw-banner.center img{margin:0 auto;}
#pw-banner.right img{margin-left:auto;}

/* The template's red double margin rule, at the patch's own left edge. */
#pw-body{position:relative;padding-left:26px;}
#pw-body::before,#pw-body::after{
  content:"";position:absolute;top:0;height:100%;width:2px;
  background:var(--margin-line-color);
}
#pw-body::before{left:0;z-index:5;}
#pw-body::after{left:5px;z-index:4;}
#pw-root.no-rule #pw-body{padding-left:0;}
#pw-root.no-rule #pw-body::before,
#pw-root.no-rule #pw-body::after{display:none;}

#pw-body p{font-size:17px;line-height:1.7;margin:0 0 8px 0;}
#pw-body p:last-child{margin-bottom:0;}
.pw-hd{font-size:19px;font-weight:700;color:var(--teal-answer);
  line-height:1.5;margin:0 0 8px 0;}
#pw-body ul{padding-left:26px;margin:6px 0 10px;}
#pw-body li{margin-bottom:6px;padding-left:4px;font-size:17px;line-height:1.6;}
#pw-body li::marker{color:var(--teal-answer);font-weight:700;}
#pw-body .diff-table{margin:10px 0;}

/* Answer box — the template's teal label colours as a callout. */
.pw-ans{
  background:var(--teal-label-bg);
  border-left:5px solid var(--teal-answer);
  border-radius:0 6px 6px 0;
  padding:12px 16px;margin:14px 0 4px;
}
.pw-ans-label{
  font-size:13px;font-weight:700;color:var(--teal-answer);
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px;
}
.pw-ans-body{font-size:18px;line-height:1.7;color:var(--text-dark);}
.pw-ans-body div{margin-bottom:2px;}

.katex{font-size:1.05em;}
.tex[data-display="1"],.katex-display{margin:.25em 0;}
.tex-error{color:#b00020;font-family:ui-monospace,monospace;font-size:.85em;}
"""


def _document(body_html: str, *, rel: str, width: int, background: str,
              banner_img: str | None, banner_w_pct: int, banner_align: str,
              margin_rule: bool, pad_x: int, pad_y: int) -> str:
    template_css = (ASSETS / "template.css").read_text(encoding="utf-8")
    banner = ""
    if banner_img:
        banner = (f'<div id="pw-banner" class="{_html.escape(banner_align)}">'
                  f'<img src="{rel}/{banner_img}" alt=""></div>')
    root_cls = "" if margin_rule else " class=\"no-rule\""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{rel}/katex/katex.min.css">
<style>
{_font_face(rel)}
{template_css}
:root{{
  --pw-w:{width}px;
  --pw-bg:{background};
  --pw-pad-x:{pad_x}px;
  --pw-pad-y:{pad_y}px;
  --pw-banner-w:{banner_w_pct}%;
}}
{_PATCH_CSS}
</style>
</head><body>
<div id="pw-root"{root_cls}>
{banner}
<div id="pw-body">{body_html}</div>
</div>
<script src="{rel}/katex/katex.min.js"></script>
<script src="{rel}/katex/contrib/mhchem.min.js"></script>
<script>
(function(){{
  document.querySelectorAll('.tex').forEach(function(el){{
    var tex = el.textContent;
    try {{
      katex.render(tex, el, {{
        displayMode: el.dataset.display === '1',
        throwOnError: false, strict: false, trust: true, output: 'html'
      }});
    }} catch (e) {{
      el.className = 'tex-error';
      el.textContent = tex;
    }}
  }});
  document.fonts.ready.then(function(){{ window.__pwReady = true; }});
}})();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(text: str, *, label: str = "en", width: int = DEFAULT_WIDTH,
           scale: int = DEFAULT_SCALE, background: str = "#FFFFFF",
           transparent: bool = False, banner_width_pct: int = 52,
           banner_align: str = "left", margin_rule: bool = True,
           pad_x: int = 26, pad_y: int = 22) -> dict:
    """
    Rasterise `text` into a patch PNG and register it.

    Returns {"id", "w", "h"} — w/h are the PNG's pixel size, which the UI uses
    to give the new block its natural aspect ratio. Raises RuntimeError when
    Chromium is unavailable, so the caller can say so plainly.
    """
    if sync_playwright is None:
        raise RuntimeError("Chromium is not installed — run "
                           "./.venv/bin/playwright install chromium")
    if not (text or "").strip():
        raise ValueError("type something first")

    variant = LABELS.get(label, LABELS["en"])
    body_html = to_html(text)
    width = max(200, min(2000, int(width)))
    scale = max(1, min(4, int(scale)))

    pid = uuid.uuid4().hex[:16]
    sess = WORK / f"_{pid}"
    sess.mkdir(parents=True, exist_ok=True)
    # The page lives one level under the assets dir so `..` reaches them, and
    # goto(file://) keeps every relative asset path (KaTeX fonts) resolvable.
    rel = ASSETS.as_uri()
    html = _document(body_html, rel=rel, width=width,
                     background="transparent" if transparent else background,
                     banner_img=variant["image"],
                     banner_w_pct=max(10, min(100, int(banner_width_pct))),
                     banner_align=banner_align, margin_rule=margin_rule,
                     pad_x=pad_x, pad_y=pad_y)
    page_file = sess / "patch.html"
    page_file.write_text(html, encoding="utf-8")

    out = WORK / f"{pid}.png"
    with _LOCK:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=_CHROME_ARGS)
            try:
                pg = browser.new_page(
                    viewport={"width": width + 40, "height": 600},
                    device_scale_factor=scale)
                pg.goto(page_file.as_uri(), wait_until="load")
                try:
                    pg.wait_for_function("window.__pwReady === true", timeout=5000)
                except Exception:
                    pass                       # fonts still usable if it lags
                png = pg.locator("#pw-root").screenshot(
                    omit_background=transparent, scale="device")
            finally:
                browser.close()
    out.write_bytes(png)

    for f in sess.iterdir():
        f.unlink()
    sess.rmdir()

    with Image.open(out) as im:
        w, h = im.size
    return {"id": pid, "w": w, "h": h}
