"""
Spelling fixes in the writer's own hand ("fix word").

The user boxes a misspelled word on the main preview and types the correct
spelling. This module cuts a tile around the word from the 300-DPI page —
at exactly the size the OpenAI image-edit API works in, so no resampling
happens anywhere — masks the word, and asks the model to rewrite it in the
same handwriting. The result is never pasted back whole (the API re-renders
the entire tile, which would leave a visible quality seam): only the word
rectangle, with a feathered edge, is cut out as an RGBA patch. imaging.py
composites that patch during preview and export, *before* colour grading,
so the new ink is palette-mapped together with the rest of the page.

Needs OPENAI_API_KEY in the environment or in a .env file next to this
module (OPENAI_IMAGE_MODEL and OPENAI_IMAGE_QUALITY are honoured too).
This is the one feature that sends anything off the machine: one tile of
one page per fix goes to OpenAI's API.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFilter

API_URL = "https://api.openai.com/v1/images/edits"
CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_SPELL_MODEL = "gpt-5-mini"   # vision chat model for the spell checker
# Tile sizes the API accepts; the crop is taken at exactly one of these so
# the word reaches the model at full native resolution.
SIZES = [(1536, 1024), (1024, 1536), (1024, 1024)]
FEATHER = 8          # px of soft edge on the composited patch
RECT_PAD = 14        # px of context kept around the word inside the mask
DEFAULT_MODEL = "gpt-image-1"
DEFAULT_QUALITY = "medium"

_ENV_LOADED = False


def load_env() -> None:
    """Read KEY=VALUE lines from .env (next to this file) into os.environ."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


def api_ready() -> bool:
    load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))


def _rewrite_prompt(text: str) -> str:
    return (
        "This is a scanned page of handwritten school notes. The handwriting "
        "inside the masked region must be completely replaced with new "
        "handwritten text. Write the new text in exactly the same handwriting "
        "as the rest of the page: same pen ink colour and pen pressure, same "
        "letter shapes, slant, x-height and letter size, same word spacing "
        "and the same line height as the surrounding lines. If horizontal "
        "ruled lines run through the region, the new writing must sit on "
        "those same ruled lines exactly like the original writing does. "
        "Start writing at the top-left of the region. If the new text needs "
        "fewer lines than the region has, leave the rest of the region as "
        "clean blank ruled paper. Every written line must be fully dark and "
        "crisp — especially the last one: finish the writing comfortably "
        "above the bottom edge of the region, never letting any stroke touch "
        "or fade into the region's edges. Preserve the paper texture and "
        "ruled lines. Change nothing outside the masked region. The new text "
        f"must read exactly, word for word:\n{text}"
    )


def _prompt(wrong: str, correct: str) -> str:
    what = (f"the misspelled handwritten word '{wrong}'" if wrong
            else "the misspelled handwritten word")
    return (
        "This is a scanned page of handwritten school notes. Inside the "
        f"masked region, {what} must be replaced with the correctly spelled "
        f"word '{correct}'. Write it in exactly the same handwriting style "
        "as every other word on the page: same pen ink colour and pen "
        "pressure, same letter shapes, same x-height and letter size, "
        "sitting on the same baseline as the neighbouring words (if a "
        "horizontal ruled line passes through them, it must pass through "
        "this word the same way). Preserve the paper texture and any ruled "
        "lines inside the region. Change nothing outside the masked region. "
        f"The word must read exactly: {correct}"
    )


def _multipart(fields: dict, files: list) -> tuple[bytes, str]:
    """Encode fields + (name, filename, png_bytes) files as multipart."""
    boundary = "----spellfix" + uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, filename, data in files:
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{name}\"; filename=\"{filename}\"\r\n"
                  f"Content-Type: image/png\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), boundary


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _call_api(tile: Image.Image, mask: Image.Image, prompt: str,
              size: tuple[int, int], n: int,
              quality: str | None = None) -> list[Image.Image]:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to .env "
                           "next to server.py")
    fields = {
        "model": os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL),
        "prompt": prompt,
        "size": f"{size[0]}x{size[1]}",
        "quality": quality or os.environ.get("OPENAI_IMAGE_QUALITY",
                                             DEFAULT_QUALITY),
        "n": str(n),
    }
    files = [("image[]", "tile.png", _png_bytes(tile)),
             ("mask", "mask.png", _png_bytes(mask))]
    body, boundary = _multipart(fields, files)
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read())
            imgs = []
            for item in data.get("data", []):
                raw = base64.b64decode(item.get("b64_json", ""))
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                if im.size != tile.size:
                    im = im.resize(tile.size, Image.LANCZOS)
                imgs.append(im)
            if not imgs:
                raise RuntimeError("the image API returned no images")
            return imgs
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                pass
            last_err = RuntimeError(
                f"image API error {e.code}: {detail or e.reason}")
            # Retry only what a retry can help (rate limit / server hiccup).
            if e.code not in (429, 500, 502, 503, 504) or attempt:
                raise last_err from None
            time.sleep(3)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = RuntimeError(f"could not reach the image API: {e}")
            if attempt:
                raise last_err from None
            time.sleep(2)
    raise last_err  # unreachable, keeps type-checkers calm


_SPELL_PROMPT = """You proofread scanned handwritten school notes (Indian \
state-board students, English or Hindi medium). Read every word on this page \
image and report genuine SPELLING mistakes only.

Rules:
- Report a word only when you are confident it is misspelled — not stylistic \
choices, abbreviations, proper nouns, chemical/math symbols, or words that \
are merely hard to read. When handwriting is ambiguous, skip it.
- These are handwritten letterforms: an 'a' often looks like 'u', 'r' like \
'v', 'n' like 'm', 'o' like 'a', and so on. Judge by the whole word and the \
sentence it sits in, NOT by individual letter shapes. If the writer clearly \
intended the correctly spelled word and only the penmanship is sloppy, do \
NOT report it. Only report words that would still be misspelled in the \
writer's neatest handwriting (a letter genuinely missing, doubled, swapped \
or wrong).
- Ignore grammar, punctuation and capitalisation.
- British and American spellings are both fine.
- For each mistake give: "wrong" (exactly as written), "correct" (the right \
spelling), "context" (3-6 surrounding words), and "x","y" — the approximate \
CENTRE of the wrong word as fractions of image width and height (0 to 1).

Answer with JSON only: {"mistakes":[{"wrong":"...","correct":"...",\
"context":"...","x":0.42,"y":0.31}, ...]}. No mistakes -> {"mistakes":[]}."""


def check_page(img: Image.Image) -> list[dict]:
    """
    Ask a vision chat model for the spelling mistakes on one page image.

    `img` is the page as the editor previews it (geometry-corrected,
    uncropped) so the returned x/y fractions line up with the on-screen
    page. Returns [{"wrong","correct","context","x","y"}]. Text only — no
    image generation happens here.
    """
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to .env "
                           "next to server.py")
    model = os.environ.get("OPENAI_SPELL_MODEL", DEFAULT_SPELL_MODEL)

    # Long side ≤ 1500 px, JPEG: plenty for reading handwriting, small upload.
    im = img.convert("RGB")
    if max(im.size) > 1500:
        s = 1500 / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    data_url = ("data:image/jpeg;base64,"
                + base64.b64encode(buf.getvalue()).decode())

    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _SPELL_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url,
                                                "detail": "high"}},
        ]}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4000,
    }
    if model.startswith(("gpt-5", "o")):     # reasoning models: keep it quick
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(CHAT_URL, method="POST",
                                 data=json.dumps(body).encode(), headers={
                                     "Authorization": f"Bearer {key}",
                                     "Content-Type": "application/json",
                                 })
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                pass
            last_err = RuntimeError(f"spell-check API error {e.code}: "
                                    f"{detail or e.reason}")
            if e.code not in (429, 500, 502, 503, 504) or attempt:
                raise last_err from None
            time.sleep(3)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = RuntimeError(f"could not reach the spell-check API: {e}")
            if attempt:
                raise last_err from None
            time.sleep(2)
    else:
        raise last_err

    try:
        text = data["choices"][0]["message"]["content"]
        mistakes = json.loads(text).get("mistakes", [])
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("the spell-check model returned no readable JSON")

    out = []
    for m in mistakes if isinstance(mistakes, list) else []:
        try:
            wrong = str(m.get("wrong", "")).strip()
            correct = str(m.get("correct", "")).strip()
            if not wrong or not correct or wrong.lower() == correct.lower():
                continue
            out.append({
                "wrong": wrong, "correct": correct,
                "context": str(m.get("context", "")).strip()[:120],
                "x": min(1.0, max(0.0, float(m.get("x", 0.5)))),
                "y": min(1.0, max(0.0, float(m.get("y", 0.5)))),
            })
        except (TypeError, ValueError):
            continue
    return out


def generate_candidates(geo: Image.Image, rect: list[float],
                        wrong: str, correct: str, n: int = 3) -> list[dict]:
    """
    Produce n correction candidates for the word boxed by `rect`.

    `geo` is the page in the editor's working space — rotated, deskewed,
    perspective-fixed, gaps removed, uncropped, raw colour — rendered at
    high DPI. `rect` is [x0, y0, x1, y1] as fractions of that image (the
    same space the main preview shows).

    Each candidate is {"id", "png" (RGBA patch bytes), "rect" (fractions
    the patch covers — slightly larger than the input, for the feather),
    "thumb" (context PNG bytes for the picker UI)}.
    """
    return _generate_patches(geo, rect,
                             _prompt(wrong.strip(), correct.strip()), n)


def generate_rewrite(geo: Image.Image, rect: list[float],
                     text: str, n: int = 2) -> list[dict]:
    """
    Rewrite the whole boxed region (a paragraph, a few lines, a heading)
    so it reads as `text`, in the writer's own hand.

    Same contract as generate_candidates — the result rides the same
    fixes/patch pipeline. A paragraph carries much more text than a word,
    so legibility matters more: quality defaults to "high" here
    (OPENAI_REWRITE_QUALITY overrides), and the context thumb shows a
    wider band of the page around the region.
    """
    quality = os.environ.get("OPENAI_REWRITE_QUALITY", "high")
    return _generate_patches(geo, rect, _rewrite_prompt(text.strip()), n,
                             quality=quality, ctx_pad=(160, 120))


def _generate_patches(geo: Image.Image, rect: list[float], prompt: str,
                      n: int, quality: str | None = None,
                      ctx_pad: tuple[int, int] = (120, 60)) -> list[dict]:
    W, H = geo.size
    x0 = max(0, min(W - 2, int(round(rect[0] * W))))
    y0 = max(0, min(H - 2, int(round(rect[1] * H))))
    x1 = max(x0 + 1, min(W, int(round(rect[2] * W))))
    y1 = max(y0 + 1, min(H, int(round(rect[3] * H))))

    # The whole word must fit inside a tile with margin; a giant box (or a
    # tiny page) scales the working copy down until it does.
    margin = 2 * (RECT_PAD + FEATHER) + 40
    scale = min(1.0,
                (SIZES[0][0] - margin) / (x1 - x0),
                (SIZES[0][1] - margin) / (y1 - y0))
    if scale < 1.0:
        geo = geo.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                         Image.LANCZOS)
        W, H = geo.size
        x0, y0 = int(x0 * scale), int(y0 * scale)
        x1, y1 = int(x1 * scale), int(y1 * scale)

    tw, th = SIZES[1] if (y1 - y0) > (x1 - x0) else SIZES[0]

    # Centre the tile on the word; clamp inside the page, pad with paper
    # white where the page itself is smaller than the tile.
    tx = max(0, min(max(0, W - tw), (x0 + x1) // 2 - tw // 2))
    ty = max(0, min(max(0, H - th), (y0 + y1) // 2 - th // 2))
    tile = Image.new("RGB", (tw, th), (250, 250, 248))
    tile.paste(geo.crop((tx, ty, min(W, tx + tw), min(H, ty + th))), (0, 0))

    # Mask: transparent where the model may edit (the word + a little pad).
    mx0 = max(0, x0 - tx - RECT_PAD)
    my0 = max(0, y0 - ty - RECT_PAD)
    mx1 = min(tw, x1 - tx + RECT_PAD)
    my1 = min(th, y1 - ty + RECT_PAD)
    mask = Image.new("RGBA", (tw, th), (0, 0, 0, 255))
    ImageDraw.Draw(mask).rectangle((mx0, my0, mx1, my1), fill=(0, 0, 0, 0))

    results = _call_api(tile, mask, prompt, (tw, th),
                        max(1, min(4, int(n))), quality)

    # Patch box: the mask zone plus feather headroom, clamped to the tile.
    px0, py0 = max(0, mx0 - FEATHER * 2), max(0, my0 - FEATHER * 2)
    px1, py1 = min(tw, mx1 + FEATHER * 2), min(th, my1 + FEATHER * 2)

    # Soft-edged alpha, shared by all candidates.
    am = Image.new("L", (tw, th), 0)
    am.paste(255, (mx0, my0, mx1, my1))
    am = am.filter(ImageFilter.GaussianBlur(FEATHER))
    alpha = am.crop((px0, py0, px1, py1))

    # The rect (in geo fractions) the stored patch covers.
    prect = [(tx + px0) / W, (ty + py0) / H, (tx + px1) / W, (ty + py1) / H]

    # Context crop shown in the picker: the region plus surroundings.
    cx0, cy0 = max(0, mx0 - ctx_pad[0]), max(0, my0 - ctx_pad[1])
    cx1, cy1 = min(tw, mx1 + ctx_pad[0]), min(th, my1 + ctx_pad[1])

    out = []
    for res in results:
        patch = res.crop((px0, py0, px1, py1)).convert("RGBA")
        patch.putalpha(alpha)
        # What the picker shows: the patch blended into the ORIGINAL tile —
        # exactly what preview/export will composite.
        blended = tile.copy()
        blended.paste(patch, (px0, py0), patch)
        thumb = blended.crop((cx0, cy0, cx1, cy1))
        if thumb.width > 560:
            thumb = thumb.resize((560, max(1, int(thumb.height * 560
                                                  / thumb.width))),
                                 Image.LANCZOS)
        out.append({
            "id": uuid.uuid4().hex[:16],
            "png": _png_bytes(patch),
            "rect": [round(v, 5) for v in prect],
            "thumb": _png_bytes(thumb),
        })
    return out
