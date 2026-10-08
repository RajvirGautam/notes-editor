#!/usr/bin/env python3
"""Swap individual notes pages inside an already-built bordered PDF — and drop the cover page.

Rebuilding a chapter through the border web tool would mean re-picking the per-batch fit/tone
settings by hand and re-exporting 50 MB of PDF to change three pages. This instead edits the
finished PDF in place: it paints the new notes photo over the slot rectangle of the page that
needs replacing and leaves the frame, header, footer and every other page byte-identical.

Two things have to match the original build or the patched page looks pasted-in:

GEOMETRY. The border frame for class 12 is 2103x3738 with the notes slot at (156, 573, 1791x2793).
Measured against the shipped PDFs, the notes photo is COVER-fitted into that slot (scaled to fill,
the small width overflow cropped) — not contain-fitted, and not stretched. See `SLOT` / `cover()`.

TONE. The build pipeline brightened the photo before placing it. Rather than guess the exact
enhancement, each replacement is CALIBRATED against the page it replaces: the old render is
compared with what is actually on the page and a per-channel gain+offset is solved by least
squares, then applied to the new render. A page whose calibration does not fit is refused rather
than patched, because a bad fit means the old render is not what is actually on that page — i.e.
the mapping is wrong and the wrong page would be overwritten.

Usage:
  python patch_bordered_pages.py --plan            # show what would change, touch nothing
  python patch_bordered_pages.py --apply           # back up, then patch + strip covers
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "Bordered Notes" / "Bio Notes FINAL"
BACKUP_DIR = ROOT / "Bordered Notes" / "Bio Notes FINAL (pre-edit backup)"
GEN = ROOT / "GPT-Notes"

TPL_W, TPL_H = 2103, 3738                       # borders/12th/config.json template_size
SLOT = dict(x=156, y=573, w=1791, h=2793)       # ...and its slot
CALIB_MAX_RESIDUAL = 14.0                       # mean abs grey error of the calibration fit

CH5 = "Chapter 05 - Molecular Basis of Inheritance.pdf"
CH9 = "Chapter 09 - Biotechnology Principles and Processes.pdf"

# page = 1-based page number in the CURRENT pdf (before the cover is stripped)
# new  = the render to put there;  old = the render currently there, used only to calibrate tone
REPLACE = [
    (CH5, 16, GEN / "Bio Ch5 MBI/generated/dia-07.jpg",
               GEN / "Bio Ch5 MBI/generated/prev/dia-07.jpg", "Meselson & Stahl, coloured"),
    (CH5, 26, GEN / "Bio Ch5 MBI/generated/page-14.jpg",
               GEN / "Bio Ch5 MBI/generated/page-14.orig.jpg", "codon table, stop/start highlighted"),
    (CH9, 12, GEN / "Bio Ch9 BPP/generated/dia-06.jpg",
               GEN / "Bio Ch9 BPP/generated/prev/dia-06.jpg", "PCR, coloured"),
    (CH9, 16, GEN / "Bio Ch9 BPP/generated/dia-02.jpg",
               GEN / "Bio Ch9 BPP/generated/prev/dia-02.jpg", "recombinant DNA schematic, coloured"),
]

# insert a brand-new page AFTER `after` — the frame is cloned from `clone`, an existing page of the
# same chapter, so the header/footer stay in the chapter's own style.
INSERT = [
    (CH9, 9, 12, GEN / "Bio Ch9 BPP/generated/dia-04.jpg", "pBR322 map (was never compiled in)"),
]

STRIP_FIRST_PAGE = True


# --------------------------------------------------------------------------- geometry / tone

def cover(im: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fill w x h and centre-crop the overflow — the build pipeline's fit."""
    k = max(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


def slot_rect(page: fitz.Page) -> fitz.Rect:
    r = page.rect
    fx, fy = r.width / TPL_W, r.height / TPL_H
    return fitz.Rect(SLOT["x"] * fx, SLOT["y"] * fy,
                     (SLOT["x"] + SLOT["w"]) * fx, (SLOT["y"] + SLOT["h"]) * fy)


def slot_pixels(page: fitz.Page, size=(384, 576)) -> Image.Image:
    pm = page.get_pixmap(dpi=110, clip=slot_rect(page))
    return Image.open(io.BytesIO(pm.tobytes("png"))).convert("RGB").resize(size, Image.LANCZOS)


def calibrate(old: Image.Image, page: fitz.Page):
    """Per-channel (gain, offset) taking the old render to what is on the page, + the residual."""
    ref = np.asarray(slot_pixels(page), float)
    src = np.asarray(cover(old.convert("RGB"), SLOT["w"], SLOT["h"]).resize(ref.shape[1::-1],
                                                                           Image.LANCZOS), float)
    gains, resid = [], []
    for c in range(3):
        b = src[:, :, c].ravel()
        g, o = np.linalg.lstsq(np.vstack([b, np.ones_like(b)]).T, ref[:, :, c].ravel(),
                               rcond=None)[0]
        gains.append((float(g), float(o)))
        resid.append(np.abs(ref[:, :, c].ravel() - (g * b + o)).mean())
    return gains, float(np.mean(resid))


def apply_tone(im: Image.Image, gains) -> Image.Image:
    a = np.asarray(im.convert("RGB"), float)
    for c, (g, o) in enumerate(gains):
        a[:, :, c] = a[:, :, c] * g + o
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def stamp(page: fitz.Page, art: Image.Image, gains) -> None:
    """Paint `art` (a 1024x1536 notes render) over the page's slot."""
    fitted = cover(apply_tone(art, gains), SLOT["w"], SLOT["h"])
    buf = io.BytesIO()
    fitted.save(buf, format="JPEG", quality=92, subsampling=0)
    page.insert_image(slot_rect(page), stream=buf.getvalue(), keep_proportion=False, overlay=True)


# --------------------------------------------------------------------------- driver

def run(apply: bool) -> int:
    problems = []
    for pdf, pageno, new, old, why in REPLACE:
        for p in (new, old):
            if not p.exists():
                problems.append(f"missing render: {p}")
    for pdf, _, _, new, why in INSERT:
        if not new.exists():
            problems.append(f"missing render: {new}")
    if problems:
        print("\n".join("ERROR: " + p for p in problems), file=sys.stderr)
        return 1

    by_pdf: dict[str, dict] = {}
    for pdf, pageno, new, old, why in REPLACE:
        by_pdf.setdefault(pdf, {"rep": [], "ins": []})["rep"].append((pageno, new, old, why))
    for pdf, after, clone, new, why in INSERT:
        by_pdf.setdefault(pdf, {"rep": [], "ins": []})["ins"].append((after, clone, new, why))

    if apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_name in sorted(p.name for p in PDF_DIR.glob("*.pdf")):
        path = PDF_DIR / pdf_name
        jobs = by_pdf.get(pdf_name, {"rep": [], "ins": []})
        # A PDF with a backup has already been through this tool. Bail out BEFORE calibrating:
        # its pages now hold the NEW renders at shifted numbers, so calibrating against the old
        # ones would fail and report a scary false "does not match that page". The real gate must
        # only ever fire on a genuinely wrong mapping.
        if (BACKUP_DIR / pdf_name).exists():
            print(f"\n{pdf_name}\n    - already patched (backup on file) — skipped")
            continue
        doc = fitz.open(path)
        before = doc.page_count
        notes = []

        for pageno, new, old, why in jobs["rep"]:
            page = doc[pageno - 1]
            if page.rotation:
                problems.append(f"{pdf_name} p{pageno}: page is rotated {page.rotation}deg")
                continue
            gains, resid = calibrate(Image.open(old), page)
            tag = f"p{pageno} <- {new.name} ({why}); calib residual {resid:.1f}"
            if resid > CALIB_MAX_RESIDUAL:
                problems.append(f"{pdf_name} {tag} — REFUSED, over {CALIB_MAX_RESIDUAL}; "
                                "the old render does not match that page")
                continue
            notes.append("replace " + tag)
            if apply:
                stamp(page, Image.open(new), gains)

        # inserts last, and from the bottom up, so earlier replace page numbers stay valid
        for after, clone, new, why in sorted(jobs["ins"], key=lambda j: -j[0]):
            gains = calib_from_siblings(doc, jobs)
            notes.append(f"insert after p{after} <- {new.name} ({why}), frame cloned from p{clone}")
            if apply:
                doc.fullcopy_page(clone - 1, after)      # clone lands at index `after`
                stamp(doc[after], Image.open(new), gains)

        if STRIP_FIRST_PAGE:
            notes.append("strip cover page 1")
            if apply:
                doc.delete_page(0)

        print(f"\n{pdf_name}  ({before} pp)")
        for n in notes:
            print(f"    - {n}")
        if apply and notes:
            shutil.copy2(path, BACKUP_DIR / pdf_name)
            tmp = path.with_suffix(".patched.tmp")
            doc.save(str(tmp), garbage=4, deflate=True)
            doc.close()
            tmp.replace(path)
            print(f"    => saved, now {before + len(jobs['ins']) - 1} pp")
            continue
        doc.close()

    if problems:
        print("\n" + "\n".join("ERROR: " + p for p in problems), file=sys.stderr)
        return 1
    if not apply:
        print("\n(--plan only; nothing written. re-run with --apply)")
    return 0


def calib_from_siblings(doc: fitz.Document, jobs) -> list:
    """Tone for an inserted page: the mean calibration of this chapter's replaced pages, or 1:1."""
    gains = []
    for pageno, new, old, _why in jobs["rep"]:
        g, resid = calibrate(Image.open(old), doc[pageno - 1])
        if resid <= CALIB_MAX_RESIDUAL:
            gains.append(g)
    if not gains:
        return [(1.0, 0.0)] * 3
    return [tuple(float(np.mean([g[c][i] for g in gains])) for i in range(2)) for c in range(3)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the PDFs (backs them up first)")
    ap.add_argument("--plan", action="store_true", help="show the plan only (default)")
    a = ap.parse_args()
    raise SystemExit(run(apply=a.apply))
