"""
Local web server for the notes deskewer.

Run:  python server.py   (then open http://127.0.0.1:5000)

Nothing leaves your machine — PDFs are processed locally and cached in a
temp folder for the life of the process.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
import uuid
import zipfile
import tempfile
import atexit
import shutil

import fitz  # PyMuPDF
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response, abort

import imaging
import borders

app = Flask(__name__, static_folder="static", static_url_path="")

PREVIEW_DPI = 110       # detection + on-screen preview
DEFAULT_EXPORT_DPI = 300

# Board-wise previous-year-question PDFs, organised as
# PYQs/<BOARD>/<MEDIUM>/<SUBJECT>/<CHAPTER>/*.pdf
PYQS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PYQs")
_SAFE_SEG = re.compile(r"^[A-Za-z0-9_-]+$")

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Saved drafts: every export snapshots the working PDFs + the full edit state
# (crops, angles, colours, watermark, borders, palette …) into
# drafts/<id>/, so any past export can be reopened and re-edited later.
DRAFTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drafts")
_DRAFT_ID = re.compile(r"^[a-f0-9]{12}$")

# Python code is loaded once at start-up; if the files on disk are edited
# later, the running process silently keeps the old behaviour. The editor
# asks /api/borders/options on load, so we report staleness there and the
# UI shows a "restart the server" warning instead of exporting stale output.
_PROCESS_START = time.time()

WORK_DIR = tempfile.mkdtemp(prefix="notes_deskew_")
atexit.register(lambda: shutil.rmtree(WORK_DIR, ignore_errors=True))

# job_id -> {"files": [ {fid, name, path, pages: [ {angle, crop, w, h} ] } ] }
JOBS: dict = {}
# (job_id, fid, page) -> deskew-DPI base raster (PIL RGB), before any rotation
BASE_CACHE: dict = {}


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        abort(404, "unknown job")
    return job


def _find_file(job: dict, fid: str) -> dict:
    for f in job["files"]:
        if f["fid"] == fid:
            return f
    abort(404, "unknown file")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    uploads = request.files.getlist("files")
    if not uploads:
        return jsonify({"error": "no files"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    files = []

    for up in uploads:
        if not up.filename.lower().endswith(".pdf"):
            continue
        fid = uuid.uuid4().hex[:8]
        path = os.path.join(job_dir, f"{fid}.pdf")
        up.save(path)

        doc = fitz.open(path)
        pages = []
        for i in range(doc.page_count):
            base = imaging.render_page(doc, i, PREVIEW_DPI)
            angle = imaging.detect_skew(base)
            deskewed = imaging.deskew_image(base, angle)
            crop = imaging.content_bbox_fractions(deskewed)
            BASE_CACHE[(job_id, fid, i)] = base
            pages.append({
                "angle": angle,
                "crop": crop,
                "removals": [],
                "w": deskewed.width,
                "h": deskewed.height,
            })
        doc.close()

        files.append({
            "fid": fid,
            "name": up.filename,
            "path": path,
            "npages": len(pages),
            "pages": pages,
        })

    if not files:
        return jsonify({"error": "no valid PDFs"}), 400

    JOBS[job_id] = {"files": files, "dir": job_dir}

    # Strip server-only fields from the response.
    payload_files = [{
        "fid": f["fid"], "name": f["name"], "npages": f["npages"],
        "pages": f["pages"],
    } for f in files]
    return jsonify({"job": job_id, "files": payload_files})


def _color_from_args(args) -> dict:
    """Build a colour dict from flat query params (c* prefixed)."""
    return {
        "mode": args.get("cmode", "none"),
        "brightness": float(args.get("cbright", "1")),
        "contrast": float(args.get("ccontrast", "1")),
        "saturation": float(args.get("csat", "1")),
        "whiten": float(args.get("cwhiten", "0")),
        "deyellow": float(args.get("cdeyellow", "0")),
    }


def _crop_from_args(args):
    """Optional crop insets from query params (cr*); None if no crop."""
    c = {
        "left": float(args.get("crl", "0") or 0),
        "top": float(args.get("crt", "0") or 0),
        "right": float(args.get("crr", "0") or 0),
        "bottom": float(args.get("crb", "0") or 0),
    }
    return None if all(v <= 0 for v in c.values()) else c


def _overlays_from_args(args) -> list[dict]:
    """Parse merged-page overlays from query param (ov, JSON list)."""
    raw = args.get("ov", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except ValueError:
        return []


def _shapes_from_args(args) -> list[dict]:
    """Parse shape-filler polygons from query param (shp, JSON list)."""
    raw = args.get("shp", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except ValueError:
        return []


def _removals_from_args(args) -> list[dict]:
    """Parse removals from query param (rms)."""
    raw = args.get("rms", "")
    if not raw:
        return []
    import json
    try:
        if raw.startswith("["):
            return json.loads(raw)
        res = []
        for pair in raw.split("|"):
            parts = pair.split(",")
            if len(parts) == 2:
                res.append({"top": float(parts[0]), "bottom": float(parts[1])})
        return res
    except Exception:
        return []


@app.route("/api/preview")
def preview():
    """
    Deskewed (+ optional crop / removals / scan / colour) preview PNG for one page.
    Pass maxdim to get a small thumbnail (crop is applied when maxdim is set).
    """
    job_id = request.args.get("job", "")
    fid = request.args.get("fid", "")
    page = int(request.args.get("page", "0"))
    angle = float(request.args.get("angle", "0"))
    rot = float(request.args.get("rot", "0") or 0)
    maxdim = int(request.args.get("maxdim", "0"))
    color = _color_from_args(request.args)
    crop = _crop_from_args(request.args)
    removals = _removals_from_args(request.args)
    watermark = request.args.get("wm", "0") == "1"
    wm_opacity = float(request.args.get("wmop", "1") or 1)
    wm_scale = float(request.args.get("wmscale", "1") or 1)
    wm_rotate = float(request.args.get("wmrot", "0") or 0)
    wm_dx = float(request.args.get("wmdx", "0") or 0)
    wm_dy = float(request.args.get("wmdy", "0") or 0)
    # Where the watermark anchors: the margin box or the whole sheet. The page's
    # crop insets come in separately (w*) because the main preview shows the
    # *uncropped* page — it needs the margins for placement without cropping.
    wm_center_margins = request.args.get("wmcm", "0") == "1"
    wm_crop = {
        "left": float(request.args.get("wcl", "0") or 0),
        "top": float(request.args.get("wct", "0") or 0),
        "right": float(request.args.get("wcr", "0") or 0),
        "bottom": float(request.args.get("wcb", "0") or 0),
    }

    _job(job_id)
    base = BASE_CACHE.get((job_id, fid, page))
    if base is None:
        abort(404, "no cached page")

    # For thumbnails, downscale the source first so all pages render fast.
    src = base
    if 0 < maxdim < max(base.size):
        scale = maxdim / max(base.size)
        src = base.resize((max(1, int(base.width * scale)),
                           max(1, int(base.height * scale))), Image.BILINEAR)

    palette = _job(job_id).get("palette")
    out = imaging.deskew_image(imaging.rotate_page(src, rot), angle)
    # Removals before crop: their fractions are in uncropped-deskewed space.
    if removals:
        out = imaging.apply_removals(out, removals)
    if crop:
        out = imaging.apply_crop(out, crop)
    out = imaging.finish_color(out, color, palette)
    # Merged-page overlays sit under the watermark, exactly like on export.
    overlays = _overlays_from_args(request.args)
    if overlays:
        out = imaging.paste_overlays(
            out, overlays, crop, palette,
            lambda ov: BASE_CACHE.get(
                (job_id, str(ov.get("fid", "")), int(ov.get("page", 0) or 0))))
    # Shape-filler polygons cover content under the watermark, like on export.
    shapes = _shapes_from_args(request.args)
    if shapes:
        out = imaging.draw_shapes(out, shapes, crop)
    if watermark:
        sheet, centre = imaging.watermark_geometry(
            out.size, wm_crop, wm_center_margins, cropped=crop is not None)
        out = imaging.apply_watermark(out, wm_opacity, wm_scale, wm_rotate,
                                      wm_dx, wm_dy, sheet, centre,
                                      align_ink=wm_center_margins)
    return Response(imaging.to_png_bytes(out), mimetype="image/png")


@app.route("/api/detect_gaps", methods=["POST", "GET"])
def detect_gaps():
    """Auto-detect blank vertical gaps for a given page."""
    req_json = request.get_json(silent=True) or {}
    job_id = (request.args.get("job") or "") or req_json.get("job", "")
    fid = (request.args.get("fid") or "") or req_json.get("fid", "")
    _page = request.args.get("page")
    page = int(_page) if _page is not None else int(req_json.get("page", 0))
    _angle = request.args.get("angle")
    angle = float(_angle) if _angle is not None else float(req_json.get("angle", 0))
    _rot = request.args.get("rot")
    rot = float(_rot) if _rot is not None else float(req_json.get("rotate", 0) or 0)
    crop = _crop_from_args(request.args) if request.args.get("crl") else req_json.get("crop")

    _job(job_id)
    base = BASE_CACHE.get((job_id, fid, page))
    if base is None:
        abort(404, "no cached page")

    deskewed = imaging.deskew_image(imaging.rotate_page(base, rot), angle)
    if crop:
        deskewed = imaging.apply_crop(deskewed, crop)

    gaps = imaging.auto_detect_blank_gaps(deskewed)
    if crop:
        # Detection ran on the cropped page (better signal — skips the header
        # and margins), but gap fractions are stored in uncropped-deskewed
        # space, so map them back through the crop insets.
        ct = float(crop.get("top", 0) or 0)
        cb = float(crop.get("bottom", 0) or 0)
        kh = max(1e-6, 1.0 - ct - cb)
        gaps = [{"top": round(ct + g["top"] * kh, 4),
                 "bottom": round(ct + g["bottom"] * kh, 4)} for g in gaps]
    return jsonify({"gaps": gaps})


@app.route("/api/sample", methods=["POST"])
def sample():
    """Learn a colour palette from an uploaded sample page (image or PDF)."""
    job_id = request.form.get("job", "")
    job = _job(job_id)
    up = request.files.get("file")
    if not up:
        return jsonify({"error": "no file"}), 400

    data = up.read()
    name = (up.filename or "").lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        img = imaging.render_page(doc, 0, PREVIEW_DPI)
        doc.close()
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")

    palette = imaging.extract_palette(img)
    job["palette"] = palette
    return jsonify(palette)


def _export_name(orig_name: str, bordered: bool, number: str, name: str,
                 nfiles: int) -> str:
    """Chapter-based filename when borders carry chapter info, else *_fixed."""
    stem = os.path.splitext(orig_name)[0]
    if bordered and (number.strip() or name.strip()):
        parts = []
        if number.strip():
            parts.append(f"Chapter {number.strip()}")
        if name.strip():
            parts.append(name.strip())
        base = " - ".join(parts)
        if nfiles > 1:
            base = f"{base} - {stem}"
        base = re.sub(r'[\\/:*?"<>|]+', "", base)
        base = re.sub(r"\s+", " ", base).strip()
        if base:
            return base + ".pdf"
    return stem + "_fixed.pdf"


@app.route("/api/export", methods=["POST"])
def export():
    """
    Body: {
      job, dpi?, settings: { fid: { pages: [ {angle, rotate, crop, removals} ] } },
      borders?: { on, class, board, stream, number, name }
    }
    Returns a single PDF (one file) or a zip (multiple files).
    """
    data = request.get_json(force=True)
    job_id = data.get("job", "")
    dpi = int(data.get("dpi", DEFAULT_EXPORT_DPI))
    settings = data.get("settings", {})
    watermark = bool(data.get("watermark", False))
    # 'all' stamps every page; 'page' stamps only pages whose settings carry wm.
    wm_scope = str(data.get("wmScope", "all") or "all")
    wm_opacity = float(data.get("wmOpacity", 1) or 1)
    wm_scale = float(data.get("wmScale", 1) or 1)
    wm_rotate = float(data.get("wmRotate", 0) or 0)
    wm_dx = float(data.get("wmDx", 0) or 0)
    wm_dy = float(data.get("wmDy", 0) or 0)
    wm_center_margins = bool(data.get("wmCenterMargins", False))
    job = _job(job_id)
    palette = job.get("palette")

    # Borders (branded frames + chapter header + optional board/stream cover).
    bd = data.get("borders") or {}
    bd_on = bool(bd.get("on"))
    bcfg = bcover = brenderer = None
    btpls: list = []
    bnumber = str(bd.get("number", "") or "")
    bname = str(bd.get("name", "") or "")
    bzoom = borders._clampf(bd.get("zoom", 1))
    bstretch_w = borders._clampf(bd.get("stretchW", 1))
    bstretch_h = borders._clampf(bd.get("stretchH", 1))
    bfit_mode = str(bd.get("fitMode") or "") \
        or ("width" if bd.get("fitWidth") else "page")
    if bfit_mode not in ("page", "width", "fill"):
        bfit_mode = "page"
    if bd_on:
        bcls = str(bd.get("class", "") or "")
        try:
            bcfg = borders.load_config(bcls)
        except LookupError as e:
            return jsonify({"error": str(e)}), 400
        btpls = borders.list_templates(bcls)
        if not btpls:
            return jsonify({"error": f"no border templates for class {bcls} "
                            f"yet — add PNGs to borders/{bcls}/templates/"}), 400
        bboard = str(bd.get("board", "") or "")
        bstream = str(bd.get("stream", "") or "")
        if bboard:
            # Streamless classes (10th) store covers under the board alone;
            # a board picked without a stream where streams DO exist simply
            # exports without a cover, as before.
            bcover = borders.find_cover(bcls, bboard, bstream)
            if bcover is None and bstream:
                return jsonify({"error": f"no cover page for {bboard} / {bstream}"}), 400
        brenderer = borders.HeaderRenderer(bcfg, bnumber, bname)

    outputs = []  # (filename, bytes)

    # Merged-page overlays reference source pages by fid+page; they may live in
    # any loaded PDF, so keep one lazily-opened doc per fid for the whole export.
    fid_paths = {f["fid"]: f["path"] for f in job["files"]}
    src_docs: dict = {}

    def _render_overlay_src(ov):
        fid = str(ov.get("fid", "") or "")
        path = fid_paths.get(fid)
        if path is None:
            return None
        d = src_docs.get(fid)
        if d is None:
            d = src_docs[fid] = fitz.open(path)
        pno = int(ov.get("page", 0) or 0)
        if not 0 <= pno < d.page_count:
            return None
        return imaging.render_page(d, pno, dpi)

    # One headless browser serves the chapter header (no-op when off). The
    # frame never changes size, so a single overlay serves every page.
    with (brenderer if brenderer is not None else contextlib.nullcontext()):
        bheader = brenderer.render(bcfg["template_size"][0], 1.0, 1.0) \
            if bd_on else None
        for f in job["files"]:
            fset = settings.get(f["fid"], {})
            page_settings = fset.get("pages", [])
            doc = fitz.open(f["path"])
            images = []
            tpl_i = 0                       # frame designs cycle per file
            # Pages render in the order the editor sends them (reordering);
            # each entry's "src" is the page's index inside the working PDF.
            if page_settings:
                ordered = [(int(ps.get("src", idx)), ps)
                           for idx, ps in enumerate(page_settings)]
            else:
                ordered = [(i, {}) for i in range(doc.page_count)]
            for i, ps in ordered:
                if not 0 <= i < doc.page_count:
                    continue
                if ps.get("deleted"):
                    continue          # page was deleted in the editor — skip it
                angle = float(ps.get("angle", f["pages"][i]["angle"]))
                rot = float(ps.get("rotate", 0) or 0)
                crop = ps.get("crop", f["pages"][i]["crop"])
                removals = ps.get("removals") if "removals" in ps else f["pages"][i].get("removals")
                color = ps.get("color") or imaging.DEFAULT_COLOR
                hi = imaging.render_page(doc, i, dpi)
                processed = imaging.process_page(hi, angle, crop, color, palette,
                                                 removals, rot)
                if ps.get("overlays"):
                    processed = imaging.paste_overlays(
                        processed, ps["overlays"], crop, palette,
                        _render_overlay_src)
                if ps.get("shapes"):
                    processed = imaging.draw_shapes(processed, ps["shapes"], crop)
                if watermark and (wm_scope != "page" or ps.get("wm")):
                    sheet, centre = imaging.watermark_geometry(
                        processed.size, crop, wm_center_margins, cropped=True)
                    processed = imaging.apply_watermark(processed, wm_opacity,
                                                        wm_scale, wm_rotate,
                                                        wm_dx, wm_dy, sheet, centre,
                                                        align_ink=wm_center_margins)
                if bd_on:
                    tpl = borders.load_template(btpls[tpl_i % len(btpls)])
                    tpl_i += 1
                    # per-page fit (falls back to the job-level values)
                    pbf = ps.get("bfit") or {}
                    pmode = str(pbf.get("mode") or bfit_mode)
                    if pmode not in ("page", "width", "fill"):
                        pmode = "page"
                    processed = borders.compose_bordered(
                        processed, tpl, bcfg, bheader,
                        zoom=borders._clampf(pbf.get("zoom", bzoom)),
                        stretch_w=borders._clampf(pbf.get("stretchW", bstretch_w)),
                        stretch_h=borders._clampf(pbf.get("stretchH", bstretch_h)),
                        fit_mode=pmode)
                images.append(processed.convert("RGB"))
            doc.close()

            if not images:
                continue              # every page of this PDF was deleted

            if bd_on and bcover is not None:
                images.insert(0, borders.load_cover(
                    bcover, bcfg["template_size"]).convert("RGB"))

            out_name = _export_name(f["name"], bd_on, bnumber, bname,
                                    len(job["files"]))
            buf = io.BytesIO()
            images[0].save(buf, "PDF", resolution=float(dpi),
                           save_all=True, append_images=images[1:])
            outputs.append((out_name, buf.getvalue()))

    for d in src_docs.values():
        d.close()

    if not outputs:
        return jsonify({"error": "every page is deleted — nothing to export"}), 400

    if len(outputs) == 1:
        name, blob = outputs[0]
        return send_file(io.BytesIO(blob), mimetype="application/pdf",
                         as_attachment=True, download_name=name)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in outputs:
            zf.writestr(name, blob)
    zbuf.seek(0)
    return send_file(zbuf, mimetype="application/zip",
                     as_attachment=True, download_name="notes_fixed.zip")


# ---------------------------------------------------------------------------
# Drafts (persistent history of exported/edited sessions)
# ---------------------------------------------------------------------------

def _draft_dir(did: str, must_exist: bool = True) -> str:
    if not _DRAFT_ID.match(did or ""):
        abort(400, "bad draft id")
    path = os.path.join(DRAFTS_DIR, did)
    if must_exist and not os.path.isfile(os.path.join(path, "draft.json")):
        abort(404, "no such draft")
    return path


def _read_draft_meta(did: str) -> dict:
    with open(os.path.join(_draft_dir(did), "draft.json"), encoding="utf-8") as fh:
        return json.load(fh)


@app.route("/api/drafts")
def drafts_list():
    """All saved drafts, newest first (metadata only, no state payload)."""
    items = []
    if os.path.isdir(DRAFTS_DIR):
        for name in os.listdir(DRAFTS_DIR):
            jpath = os.path.join(DRAFTS_DIR, name, "draft.json")
            if not os.path.isfile(jpath):
                continue
            try:
                with open(jpath, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                continue
            items.append({
                "id": meta.get("id", name),
                "name": meta.get("name", "Untitled draft"),
                "created": meta.get("created", 0),
                "updated": meta.get("updated", 0),
                "files": [{"name": f.get("name"), "npages": f.get("npages")}
                          for f in meta.get("files", [])],
                "npages": meta.get("npages", 0),
                "exports": meta.get("exports", 0),
            })
    items.sort(key=lambda m: m.get("updated", 0), reverse=True)
    return jsonify({"drafts": items})


@app.route("/api/drafts/save", methods=["POST"])
def drafts_save():
    """
    Snapshot the current job into a draft: copies each working PDF (with any
    appended PYQs baked in) plus the client's full edit state.
    Body: { job, draft?, name?, state, exported? }.  Passing an existing
    draft id updates it in place, so re-exports don't pile up duplicates.
    """
    data = request.get_json(force=True)
    job = _job(data.get("job", ""))

    did = str(data.get("draft") or "")
    existing = None
    if did:
        if not _DRAFT_ID.match(did):
            abort(400, "bad draft id")
        # A stale id (draft deleted meanwhile) just re-creates it fresh.
        jpath = os.path.join(DRAFTS_DIR, did, "draft.json")
        if os.path.isfile(jpath):
            with open(jpath, encoding="utf-8") as fh:
                existing = json.load(fh)
    else:
        did = uuid.uuid4().hex[:12]

    ddir = os.path.join(DRAFTS_DIR, did)
    fdir = os.path.join(ddir, "files")
    os.makedirs(fdir, exist_ok=True)

    files_meta = []
    for i, f in enumerate(job["files"]):
        stored = f"{i:02d}.pdf"
        shutil.copyfile(f["path"], os.path.join(fdir, stored))
        files_meta.append({"stored": stored, "name": f["name"],
                           "npages": f["npages"]})
    # An update with fewer files than before leaves stale PDFs — drop them.
    keep = {fm["stored"] for fm in files_meta}
    for name in os.listdir(fdir):
        if name.endswith(".pdf") and name not in keep:
            os.remove(os.path.join(fdir, name))

    # First-page thumbnail for the history list (raw page — recognisable
    # enough, and it costs one low-DPI render).
    try:
        doc = fitz.open(job["files"][0]["path"])
        thumb = imaging.render_page(doc, 0, 40)
        doc.close()
        thumb.save(os.path.join(ddir, "thumb.png"))
    except Exception:
        pass

    now = time.time()
    meta = {
        "id": did,
        "name": str(data.get("name") or "").strip() or "Untitled draft",
        "created": existing["created"] if existing else now,
        "updated": now,
        "exports": (existing.get("exports", 0) if existing else 0)
                   + (1 if data.get("exported") else 0),
        "files": files_meta,
        "npages": sum(fm["npages"] for fm in files_meta),
        "state": data.get("state") or {},
    }
    tmp = os.path.join(ddir, "draft.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, os.path.join(ddir, "draft.json"))
    return jsonify({"draft": did, "name": meta["name"], "updated": now})


@app.route("/api/drafts/<did>/thumb")
def drafts_thumb(did: str):
    path = os.path.join(_draft_dir(did), "thumb.png")
    if not os.path.isfile(path):
        abort(404, "no thumbnail")
    return send_file(path, mimetype="image/png")


@app.route("/api/drafts/<did>/open", methods=["POST"])
def drafts_open(did: str):
    """
    Rebuild a live editing job from a saved draft: the stored PDFs go through
    the same render/deskew/margin pipeline as a fresh upload (so previews and
    the "Auto" reset baselines exist), and the saved edit state rides along
    for the client to lay on top.
    """
    meta = _read_draft_meta(did)
    ddir = _draft_dir(did)

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    files = []
    for fm in meta.get("files", []):
        src = os.path.join(ddir, "files", fm["stored"])
        if not os.path.isfile(src):
            continue
        fid = uuid.uuid4().hex[:8]
        path = os.path.join(job_dir, f"{fid}.pdf")
        # Work on a copy — later edits/PYQ appends must never touch the draft.
        shutil.copyfile(src, path)

        doc = fitz.open(path)
        pages = []
        for i in range(doc.page_count):
            base = imaging.render_page(doc, i, PREVIEW_DPI)
            angle = imaging.detect_skew(base)
            deskewed = imaging.deskew_image(base, angle)
            crop = imaging.content_bbox_fractions(deskewed)
            BASE_CACHE[(job_id, fid, i)] = base
            pages.append({
                "angle": angle,
                "crop": crop,
                "removals": [],
                "w": deskewed.width,
                "h": deskewed.height,
            })
        doc.close()
        files.append({
            "fid": fid,
            "name": fm["name"],
            "path": path,
            "npages": len(pages),
            "pages": pages,
        })

    if not files:
        return jsonify({"error": "draft has no PDFs on disk"}), 404

    JOBS[job_id] = {"files": files, "dir": job_dir}
    state = meta.get("state") or {}
    if state.get("palette"):
        JOBS[job_id]["palette"] = state["palette"]

    payload_files = [{
        "fid": f["fid"], "name": f["name"], "npages": f["npages"],
        "pages": f["pages"],
    } for f in files]
    return jsonify({"job": job_id, "files": payload_files,
                    "draft": did, "name": meta.get("name"), "state": state})


@app.route("/api/drafts/<did>", methods=["DELETE"])
def drafts_delete(did: str):
    shutil.rmtree(_draft_dir(did), ignore_errors=True)
    return jsonify({"ok": True})


def _pyqs_path(*parts) -> str:
    """Resolve a folder inside PYQS_DIR from validated path segments."""
    for p in parts:
        if not p or not _SAFE_SEG.match(p):
            abort(400, "bad PYQs path segment")
    path = os.path.join(PYQS_DIR, *parts)
    if not os.path.isdir(path):
        abort(404, "no such PYQs folder")
    return path


@app.route("/api/pyqs/options")
def pyqs_options():
    """
    List the next level of the PYQs tree. Pass none/some of board, medium,
    subject, chapter; returns subfolders (dirs) and PDFs at that level.
    """
    parts = []
    for key in ("board", "medium", "subject", "chapter"):
        val = request.args.get(key, "")
        if not val:
            break
        parts.append(val)
    path = _pyqs_path(*parts) if parts else PYQS_DIR
    if not os.path.isdir(path):
        abort(404, "PYQs folder missing — create it next to server.py")
    dirs, pdfs = [], []
    for name in os.listdir(path):
        if name.startswith("."):
            continue
        if os.path.isdir(os.path.join(path, name)):
            dirs.append(name)
        elif name.lower().endswith(".pdf"):
            pdfs.append(name)
    dirs.sort(key=lambda n: (0, int(n)) if n.isdigit() else (1, n))
    pdfs.sort()
    return jsonify({"dirs": dirs, "pdfs": pdfs})


@app.route("/api/pyqs/append", methods=["POST"])
def pyqs_append():
    """
    Append every PDF in PYQs/<board>/<medium>/<subject>/<chapter>/ to the end
    of one loaded file. The new pages go through the same auto-deskew and
    auto-margin pipeline as uploaded pages, so they are editable like any
    other page and export as part of the same PDF.
    """
    data = request.get_json(force=True)
    job_id = data.get("job", "")
    job = _job(job_id)
    f = _find_file(job, data.get("fid", ""))
    folder = _pyqs_path(str(data.get("board", "")), str(data.get("medium", "")),
                        str(data.get("subject", "")), str(data.get("chapter", "")))

    pdf_names = sorted(n for n in os.listdir(folder)
                       if n.lower().endswith(".pdf") and not n.startswith("."))
    if not pdf_names:
        return jsonify({"error": "no PYQ PDFs uploaded in that chapter folder yet"}), 404

    doc = fitz.open(f["path"])
    start = doc.page_count
    for name in pdf_names:
        src = fitz.open(os.path.join(folder, name))
        doc.insert_pdf(src)
        src.close()
    tmp = f["path"] + ".tmp"
    doc.save(tmp)
    doc.close()
    os.replace(tmp, f["path"])

    doc = fitz.open(f["path"])
    new_pages = []
    for i in range(start, doc.page_count):
        base = imaging.render_page(doc, i, PREVIEW_DPI)
        angle = imaging.detect_skew(base)
        deskewed = imaging.deskew_image(base, angle)
        crop = imaging.content_bbox_fractions(deskewed)
        BASE_CACHE[(job_id, f["fid"], i)] = base
        page = {
            "angle": angle,
            "crop": crop,
            "removals": [],
            "w": deskewed.width,
            "h": deskewed.height,
        }
        f["pages"].append(page)
        new_pages.append(page)
    doc.close()
    f["npages"] = len(f["pages"])

    return jsonify({"fid": f["fid"], "start": start, "npages": f["npages"],
                    "pages": new_pages, "sources": pdf_names})


# ---------------------------------------------------------------------------
# Borders (branded frames integrated from arivihan-notes-composer)
# ---------------------------------------------------------------------------

@app.route("/fonts/<name>")
def font_file(name: str):
    """Serve the Urbanist variable font to the editor's live border preview."""
    if not re.match(r"^[A-Za-z0-9._-]+$", name) or name.startswith("."):
        abort(400, "bad font name")
    path = os.path.join(FONTS_DIR, name)
    if not os.path.isfile(path):
        abort(404, "no such font")
    resp = send_file(path)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/api/borders/options")
def borders_options():
    """Classes, boards/streams and frame geometry for the Borders tab."""
    data = borders.options()
    code = [os.path.abspath(__file__), borders.__file__, imaging.__file__]
    try:
        newest = max(os.path.getmtime(f) for f in code if f and os.path.isfile(f))
        data["stale_server"] = newest > _PROCESS_START + 1
    except OSError:
        data["stale_server"] = False
    return jsonify(data)


@app.route("/api/borders/template")
def borders_template():
    """
    Downscaled PNG of one frame design, for the live in-editor overlay.
    Params: class (e.g. 12th), i (page's template index), w (pixel width).
    """
    cls = request.args.get("class", "")
    idx = int(request.args.get("i", "0") or 0)
    width = int(request.args.get("w", "1000") or 1000)
    try:
        data = borders.template_preview_png(cls, idx, width)
    except LookupError as e:
        abort(404, str(e))
    resp = Response(data, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


if __name__ == "__main__":
    import webbrowser
    import threading

    port = int(os.environ.get("PORT", "5001"))  # 5000 is taken by macOS AirPlay
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"\n  Notes Deskewer running at {url}\n  (Ctrl-C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
