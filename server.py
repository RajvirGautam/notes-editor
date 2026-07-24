"""
Local web server for the notes deskewer.

Run:  python server.py   (then open http://127.0.0.1:5000)

Nothing leaves your machine — PDFs are processed locally and cached in a
temp folder for the life of the process.
"""

from __future__ import annotations

import io
import os
import uuid
import zipfile
import tempfile
import atexit
import shutil

import fitz  # PyMuPDF
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response, abort

import imaging

app = Flask(__name__, static_folder="static", static_url_path="")

PREVIEW_DPI = 110       # detection + on-screen preview
DEFAULT_EXPORT_DPI = 200

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


@app.route("/api/preview")
def preview():
    """
    Deskewed (+ optional crop / scan / colour) preview PNG for one page.
    Pass maxdim to get a small thumbnail (crop is applied when maxdim is set).
    """
    job_id = request.args.get("job", "")
    fid = request.args.get("fid", "")
    page = int(request.args.get("page", "0"))
    angle = float(request.args.get("angle", "0"))
    maxdim = int(request.args.get("maxdim", "0"))
    color = _color_from_args(request.args)
    crop = _crop_from_args(request.args)
    watermark = request.args.get("wm", "0") == "1"
    wm_opacity = float(request.args.get("wmop", "1") or 1)

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
    out = imaging.deskew_image(src, angle)
    if crop:
        out = imaging.apply_crop(out, crop)
    out = imaging.finish_color(out, color, palette)
    if watermark:
        out = imaging.apply_watermark(out, wm_opacity)
    return Response(imaging.to_png_bytes(out), mimetype="image/png")


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


@app.route("/api/export", methods=["POST"])
def export():
    """
    Body: {
      job, dpi?, settings: { fid: { pages: [ {angle, crop} ] } }
    }
    Returns a single PDF (one file) or a zip (multiple files).
    """
    data = request.get_json(force=True)
    job_id = data.get("job", "")
    dpi = int(data.get("dpi", DEFAULT_EXPORT_DPI))
    settings = data.get("settings", {})
    watermark = bool(data.get("watermark", False))
    wm_opacity = float(data.get("wmOpacity", 1) or 1)
    job = _job(job_id)
    palette = job.get("palette")

    outputs = []  # (filename, bytes)

    for f in job["files"]:
        fset = settings.get(f["fid"], {})
        page_settings = fset.get("pages", [])
        doc = fitz.open(f["path"])
        images = []
        for i in range(doc.page_count):
            ps = page_settings[i] if i < len(page_settings) else {}
            angle = float(ps.get("angle", f["pages"][i]["angle"]))
            crop = ps.get("crop", f["pages"][i]["crop"])
            color = ps.get("color") or imaging.DEFAULT_COLOR
            hi = imaging.render_page(doc, i, dpi)
            processed = imaging.process_page(hi, angle, crop, color, palette)
            if watermark:
                processed = imaging.apply_watermark(processed, wm_opacity)
            images.append(processed.convert("RGB"))
        doc.close()

        out_name = os.path.splitext(f["name"])[0] + "_fixed.pdf"
        buf = io.BytesIO()
        if images:
            images[0].save(buf, "PDF", resolution=float(dpi),
                           save_all=True, append_images=images[1:])
        outputs.append((out_name, buf.getvalue()))

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


if __name__ == "__main__":
    import webbrowser
    import threading

    port = int(os.environ.get("PORT", "5001"))  # 5000 is taken by macOS AirPlay
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"\n  Notes Deskewer running at {url}\n  (Ctrl-C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
