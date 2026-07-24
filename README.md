# Notes Deskewer

Straighten and crop scanned handwritten notes — **fully local**, batch-friendly.

Drop in one or more PDFs; every page is automatically de-tilted and its margins
auto-detected. Fine-tune the angle and drag the margin guides per page (or copy
one page's settings to the whole batch), then export clean PDFs.

## Run

```bash
./run.sh
```

First run builds a virtualenv and installs dependencies, then opens
<http://127.0.0.1:5000> in your browser. Later runs start instantly.

To use a specific Python: `PYTHON=python3.12 ./run.sh`

## How it works

| Step | What happens |
|------|--------------|
| **Render** | Each PDF page is rasterised with PyMuPDF (no external Poppler needed). |
| **Deskew** | The tilt angle is found with the projection-profile method — the page is virtually rotated across a range of angles and the one where text rows line up most sharply wins. Coarse (1°) then fine (0.1°) search. |
| **Auto-crop** | The handwriting's bounding box is detected (with small padding) to suggest margins to trim. |
| **Preview** | You see the straightened page live; nudge the angle or drag the blue guides. |
| **Export** | Pages are re-rendered at your chosen DPI (150/200/300), rotated + cropped, and rebuilt into `*_fixed.pdf`. Multiple inputs come back as a `.zip`. |

## Controls

- **Tilt** — slider, ±1°/±0.1° buttons, or `Auto` to restore the detected angle.
- **Crop margins** — type exact percentages per side, or drag the guides.
  `Auto-detect` / `No crop` reset the page.
- **Apply to batch** — copy the current page's angle or margins to *every* page
  in *every* loaded PDF (great when all scans share the same margins).
- **Navigation** — the left rail lists every page (green dot = edited);
  `←` / `→` also step through pages.

## Files

- `server.py` — Flask app (upload, live preview, export).
- `imaging.py` — the deskew + crop engine (pure Pillow + numpy, no OpenCV).
- `static/index.html` — the entire UI.

Nothing is uploaded anywhere. PDFs live in a temp folder for the life of the
process and are deleted on exit.
# notes-editor
