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
- **Merge pages** — pull another page's content onto the current page when both
  are half-empty: click *📥 Pull content from another page*, then click the
  source page's thumbnail in the left rail. The pulled content lands under the
  current page's content as a floating block you can drag anywhere and resize
  with its corner handle (each block has its own controls; pull as many as you
  like). The source page is deleted automatically (untick the checkbox to keep
  it, or restore it from its thumbnail). The block follows the source page's
  crop/tilt/colour live, and export + thumbnails composite it exactly as shown.
  A block parked past the crop margins still exports in full — the sheet grows
  outward (blank paper) to keep it, rather than clipping it at the margin. Use
  *⤵ Fit inside the margins* if you'd rather keep the sheet at its trimmed size.
- **Shape filler** — hide anything on a page under a filled free polygon:
  *🔷 Draw polygon*, click to place points, click the first (green) point or
  press Enter to close. Drag the white dots to move points; dragging a purple
  mid-edge dot inserts a new point there (subdivide as much as you like);
  double-click a dot removes it; drag the fill to move the whole shape.
  Fill colour comes from the colour picker, or *💧 Pick from page* grabs the
  exact pixel colour you click on the page itself — ideal for matching the
  paper tone so the patch is invisible. Shapes are burned into thumbnails
  and the export exactly as previewed.
- **Navigation** — the left rail lists every page (green dot = edited);
  `←` / `→` also step through pages.

## Borders (branded frames)

Integrated from the **arivihan-notes-composer** repo. The Borders tab (after
PYQs) wraps every exported page in the branded Arivihan frame:

- **Class** (10th / 12th) picks the asset set under `borders/<class>/`
  (only 12th has assets today — drop 10th's in later, no code changes).
- **Chapter no. + name** are stamped into the header pill
  (`CHAPTER - 05 | Name`, Urbanist, auto-shrinking/wrapping), rendered as real
  HTML/CSS with headless Chromium on export (Pillow fallback if missing).
- **Board + stream** attach that cover from `borders/<class>/front_pages/`
  (named `Board__Stream.jpg`) as page 1 of the export.
- Frame designs in `borders/<class>/templates/` cycle page by page.
- **Fit modes** (toggle chips — click an active one to go back to whole-page
  fit): **Fit width** (on by default; the composer's "my pages are not A4"
  option — auto-zooms so the width fills the window edge-to-edge, vertical
  slack padded in paper tone, very tall pages tuck a sliver under the frame)
  and **Fill · slight stretch** (covers the window completely by stretching
  at most ~15% toward its shape — barely visible on notes — and cropping the
  little that remains). Zoom/stretch sliders layer on top of either.
- The frame is **always its exact designed size** — the composer's original
  9:16 art, 2103 × 3738 px — never stretched or distorted. The cropped/rotated
  page is fitted whole into the frame's slot — centred, with paper-tone space
  filling any aspect difference, nothing trimmed. The preview renders the
  frame live around the margin guides (uniformly scaled) and re-adjusts as
  you drag them; the export matches what you see.

Geometry (template size, slot rect, header layout) lives in
`borders/<class>/config.json`.

## History (saved drafts)

Every **Export** automatically saves a draft into `drafts/<id>/` next to
`server.py`: a copy of the working PDFs (appended PYQs baked in) plus the
complete edit state — per-page crops, tilt, rotation, gap removals, colours,
the learned palette, watermark placement, border/chapter details, deleted
pages and the export DPI.

- **🕘 History** (top bar) lists all drafts — newest first, with a thumbnail,
  chapter/board name and date. **Open** rebuilds the session exactly as it
  was exported, ready for further changes; **🗑** deletes the draft.
- Re-exporting a reopened draft **updates the same draft** (no duplicates);
  a fresh upload starts a new one.
- **💾 Save draft** (inside History) snapshots the current session at any
  time without exporting.
- Drafts live on disk forever (they survive server restarts) but are
  git-ignored so PDF blobs don't bloat the repo — remove the `drafts/` line
  from `.gitignore` if you want to commit them.

## Files

- `server.py` — Flask app (upload, live preview, export).
- `imaging.py` — the deskew + crop engine (pure Pillow + numpy, no OpenCV).
- `borders.py` — branded border frames + chapter header (from notes-composer).
- `borders/` — per-class frame templates, cover pages and layout config.
- `static/index.html` — the entire UI.

Nothing is uploaded anywhere. PDFs live in a temp folder for the life of the
process and are deleted on exit.
# notes-editor
