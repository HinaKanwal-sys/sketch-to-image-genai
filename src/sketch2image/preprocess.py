"""Sketch preprocessing and control-map extraction.

A raw sketch (scan or phone photo) is turned into a clean binary line drawing,
and from that into the *control image* that a specific ControlNet expects:

    raw sketch ──clean_sketch()──> binary lines (black on white, square, /8 size)
                                 └──make_control(mode)──> control image

ControlNet conventions (lllyasviel ControlNet v1.1):
  * scribble / lineart / softedge: white lines on a black background
  * canny: white 1-px edges on black, extracted with the Canny operator

The OpenCV-based extractors below need no extra model downloads. If
`controlnet_aux` is installed, the learned HED / PiDiNet / Lineart annotators
can be used instead (modes ending in "_aux"), which is closer to what the
ControlNets saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

CONTROL_MODES = ("scribble", "lineart", "softedge", "canny", "softedge_aux", "lineart_aux")


@dataclass
class CleanConfig:
    size: int = 512                # output side length (square); must be divisible by 8
    background_blur: int = 41      # kernel for illumination flattening (odd)
    block_size: int = 31           # adaptive-threshold neighbourhood (odd)
    c: int = 12                    # adaptive-threshold offset; higher = fewer, darker lines kept
    min_component_px: int = 30     # remove specks smaller than this (at working resolution)
    stroke_px: int = 0             # 0 = keep stroke width, >0 = dilate/thin to roughly this width
    margin: float = 0.06           # white border added around the cropped drawing
    remove_ruled_lines: bool = False  # notebook paper: drop long straight lines (see _remove_ruled_lines)
    ruled_min_frac: float = 0.6    # a ruled line must span at least this fraction of the image


def load_image(path_or_img) -> Image.Image:
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")
    return Image.open(path_or_img).convert("RGB")


def clean_sketch(img, cfg: CleanConfig | None = None) -> Image.Image:
    """Return a clean binary sketch (black lines on white), square, cfg.size px.

    Steps
    1. grayscale
    2. flatten uneven lighting: divide by a heavily blurred copy (background estimate)
    3. adaptive threshold: robust to remaining shading and paper texture
    3b. optionally remove ruled notebook lines
    4. remove small connected components (dust, paper grain)
    5. crop to the drawing's bounding box, pad to a square with a margin, resize
    6. optionally normalise stroke width
    """
    cfg = cfg or CleanConfig()
    gray = cv2.cvtColor(np.asarray(load_image(img)), cv2.COLOR_RGB2GRAY)

    bg = cv2.medianBlur(gray, cfg.background_blur | 1)
    flat = cv2.divide(gray, bg, scale=255)

    binary = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg.block_size | 1, cfg.c
    )  # lines = 255

    if cfg.remove_ruled_lines:
        binary = _remove_ruled_lines(binary, cfg.ruled_min_frac)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component_px
    binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    binary = _crop_pad_square(binary, cfg.margin)
    binary = cv2.resize(binary, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
    binary = np.where(binary > 96, 255, 0).astype(np.uint8)

    if cfg.stroke_px > 0:
        binary = _normalise_stroke(binary, cfg.stroke_px)
    return Image.fromarray(255 - binary).convert("RGB")  # black lines on white


def _remove_ruled_lines(binary: np.ndarray, min_frac: float) -> np.ndarray:
    """Remove the printed lines of notebook paper.

    Ruled lines are long, straight and all parallel, while the drawing's strokes are shorter or curved.
    I find long segments with a probabilistic Hough transform, keep only those parallel to the dominant
    direction (so a long straight edge of the drawing in another direction survives), erase them, and
    close the small gaps this cuts into strokes that crossed a ruled line.
    """
    h, w = binary.shape
    min_len = int(min_frac * min(h, w))
    segs = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=min_len // 2, minLineLength=min_len, maxLineGap=25)
    if segs is None:
        return binary
    segs = segs[:, 0, :]
    angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    hist = np.histogram(angles, bins=36, range=(0, 180))[0]
    main = (np.argmax(hist) + 0.5) * 5
    diff = np.minimum(np.abs(angles - main), 180 - np.abs(angles - main))
    mask = np.zeros_like(binary)
    for (x0, y0, x1, y1) in segs[diff < 8]:
        # extend each segment a little (lines fade in shadows); extending further would widen the
        # erased band where nearly parallel segments diverge, cutting deep gaps into crossing strokes
        dx, dy = x1 - x0, y1 - y0
        t = 0.1 * max(h, w) / max(np.hypot(dx, dy), 1)
        p0 = (int(x0 - t * dx), int(y0 - t * dy))
        p1 = (int(x1 + t * dx), int(y1 + t * dy))
        cv2.line(mask, p0, p1, 255, thickness=7)
    out = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

    # pieces of ruled lines that bend with the paper survive the straight mask: drop thin
    # components that run parallel to the ruled lines
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        if len(xs) < 5:
            continue
        # principal axis of the pixel cloud: direction, length and thickness of the piece
        pts = np.column_stack([xs, ys]).astype(np.float64)
        evals, evecs = np.linalg.eigh(np.cov(pts, rowvar=False))
        length, thick = 4 * np.sqrt(max(evals[1], 0)), 4 * np.sqrt(max(evals[0], 0))
        ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180
        d = abs((ang - main + 90) % 180 - 90)
        if thick <= 8 and length >= 4 * max(thick, 1) and d < 10:
            out[labels == i] = 0

    # bridge the gaps with a closing whose kernel is a short line perpendicular to the ruled lines:
    # it reconnects strokes that crossed a ruled line without smearing along it
    L = 31
    kline = np.zeros((L, L), np.uint8)
    a = np.radians(main + 90)
    c = L // 2
    cv2.line(kline, (int(c - c * np.cos(a)), int(c - c * np.sin(a))),
             (int(c + c * np.cos(a)), int(c + c * np.sin(a))), 1, thickness=3)
    closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kline)
    near = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0  # only repair where a ruled line was erased
    added = ((closed > 0) & (out == 0) & near).astype(np.uint8)
    # a stroke crossing a ruled line leaves a narrow patch; wide patches come from hatching above and
    # below the line being joined, so they are dropped
    d = np.array([np.cos(np.radians(main)), np.sin(np.radians(main))])
    n, labels, _, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        proj = xs * d[0] + ys * d[1]
        if proj.max() - proj.min() > 14:
            added[labels == i] = 0
    return np.where(added > 0, 255, out).astype(np.uint8)


def _crop_pad_square(lines: np.ndarray, margin: float) -> np.ndarray:
    ys, xs = np.nonzero(lines)
    if len(xs) == 0:
        return lines
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = lines[y0:y1, x0:x1]
    h, w = crop.shape
    side = int(max(h, w) * (1 + 2 * margin))
    out = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy : oy + h, ox : ox + w] = crop
    return out


def _normalise_stroke(lines: np.ndarray, width: int) -> np.ndarray:
    from skimage.morphology import skeletonize

    skel = skeletonize(lines > 0).astype(np.uint8) * 255
    if width <= 1:
        return skel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(skel, k)


def make_control(clean: Image.Image, mode: str = "scribble") -> Image.Image:
    """Turn a clean sketch (black on white) into the control image for `mode`."""
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode {mode!r}; choose from {CONTROL_MODES}")
    g = cv2.cvtColor(np.asarray(clean.convert("RGB")), cv2.COLOR_RGB2GRAY)
    lines = 255 - g  # white lines on black

    if mode == "scribble":
        # the scribble ControlNet was trained on thick, synthesised scribbles
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.dilate(lines, k)
    elif mode == "lineart":
        out = lines
    elif mode == "softedge":
        # HED-like soft boundaries: blur and re-normalise
        out = cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 1.5)
        out = np.clip(out / (out.max() + 1e-6) * 255 * 1.4, 0, 255).astype(np.uint8)
    elif mode == "canny":
        out = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 100, 200)
    elif mode == "softedge_aux":
        from controlnet_aux import PidiNetDetector

        det = _aux_cache("pidi", lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"))
        res = det(clean, detect_resolution=clean.width, image_resolution=clean.width, safe=True)
        return res.resize(clean.size)
    elif mode == "lineart_aux":
        from controlnet_aux import LineartDetector

        det = _aux_cache("lineart", lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"))
        res = det(clean, detect_resolution=clean.width, image_resolution=clean.width)
        return res.resize(clean.size)
    return Image.fromarray(out).convert("RGB")


_AUX: dict = {}


def _aux_cache(key, factory):
    if key not in _AUX:
        _AUX[key] = factory()
    return _AUX[key]


def sketch_lines_mask(clean: Image.Image) -> np.ndarray:
    """Boolean mask of sketch line pixels from a clean (black-on-white) sketch."""
    g = np.asarray(clean.convert("L"))
    return g < 128
