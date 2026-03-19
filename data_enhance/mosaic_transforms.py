"""
Mosaic and CrossImageCopyPaste augmentation transforms for YOLO detection.

- Mosaic:              combines 4 images into a 2×2 grid; all bboxes are
                       remapped to the canvas coordinate space.
- CrossImageCopyPaste: pastes object crops from other dataset images onto the
                       current image at non-overlapping positions.

Both transforms read additional images and their YOLO label files from disk,
so they require access to the dataset directory at transform time.

Label file format expected by both transforms (one line per object):
    <class_id> <x_center> <y_center> <width> <height>   (all values normalised)

Note on label fields
--------------------
When extra images are loaded from disk the only label information available is
the integer class_id from the YOLO .txt file.  That integer is appended as the
extra field of each injected bbox.  If your pipeline uses string labels for the
*current* image, the output bbox list will contain a mix of string and integer
labels for the extra objects.  The simplest remedy is to keep labels as
integers throughout your pipeline.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yolo_labels(label_path: Path) -> list[tuple[float, float, float, float, int]]:
    """
    Parse a YOLO .txt label file.

    Returns a list of (x_min, y_min, x_max, y_max, class_id) tuples in
    normalised [0, 1] coordinates (albumentations internal format).
    """
    bboxes: list[tuple[float, float, float, float, int]] = []
    if not label_path.exists():
        return bboxes
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
            x_min = max(0.0, xc - w / 2)
            y_min = max(0.0, yc - h / 2)
            x_max = min(1.0, xc + w / 2)
            y_max = min(1.0, yc + h / 2)
            if x_max > x_min and y_max > y_min:
                bboxes.append((x_min, y_min, x_max, y_max, cls_id))
    return bboxes


def _load_random_sample(
    image_dir: Path,
    label_dir: Path,
    exclude: Path | None = None,
) -> tuple[np.ndarray, list[tuple]]:
    """
    Pick a random image from *image_dir*, load it as RGB, and load the
    corresponding YOLO label file from *label_dir*.

    Returns (image_rgb, bboxes) where bboxes are in albumentations format.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts and p != exclude
    ]
    if not candidates:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    img_path = random.choice(candidates)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise IOError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    label_path = label_dir / (img_path.stem + ".txt")
    bboxes = _load_yolo_labels(label_path)
    return img, bboxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------

class Mosaic(DualTransform):
    """
    Mosaic augmentation: tile 4 images into a 2×2 grid on a single canvas.

    The current image occupies one randomly chosen quadrant; three additional
    images are sampled from *image_dir* / *label_dir*.  All bounding boxes
    (current + extra) are remapped to their new canvas positions.

    Args:
        image_dir (str | Path): Directory containing training images.
        label_dir (str | Path): Directory containing YOLO .txt label files
            with the same stem as the corresponding image.
        img_size (int): Side length (px) of the square output canvas.
        center_range (tuple[float, float]): Fractional range within which the
            mosaic split-point is sampled.  (0.25, 0.75) keeps each quadrant
            between 25 % and 75 % of the canvas.
        p (float): Probability of applying the transform.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        img_size: int = 640,
        center_range: tuple[float, float] = (0.25, 0.75),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.center_range = center_range

    # ------------------------------------------------------------------
    # Albumentations API
    # ------------------------------------------------------------------

    def apply(
        self,
        img: np.ndarray,
        mosaic_data: dict | None = None,
        **params: Any,
    ) -> np.ndarray:
        s = self.img_size
        if mosaic_data is None:
            return cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)

        cx, cy = mosaic_data["cx"], mosaic_data["cy"]
        quad_order = mosaic_data["quad_order"]          # [0..3] permutation
        extra_images = mosaic_data["extra_images"]      # list of 3 RGB arrays

        # Quadrant regions (x1, y1, x2, y2) in canvas pixels
        quads = [
            (0,  0,  cx, cy),   # top-left
            (cx, 0,  s,  cy),   # top-right
            (0,  cy, cx, s ),   # bottom-left
            (cx, cy, s,  s ),   # bottom-right
        ]

        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        for i, src in enumerate([img] + list(extra_images)):
            x1, y1, x2, y2 = quads[quad_order[i]]
            w_q, h_q = x2 - x1, y2 - y1
            if w_q <= 0 or h_q <= 0:
                continue
            tile = cv2.resize(src, (w_q, h_q), interpolation=cv2.INTER_LINEAR)
            if tile.ndim == 2:                          # grayscale → 3-ch
                tile = np.stack([tile] * 3, axis=-1)
            canvas[y1:y2, x1:x2] = tile

        return canvas

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        # Segmentation masks are not composited — detection-only transform.
        return mask

    def apply_to_bboxes(
        self,
        bboxes: Sequence[Any],
        mosaic_data: dict | None = None,
        **params: Any,
    ) -> list:
        if mosaic_data is None:
            return list(bboxes)

        s = self.img_size
        cx, cy = mosaic_data["cx"], mosaic_data["cy"]
        quad_order = mosaic_data["quad_order"]
        extra_bboxes_list = mosaic_data["extra_bboxes"]   # list of 3 bbox lists

        quads = [
            (0,  0,  cx, cy),
            (cx, 0,  s,  cy),
            (0,  cy, cx, s ),
            (cx, cy, s,  s ),
        ]

        def _remap(bbox_list: Sequence[Any], qidx: int) -> list:
            x1q, y1q, x2q, y2q = quads[qidx]
            w_q, h_q = x2q - x1q, y2q - y1q
            out = []
            for bbox in bbox_list:
                xmin, ymin, xmax, ymax = bbox[:4]
                extra = bbox[4:]
                nx_min = np.clip((xmin * w_q + x1q) / s, 0.0, 1.0)
                ny_min = np.clip((ymin * h_q + y1q) / s, 0.0, 1.0)
                nx_max = np.clip((xmax * w_q + x1q) / s, 0.0, 1.0)
                ny_max = np.clip((ymax * h_q + y1q) / s, 0.0, 1.0)
                if nx_max > nx_min and ny_max > ny_min:
                    out.append((nx_min, ny_min, nx_max, ny_max, *extra))
            return out

        result = _remap(bboxes, quad_order[0])
        for i, extra_bbs in enumerate(extra_bboxes_list):
            result.extend(_remap(extra_bbs, quad_order[i + 1]))
        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("image_dir", "label_dir", "img_size", "center_range")

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        s = self.img_size
        lo = int(self.center_range[0] * s)
        hi = int(self.center_range[1] * s)
        cx = random.randint(lo, hi)
        cy = random.randint(lo, hi)

        extra_images, extra_bboxes = [], []
        for _ in range(3):
            img_e, bbs_e = _load_random_sample(self.image_dir, self.label_dir)
            extra_images.append(img_e)
            extra_bboxes.append(bbs_e)

        quad_order = list(range(4))
        random.shuffle(quad_order)

        return {
            "mosaic_data": {
                "cx": cx,
                "cy": cy,
                "extra_images": extra_images,
                "extra_bboxes": extra_bboxes,
                "quad_order": quad_order,
            }
        }


# ---------------------------------------------------------------------------
# CrossImageCopyPaste
# ---------------------------------------------------------------------------

class CrossImageCopyPaste(DualTransform):
    """
    Copy-Paste augmentation: paste object crops from other dataset images onto
    the current image and append the corresponding bounding boxes.

    Unlike :class:`BboxRelocate` (which moves crops *within* the same image),
    this transform samples crops from entirely different images, improving
    diversity and small-object recall.

    Args:
        image_dir (str | Path): Directory containing training images.
        label_dir (str | Path): Directory containing YOLO .txt label files.
        max_paste (int): Maximum number of crops to paste per call.
        scale_range (tuple[float, float]): (min, max) scale factors applied to
            each crop before pasting.  (1.0, 1.0) preserves the original size.
        allow_overlap (bool): If False (default), pasted boxes must not overlap
            each other or the existing boxes in the current image.
        max_attempts (int): Placement attempts per crop before giving up.
        p (float): Probability of applying the transform.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        max_paste: int = 3,
        scale_range: tuple[float, float] = (0.5, 1.5),
        allow_overlap: bool = False,
        max_attempts: int = 50,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.max_paste = max_paste
        self.scale_range = scale_range
        self.allow_overlap = allow_overlap
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_position(
        self,
        crop_w: int,
        crop_h: int,
        canvas_w: int,
        canvas_h: int,
        placed: list[tuple[int, int, int, int]],
    ) -> tuple[int, int] | None:
        max_x = canvas_w - crop_w
        max_y = canvas_h - crop_h
        if max_x < 0 or max_y < 0:
            return None                                 # crop too large
        for _ in range(self.max_attempts):
            x1 = random.randint(0, max_x)
            y1 = random.randint(0, max_y)
            candidate = (x1, y1, x1 + crop_w, y1 + crop_h)
            if self.allow_overlap or all(
                _iou(candidate, p) == 0.0 for p in placed
            ):
                return x1, y1
        return None                                     # could not place

    # ------------------------------------------------------------------
    # Albumentations API
    # ------------------------------------------------------------------

    def apply(
        self,
        img: np.ndarray,
        paste_items: list[dict] | None = None,
        **params: Any,
    ) -> np.ndarray:
        if not paste_items:
            return img
        canvas = img.copy()
        h, w = canvas.shape[:2]
        for item in paste_items:
            crop = item["crop"]
            nx1, ny1, nx2, ny2 = item["dst_px"]
            ch, cw = crop.shape[:2]
            dx2 = min(nx1 + cw, w)
            dy2 = min(ny1 + ch, h)
            canvas[ny1:dy2, nx1:dx2] = crop[: dy2 - ny1, : dx2 - nx1]
        return canvas

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_bboxes(
        self,
        bboxes: Sequence[Any],
        paste_items: list[dict] | None = None,
        image_shape: tuple[int, int] | None = None,
        **params: Any,
    ) -> list:
        result = list(bboxes)
        if not paste_items:
            return result
        h, w = image_shape if image_shape else (params["rows"], params["cols"])
        for item in paste_items:
            nx1, ny1, nx2, ny2 = item["dst_px"]
            cls_id = item["cls_id"]
            result.append((nx1 / w, ny1 / h, nx2 / w, ny2 / h, cls_id))
        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "image_dir", "label_dir", "max_paste",
            "scale_range", "allow_overlap", "max_attempts",
        )

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        img = data["image"]
        h, w = img.shape[:2]

        # Existing bboxes already placed on the canvas (pixel coords).
        existing_bboxes_norm = data.get("bboxes", [])
        placed: list[tuple[int, int, int, int]] = [
            (
                int(b[0] * w), int(b[1] * h),
                int(b[2] * w), int(b[3] * h),
            )
            for b in existing_bboxes_norm
        ]

        # Sample a source image from the dataset.
        src_img, src_bboxes = _load_random_sample(self.image_dir, self.label_dir)
        src_h, src_w = src_img.shape[:2]

        if not src_bboxes:
            return {"paste_items": [], "image_shape": (h, w)}

        random.shuffle(src_bboxes)
        n_paste = random.randint(1, self.max_paste)
        paste_items: list[dict] = []

        for bbox in src_bboxes[:n_paste]:
            xmin, ymin, xmax, ymax, cls_id = bbox

            # Extract crop from source image.
            sx1 = int(xmin * src_w)
            sy1 = int(ymin * src_h)
            sx2 = int(xmax * src_w)
            sy2 = int(ymax * src_h)
            sx1, sx2 = max(0, sx1), min(src_w, sx2)
            sy1, sy2 = max(0, sy1), min(src_h, sy2)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            crop = src_img[sy1:sy2, sx1:sx2].copy()

            # Optionally rescale the crop.
            scale = random.uniform(*self.scale_range)
            if scale != 1.0:
                new_cw = max(1, int(crop.shape[1] * scale))
                new_ch = max(1, int(crop.shape[0] * scale))
                crop = cv2.resize(crop, (new_cw, new_ch), interpolation=cv2.INTER_LINEAR)

            crop_h, crop_w = crop.shape[:2]

            # Find a valid position on the target canvas.
            pos = self._find_position(crop_w, crop_h, w, h, placed)
            if pos is None:
                continue                                 # cannot fit — skip

            nx1, ny1 = pos
            nx2, ny2 = nx1 + crop_w, ny1 + crop_h
            placed.append((nx1, ny1, nx2, ny2))
            paste_items.append({
                "crop": crop,
                "dst_px": (nx1, ny1, nx2, ny2),
                "cls_id": cls_id,
            })

        return {"paste_items": paste_items, "image_shape": (h, w)}
