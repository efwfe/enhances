"""
Custom Albumentations transforms for background manipulation.

- BackgroundReplace: replaces the non-bbox region with a random background image.
- BboxRelocate:      cuts out the bbox crop and pastes it at a random position on a background image.

Both transforms support YOLO / Pascal-VOC / COCO bbox formats via Albumentations' bbox pipeline.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_random_bg(bg_dir: Path, target_h: int, target_w: int) -> np.ndarray:
    """Pick a random image from *bg_dir* and resize it to (target_h, target_w)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in bg_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No image files found in background directory: {bg_dir}")
    path = random.choice(files)
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read background image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _load_bg_native(bg_dir: Path) -> np.ndarray:
    """Pick a random image from *bg_dir* and return it at its native resolution (RGB)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in bg_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No image files found in background directory: {bg_dir}")
    path = random.choice(files)
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read background image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _union_mask(bboxes_norm: list[tuple[float, float, float, float]],
                h: int, w: int) -> np.ndarray:
    """
    Build a uint8 mask (h, w) where pixels inside ANY bounding box == 255.
    bboxes_norm: list of (x_min, y_min, x_max, y_max) in [0,1] (albumentations internal format).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for x_min, y_min, x_max, y_max in bboxes_norm:
        x1 = int(round(x_min * w))
        y1 = int(round(y_min * h))
        x2 = int(round(x_max * w))
        y2 = int(round(y_max * h))
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _grabcut_mask(
    img: np.ndarray,
    bboxes_norm: list[tuple[float, float, float, float]],
    h: int,
    w: int,
    iter_count: int = 5,
) -> np.ndarray:
    """
    Build a foreground mask using GrabCut initialized with each bbox rect.
    For each bbox, GrabCut iterates to separate foreground from background.
    Falls back to rectangular mask for tiny boxes or on errors.

    Returns uint8 mask (h, w) where foreground pixels == 255.
    """
    # GrabCut requires 3-channel uint8
    if img.ndim != 3 or img.shape[2] != 3:
        return _union_mask(bboxes_norm, h, w)
    img_u8 = img if img.dtype == np.uint8 else img.astype(np.uint8)

    combined = np.zeros((h, w), dtype=np.uint8)
    for x_min, y_min, x_max, y_max in bboxes_norm:
        x1 = int(round(x_min * w))
        y1 = int(round(y_min * h))
        x2 = int(round(x_max * w))
        y2 = int(round(y_max * h))
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        # GrabCut needs a minimum region size to fit its GMM models
        if bw < 4 or bh < 4:
            combined[y1:y2, x1:x2] = 255
            continue
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                img_u8, gc_mask, (x1, y1, bw, bh),
                bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT,
            )
            fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            combined = cv2.bitwise_or(combined, fg)
        except Exception:
            # Fallback: rectangular mask
            combined[y1:y2, x1:x2] = 255

    return combined


# ---------------------------------------------------------------------------
# BackgroundReplace
# ---------------------------------------------------------------------------

class BackgroundReplace(DualTransform):
    """
    Replace the area **outside** all bounding boxes with a randomly chosen
    background image from *bg_dir*.

    When *use_grabcut* is True (default), GrabCut is run for each bbox to
    extract the foreground at pixel level before replacing the background,
    avoiding rectangular background remnants inside the bbox region.
    Falls back to a rectangular mask for very small boxes or on errors.

    The bounding boxes and their labels are kept unchanged.

    Args:
        bg_dir (str | Path): Directory containing background images.
        blend_border (int): Width (pixels) of a Gaussian blending region on
            the foreground mask edges to reduce hard seams.  0 = hard cut.
        use_grabcut (bool): If True, use GrabCut to extract per-pixel
            foreground masks.  If False, use rectangular bbox masks.
        grabcut_iters (int): Number of GrabCut iterations per bbox.
        p (float): Probability of applying this transform.
    """

    def __init__(
        self,
        bg_dir: str | Path,
        blend_border: int = 0,
        use_grabcut: bool = True,
        grabcut_iters: int = 5,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.bg_dir = Path(bg_dir)
        self.blend_border = blend_border
        self.use_grabcut = use_grabcut
        self.grabcut_iters = grabcut_iters

    # ------------------------------------------------------------------
    # Albumentations API
    # ------------------------------------------------------------------

    def apply(self, img: np.ndarray, bboxes: Sequence[tuple] = (), **params: Any) -> np.ndarray:
        # Load background at native resolution, then unify to the lower of the two.
        bg = _load_bg_native(self.bg_dir)
        bg_h, bg_w = bg.shape[:2]
        img_h, img_w = img.shape[:2]

        out_h = min(img_h, bg_h)
        out_w = min(img_w, bg_w)

        # Downscale whichever side is larger (INTER_AREA gives best quality when shrinking)
        if (img_h, img_w) != (out_h, out_w):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if (bg_h, bg_w) != (out_h, out_w):
            bg = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_AREA)

        h, w = out_h, out_w

        # Make sure bg has same channel count as img
        if img.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)

        norm_bboxes = list(bboxes)
        if self.use_grabcut:
            mask = _grabcut_mask(img, norm_bboxes, h, w, self.grabcut_iters)
        else:
            mask = _union_mask(norm_bboxes, h, w)

        if self.blend_border > 0:
            kernel_size = 2 * self.blend_border + 1
            mask_f = cv2.GaussianBlur(
                mask.astype(np.float32),
                (kernel_size, kernel_size),
                sigmaX=self.blend_border / 2,
            ) / 255.0
        else:
            mask_f = (mask / 255.0).astype(np.float32)

        if img.ndim == 3:
            mask_f = mask_f[:, :, np.newaxis]

        result = (img.astype(np.float32) * mask_f
                  + bg.astype(np.float32) * (1.0 - mask_f))
        return result.astype(img.dtype)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        # Segmentation masks follow the image crop/geometry — no change needed
        # since we only replace pixels outside bboxes.
        return mask

    def apply_to_bboxes(self, *args: Any, **params: Any) -> Any:
        # Bboxes are untouched (we only change the background).
        # Use *args to avoid name collision when albumentations passes bboxes
        # both as positional arg and as a keyword param from get_params_dependent_on_data.
        return args[0]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("bg_dir", "blend_border", "use_grabcut", "grabcut_iters")

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        # Pass current bboxes (already in albumentations normalised format) into apply()
        bboxes = data.get("bboxes", [])
        # Each element: BboxProcessor stores (x_min, y_min, x_max, y_max, *extra)
        norm_bboxes = [tuple(b[:4]) for b in bboxes]
        return {"bboxes": norm_bboxes}


# ---------------------------------------------------------------------------
# BboxRelocate
# ---------------------------------------------------------------------------

class BboxRelocate(DualTransform):
    """
    Cut every bounding-box crop from the image and paste it at a **random
    position** on a new background image.

    The bounding boxes are updated to reflect the new positions.

    Args:
        bg_dir (str | Path): Directory containing background images.
        allow_overlap (bool): If False (default), relocated boxes must not
            overlap each other.  If True, boxes are placed independently.
        max_attempts (int): Max random placement attempts per box when
            *allow_overlap* is False.  If exceeded the box is placed anyway.
        p (float): Probability of applying this transform.
    """

    def __init__(
        self,
        bg_dir: str | Path,
        allow_overlap: bool = False,
        max_attempts: int = 50,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.bg_dir = Path(bg_dir)
        self.allow_overlap = allow_overlap
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def _find_position(
        self,
        bw: int,
        bh: int,
        canvas_w: int,
        canvas_h: int,
        placed: list[tuple[int, int, int, int]],
    ) -> tuple[int, int]:
        """Return (x1, y1) for a box of size bw×bh inside a canvas_w×canvas_h."""
        max_x = canvas_w - bw
        max_y = canvas_h - bh
        if max_x < 0 or max_y < 0:
            return 0, 0  # box is larger than canvas — pin to top-left

        for _ in range(self.max_attempts):
            x1 = random.randint(0, max_x)
            y1 = random.randint(0, max_y)
            candidate = (x1, y1, x1 + bw, y1 + bh)
            if self.allow_overlap or all(
                self._iou(candidate, p) == 0.0 for p in placed
            ):
                return x1, y1

        # Fallback: random position ignoring overlap
        return random.randint(0, max_x), random.randint(0, max_y)

    # ------------------------------------------------------------------
    # Albumentations API
    # ------------------------------------------------------------------

    def apply(
        self,
        img: np.ndarray,
        bboxes: Sequence[tuple] = (),
        new_positions: list[tuple[int, int, int, int]] | None = None,
        **params: Any,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        bg = _load_random_bg(self.bg_dir, h, w)

        if img.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)

        if new_positions is None or not bboxes:
            return bg

        canvas = bg.copy()
        for bbox_norm, (nx1, ny1, nx2, ny2) in zip(bboxes, new_positions):
            x1 = int(round(bbox_norm[0] * w))
            y1 = int(round(bbox_norm[1] * h))
            x2 = int(round(bbox_norm[2] * w))
            y2 = int(round(bbox_norm[3] * h))
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            # Clip destination to canvas bounds
            dx1, dy1 = nx1, ny1
            dx2 = min(nx1 + cw, w)
            dy2 = min(ny1 + ch, h)
            canvas[dy1:dy2, dx1:dx2] = crop[: dy2 - dy1, : dx2 - dx1]

        return canvas

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_bboxes(
        self,
        bboxes: Any,
        new_positions: list[tuple[int, int, int, int]] | None = None,
        image_shape: tuple[int, int] | None = None,
        **params: Any,
    ) -> Any:
        is_array = isinstance(bboxes, np.ndarray)
        if new_positions is None or len(bboxes) == 0:
            return bboxes

        h, w = image_shape if image_shape else (params["rows"], params["cols"])
        updated = []
        for bbox, (nx1, ny1, nx2, ny2) in zip(bboxes, new_positions):
            extra = bbox[4:]
            updated.append((nx1 / w, ny1 / h, nx2 / w, ny2 / h, *extra))
        if is_array:
            return np.array(updated, dtype=np.float32)
        return updated

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("bg_dir", "allow_overlap", "max_attempts")

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        bboxes = data.get("bboxes", [])
        if len(bboxes) == 0:
            return {"new_positions": None}

        img = data["image"]
        h, w = img.shape[:2]
        norm_bboxes = [b[:4] for b in bboxes]

        placed: list[tuple[int, int, int, int]] = []
        new_positions: list[tuple[int, int, int, int]] = []
        for x_min, y_min, x_max, y_max in norm_bboxes:
            bw = max(1, int(round((x_max - x_min) * w)))
            bh = max(1, int(round((y_max - y_min) * h)))
            x1, y1 = self._find_position(bw, bh, w, h, placed)
            x2, y2 = x1 + bw, y1 + bh
            placed.append((x1, y1, x2, y2))
            new_positions.append((x1, y1, x2, y2))

        return {"new_positions": new_positions, "image_shape": (h, w)}
