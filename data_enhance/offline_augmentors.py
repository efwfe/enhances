"""
Offline generative augmentation using ComfyUI.

Two augmentors are provided for expanding a YOLO-format dataset:

MultiViewAugmentor
    Crops a bbox target, sends it to a multi-view generative model via ComfyUI,
    and composites each returned view back into a scene image.  Each view
    becomes a new labelled sample in the output dataset.

BboxScaleAugmentor
    Rescales bbox targets to different sizes.  Strategy:
      1. Extract the object crop (optionally refined with GrabCut).
      2. Resize to the target scale.
      3. Paste at a new (non-overlapping) position.
      4. Use ComfyUI inpainting to fill the hole left at the original position.
    This preserves the original object appearance and only uses generative AI
    for background reconstruction — giving more controllable results than
    in-painting the object itself.

Both classes operate on YOLO datasets stored as::

    image_dir/   *.jpg / *.png
    label_dir/   *.txt  (one per image, YOLO format)

and write augmented samples to a separate output directory pair.

Workflow templates
------------------
ComfyUI workflows are JSON files exported from the ComfyUI UI (Save → API
format).  Dynamic values (input filenames, angles, scale prompts …) are
marked with ``__PLACEHOLDER__`` strings that this module substitutes before
sending.  The exact placeholders expected by each augmentor are documented on
the respective class.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .comfyui_client import ComfyUIClient


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

BBox = tuple[float, float, float, float, int]  # x_min, y_min, x_max, y_max, cls_id


def _load_yolo(label_path: Path) -> list[BBox]:
    """Parse a YOLO .txt file → list of (x_min, y_min, x_max, y_max, cls_id)."""
    boxes: list[BBox] = []
    if not label_path.exists():
        return boxes
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x_min = max(0.0, xc - w / 2)
            y_min = max(0.0, yc - h / 2)
            x_max = min(1.0, xc + w / 2)
            y_max = min(1.0, yc + h / 2)
            if x_max > x_min and y_max > y_min:
                boxes.append((x_min, y_min, x_max, y_max, cls_id))
    return boxes


def _save_yolo(label_path: Path, boxes: list[BBox]) -> None:
    """Write a YOLO .txt file from a list of (x_min, y_min, x_max, y_max, cls_id)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as fh:
        for x_min, y_min, x_max, y_max, cls_id in boxes:
            xc = (x_min + x_max) / 2
            yc = (y_min + y_max) / 2
            w  = x_max - x_min
            h  = y_max - y_min
            fh.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _save_image(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _load_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Workflow template helpers
# ---------------------------------------------------------------------------

def load_workflow(path: str | Path) -> dict:
    """Load a ComfyUI workflow JSON template from disk."""
    with open(path) as fh:
        return json.load(fh)


def patch_workflow(workflow: dict, **substitutions: Any) -> dict:
    """
    Replace ``__KEY__`` placeholders in the workflow JSON with values.

    The replacement is done on the serialised JSON string so it works
    regardless of nesting depth.

    Example::

        wf = patch_workflow(template, INPUT_IMAGE="crop_001.png", AZIMUTH=45.0)
    """
    text = json.dumps(workflow)
    for key, value in substitutions.items():
        placeholder = f"__{key}__"
        text = text.replace(f'"{placeholder}"', json.dumps(value))
        text = text.replace(placeholder, str(value))
    return json.loads(text)


# ---------------------------------------------------------------------------
# Crop / mask helpers
# ---------------------------------------------------------------------------

def _bbox_to_pixels(
    box: BBox, img_h: int, img_w: int
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max, _ = box
    return (
        max(0, int(round(x_min * img_w))),
        max(0, int(round(y_min * img_h))),
        min(img_w, int(round(x_max * img_w))),
        min(img_h, int(round(y_max * img_h))),
    )


def _grabcut_refine(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    iters: int = 5,
) -> np.ndarray:
    """
    Run GrabCut on a bbox region and return a uint8 foreground mask (0/255).
    Falls back to a rectangular mask on errors or tiny boxes.
    """
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    img_u8 = img if img.dtype == np.uint8 else img.astype(np.uint8)
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_u8, gc_mask, (x1, y1, bw, bh), bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = np.zeros((h, w), dtype=np.uint8)
        fg[y1:y2, x1:x2] = 255
    return fg


def _iou_px(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _find_free_position(
    crop_w: int,
    crop_h: int,
    canvas_w: int,
    canvas_h: int,
    occupied: list[tuple[int, int, int, int]],
    max_attempts: int = 100,
) -> tuple[int, int] | None:
    """Return (x1, y1) for a crop that does not overlap any occupied rect."""
    max_x = canvas_w - crop_w
    max_y = canvas_h - crop_h
    if max_x < 0 or max_y < 0:
        return None
    for _ in range(max_attempts):
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        candidate = (x1, y1, x1 + crop_w, y1 + crop_h)
        if all(_iou_px(candidate, occ) == 0.0 for occ in occupied):
            return x1, y1
    return None


# ---------------------------------------------------------------------------
# MultiViewAugmentor
# ---------------------------------------------------------------------------

class MultiViewAugmentor:
    """
    Generate multi-view variants of bbox objects via a ComfyUI workflow and
    composite them into new scene images for YOLO dataset expansion.

    Workflow template placeholders
    --------------------------------
    ``__INPUT_IMAGE__``  — filename of the uploaded bbox crop (string).
    ``__AZIMUTH__``      — camera azimuth angle in degrees (number).
    ``__ELEVATION__``    — camera elevation angle in degrees (number).

    Any additional keyword arguments passed to :meth:`augment_sample` or
    :meth:`augment_dataset` are forwarded as extra substitutions.

    Args:
        client:           Initialised :class:`ComfyUIClient`.
        workflow_path:    Path to the ComfyUI multi-view workflow JSON template.
        azimuths:         List of azimuth angles (degrees) to generate.
                          Defaults to [-90, -45, 45, 90].
        elevations:       List of elevation angles (degrees) to pair with each
                          azimuth (cycles if shorter than azimuths).
                          Defaults to [0, 0, 0, 0].
        bg_dir:           Optional directory of background images.  If given,
                          generated crops are composited onto a random background
                          rather than the original scene.
        crop_padding:     Fractional padding added around the bbox before
                          cropping (relative to bbox size).  0.1 = 10% padding.
        use_grabcut:      Refine crop mask with GrabCut before compositing.
        grabcut_iters:    GrabCut iterations.
    """

    def __init__(
        self,
        client: ComfyUIClient,
        workflow_path: str | Path,
        azimuths: list[float] | None = None,
        elevations: list[float] | None = None,
        bg_dir: str | Path | None = None,
        crop_padding: float = 0.1,
        use_grabcut: bool = False,
        grabcut_iters: int = 5,
    ) -> None:
        self.client = client
        self.workflow_template = load_workflow(workflow_path)
        self.azimuths = azimuths if azimuths is not None else [-90.0, -45.0, 45.0, 90.0]
        self.elevations = elevations if elevations is not None else [0.0] * len(self.azimuths)
        self.bg_dir = Path(bg_dir) if bg_dir else None
        self.crop_padding = crop_padding
        self.use_grabcut = use_grabcut
        self.grabcut_iters = grabcut_iters

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _padded_crop(
        self, img: np.ndarray, box: BBox
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return (crop_rgb, (px1, py1, px2, py2)) with padding applied."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _bbox_to_pixels(box, h, w)
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)
        return img[py1:py2, px1:px2].copy(), (px1, py1, px2, py2)

    def _load_random_bg(self, h: int, w: int) -> np.ndarray:
        if self.bg_dir is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in self.bg_dir.iterdir() if p.suffix.lower() in exts]
        if not files:
            return np.zeros((h, w, 3), dtype=np.uint8)
        bg = _load_image(random.choice(files))
        return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    def _composite_into_scene(
        self,
        scene: np.ndarray,
        generated_crop: np.ndarray,
        px1: int, py1: int, px2: int, py2: int,
        original_box: BBox,
    ) -> tuple[np.ndarray, BBox]:
        """
        Resize *generated_crop* to fit the padded region and blend it into *scene*.
        Returns the new scene and updated bbox (unchanged coordinates).
        """
        h, w = scene.shape[:2]
        target_h = py2 - py1
        target_w = px2 - px1
        resized = cv2.resize(generated_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out = scene.copy()
        out[py1:py2, px1:px2] = resized
        return out, original_box

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def augment_sample(
        self,
        image: np.ndarray,
        boxes: list[BBox],
        target_idx: int = 0,
        **extra_patches: Any,
    ) -> list[tuple[np.ndarray, list[BBox]]]:
        """
        Generate multi-view augmentations for *boxes[target_idx]*.

        Returns:
            List of (augmented_image, all_boxes) — one entry per view.
            Boxes for non-target objects are kept unchanged.
        """
        if not boxes or target_idx >= len(boxes):
            return []

        target_box = boxes[target_idx]
        crop, (px1, py1, px2, py2) = self._padded_crop(image, target_box)
        h, w = image.shape[:2]

        # Choose scene base: use bg_dir if provided, else original image.
        if self.bg_dir is not None:
            scene_base = self._load_random_bg(h, w)
            # Paste all non-target boxes onto the background scene
            for i, box in enumerate(boxes):
                if i == target_idx:
                    continue
                bx1, by1, bx2, by2 = _bbox_to_pixels(box, h, w)
                scene_base[by1:by2, bx1:bx2] = image[by1:by2, bx1:bx2]
        else:
            scene_base = image.copy()

        results: list[tuple[np.ndarray, list[BBox]]] = []
        upload_name = f"mv_crop_{id(self)}.png"
        server_name = self.client.upload_image(crop, upload_name)

        for azimuth, elevation in zip(
            self.azimuths,
            [self.elevations[i % len(self.elevations)] for i in range(len(self.azimuths))],
        ):
            workflow = patch_workflow(
                self.workflow_template,
                INPUT_IMAGE=server_name,
                AZIMUTH=azimuth,
                ELEVATION=elevation,
                **extra_patches,
            )
            generated_images = self.client.run_workflow(workflow)
            if not generated_images:
                continue

            # Use the first output image from the workflow
            gen_img = generated_images[0]

            scene, updated_box = self._composite_into_scene(
                scene_base, gen_img, px1, py1, px2, py2, target_box
            )
            new_boxes = list(boxes)
            new_boxes[target_idx] = updated_box
            results.append((scene, new_boxes))

        return results

    def augment_dataset(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        output_image_dir: str | Path,
        output_label_dir: str | Path,
        target_classes: list[int] | None = None,
        max_per_image: int | None = None,
        suffix: str = "_mv",
    ) -> int:
        """
        Run multi-view augmentation over an entire YOLO dataset.

        Args:
            image_dir:        Source images directory.
            label_dir:        Source YOLO labels directory.
            output_image_dir: Where to write augmented images.
            output_label_dir: Where to write augmented YOLO labels.
            target_classes:   If given, only augment boxes belonging to these
                              class IDs.  None = augment all classes.
            max_per_image:    Cap on augmented samples generated per source
                              image (across all its boxes).  None = no cap.
            suffix:           String appended before the extension in output
                              filenames, e.g. ``_mv`` → ``img001_mv_az45.jpg``.

        Returns:
            Total number of new samples written.
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_image_dir = Path(output_image_dir)
        output_label_dir = Path(output_label_dir)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)

        written = 0
        for img_path in image_paths:
            label_path = label_dir / (img_path.stem + ".txt")
            boxes = _load_yolo(label_path)
            if not boxes:
                continue

            image = _load_image(img_path)
            sample_count = 0

            for idx, box in enumerate(boxes):
                cls_id = box[4]
                if target_classes is not None and cls_id not in target_classes:
                    continue
                if max_per_image is not None and sample_count >= max_per_image:
                    break

                augmented = self.augment_sample(image, boxes, target_idx=idx)
                for view_i, (aug_img, aug_boxes) in enumerate(augmented):
                    azimuth = self.azimuths[view_i % len(self.azimuths)]
                    stem = f"{img_path.stem}{suffix}_box{idx}_az{int(azimuth)}"
                    _save_image(output_image_dir / (stem + img_path.suffix), aug_img)
                    _save_yolo(output_label_dir / (stem + ".txt"), aug_boxes)
                    written += 1
                    sample_count += 1
                    if max_per_image is not None and sample_count >= max_per_image:
                        break

        return written


# ---------------------------------------------------------------------------
# BboxScaleAugmentor
# ---------------------------------------------------------------------------

class BboxScaleAugmentor:
    """
    Rescale bbox objects to different sizes and use ComfyUI inpainting to
    seamlessly fill the hole left at the original position.

    Strategy
    --------
    1. Extract the object crop (optionally refined with GrabCut).
    2. Resize the crop to *target_scale* × original size.
    3. Find a non-overlapping placement for the resized crop.
    4. Build a binary mask of the original bbox area.
    5. Send (original image, mask) to ComfyUI inpainting to fill the hole.
    6. Paste the resized crop at the new position on the inpainted image.
    7. Write new image + YOLO labels.

    The object itself is not passed through the generative model — only the
    background reconstruction uses inpainting.  This preserves object identity
    across scale changes.

    If *use_inpainting* is False, a simple background patch from *bg_dir* is
    used instead of ComfyUI (fast but less realistic).

    Workflow template placeholders
    --------------------------------
    ``__INPUT_IMAGE__``  — filename of the scene image before inpainting (string).
    ``__MASK_IMAGE__``   — filename of the inpaint mask (white = region to fill).
    ``__PROMPT__``       — optional text prompt forwarded to the inpainting model.

    Args:
        client:           Initialised :class:`ComfyUIClient`.
        workflow_path:    Path to the ComfyUI inpainting workflow JSON template.
                          Ignored when *use_inpainting* is False.
        scale_values:     List of scale factors to generate.  E.g. [0.5, 2.0].
                          Defaults to [0.5, 0.75, 1.5, 2.0].
        bg_dir:           Background image directory (used when
                          *use_inpainting* is False, or as a fallback).
        use_inpainting:   If True, use ComfyUI to fill holes; otherwise use a
                          background-patch fill.
        use_grabcut:      Refine the crop mask with GrabCut.
        grabcut_iters:    GrabCut iterations.
        inpaint_prompt:   Default text prompt for the inpainting workflow.
        max_placement_attempts: Attempts to find a non-overlapping position.
    """

    def __init__(
        self,
        client: ComfyUIClient,
        workflow_path: str | Path | None = None,
        scale_values: list[float] | None = None,
        bg_dir: str | Path | None = None,
        use_inpainting: bool = True,
        use_grabcut: bool = True,
        grabcut_iters: int = 5,
        inpaint_prompt: str = "background, seamless",
        max_placement_attempts: int = 100,
    ) -> None:
        if use_inpainting and workflow_path is None:
            raise ValueError("workflow_path is required when use_inpainting=True")
        self.client = client
        self.workflow_template = load_workflow(workflow_path) if workflow_path else None
        self.scale_values = scale_values if scale_values is not None else [0.5, 0.75, 1.5, 2.0]
        self.bg_dir = Path(bg_dir) if bg_dir else None
        self.use_inpainting = use_inpainting
        self.use_grabcut = use_grabcut
        self.grabcut_iters = grabcut_iters
        self.inpaint_prompt = inpaint_prompt
        self.max_placement_attempts = max_placement_attempts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_random_bg(self, h: int, w: int) -> np.ndarray | None:
        if self.bg_dir is None:
            return None
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in self.bg_dir.iterdir() if p.suffix.lower() in exts]
        if not files:
            return None
        bg = _load_image(random.choice(files))
        return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    def _fill_hole_with_bg(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fill masked region using a background patch or a blurred neighbour fill."""
        bg = self._load_random_bg(*image.shape[:2])
        if bg is not None:
            mask_f = (mask / 255.0).astype(np.float32)[:, :, np.newaxis]
            return (image.astype(np.float32) * (1 - mask_f) + bg.astype(np.float32) * mask_f).astype(np.uint8)
        # Fallback: inpaint with cv2.INPAINT_TELEA
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def _fill_hole_with_comfyui(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **extra_patches: Any,
    ) -> np.ndarray:
        """Upload image+mask and run the inpainting workflow."""
        uid = f"{id(self)}_{random.randint(0, 99999)}"
        img_name = self.client.upload_image(image, f"inpaint_img_{uid}.png")
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        mask_name = self.client.upload_image(mask_rgb, f"inpaint_mask_{uid}.png")

        workflow = patch_workflow(
            self.workflow_template,
            INPUT_IMAGE=img_name,
            MASK_IMAGE=mask_name,
            PROMPT=self.inpaint_prompt,
            **extra_patches,
        )
        results = self.client.run_workflow(workflow)
        if not results:
            # Fallback if workflow returned nothing
            return self._fill_hole_with_bg(image, mask)
        return cv2.resize(results[0], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def augment_sample(
        self,
        image: np.ndarray,
        boxes: list[BBox],
        target_idx: int = 0,
        target_scale: float | None = None,
        **extra_patches: Any,
    ) -> tuple[np.ndarray, list[BBox]] | None:
        """
        Produce one scaled variant for *boxes[target_idx]*.

        Args:
            image:        Source RGB image.
            boxes:        YOLO-format bboxes (x_min, y_min, x_max, y_max, cls_id).
            target_idx:   Index of the box to rescale.
            target_scale: Scale multiplier.  None → random choice from
                          *self.scale_values*.

        Returns:
            (augmented_image, updated_boxes) or None if placement failed.
        """
        if not boxes or target_idx >= len(boxes):
            return None

        if target_scale is None:
            target_scale = random.choice(self.scale_values)

        h, w = image.shape[:2]
        box = boxes[target_idx]
        x1, y1, x2, y2 = _bbox_to_pixels(box, h, w)

        # --- Step 1: extract object crop ---
        crop = image[y1:y2, x1:x2].copy()
        orig_bw, orig_bh = x2 - x1, y2 - y1
        if orig_bw <= 0 or orig_bh <= 0:
            return None

        # --- Step 2: resize crop ---
        new_bw = max(1, int(round(orig_bw * target_scale)))
        new_bh = max(1, int(round(orig_bh * target_scale)))
        resized_crop = cv2.resize(crop, (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)

        # --- Step 3: find non-overlapping position ---
        occupied = [_bbox_to_pixels(b, h, w) for i, b in enumerate(boxes) if i != target_idx]
        pos = _find_free_position(new_bw, new_bh, w, h, occupied, self.max_placement_attempts)
        if pos is None:
            # Fallback: place at original position (same location, different scale)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            nx1 = max(0, cx - new_bw // 2)
            ny1 = max(0, cy - new_bh // 2)
        else:
            nx1, ny1 = pos
        nx2 = min(w, nx1 + new_bw)
        ny2 = min(h, ny1 + new_bh)

        # --- Step 4: build mask for original position ---
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        if self.use_grabcut:
            fg = _grabcut_refine(image, x1, y1, x2, y2, self.grabcut_iters)
            hole_mask = fg
        else:
            hole_mask[y1:y2, x1:x2] = 255

        # --- Step 5: fill the hole ---
        if self.use_inpainting:
            filled = self._fill_hole_with_comfyui(image, hole_mask, **extra_patches)
        else:
            filled = self._fill_hole_with_bg(image, hole_mask)

        # --- Step 6: paste resized crop at new position ---
        result = filled.copy()
        paste_h = ny2 - ny1
        paste_w = nx2 - nx1
        result[ny1:ny2, nx1:nx2] = resized_crop[:paste_h, :paste_w]

        # --- Step 7: update boxes ---
        new_box: BBox = (
            nx1 / w, ny1 / h, nx2 / w, ny2 / h, box[4]
        )
        new_boxes = list(boxes)
        new_boxes[target_idx] = new_box

        return result, new_boxes

    def augment_dataset(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        output_image_dir: str | Path,
        output_label_dir: str | Path,
        target_classes: list[int] | None = None,
        max_per_image: int | None = None,
        suffix: str = "_scale",
    ) -> int:
        """
        Run scale augmentation over an entire YOLO dataset.

        For each source image × each target box × each scale value, one new
        sample is produced.

        Args:
            image_dir:        Source images directory.
            label_dir:        Source YOLO labels directory.
            output_image_dir: Where to write augmented images.
            output_label_dir: Where to write augmented YOLO labels.
            target_classes:   Restrict augmentation to these class IDs.
            max_per_image:    Cap on samples generated per source image.
            suffix:           Appended to output filename stems.

        Returns:
            Total number of new samples written.
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_image_dir = Path(output_image_dir)
        output_label_dir = Path(output_label_dir)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)

        written = 0
        for img_path in image_paths:
            label_path = label_dir / (img_path.stem + ".txt")
            boxes = _load_yolo(label_path)
            if not boxes:
                continue

            image = _load_image(img_path)
            sample_count = 0

            for idx, box in enumerate(boxes):
                cls_id = box[4]
                if target_classes is not None and cls_id not in target_classes:
                    continue

                for scale in self.scale_values:
                    if max_per_image is not None and sample_count >= max_per_image:
                        break

                    result = self.augment_sample(image, boxes, target_idx=idx, target_scale=scale)
                    if result is None:
                        continue

                    aug_img, aug_boxes = result
                    scale_tag = f"{scale:.2f}".replace(".", "p")
                    stem = f"{img_path.stem}{suffix}_box{idx}_s{scale_tag}"
                    _save_image(output_image_dir / (stem + img_path.suffix), aug_img)
                    _save_yolo(output_label_dir / (stem + ".txt"), aug_boxes)
                    written += 1
                    sample_count += 1

        return written
