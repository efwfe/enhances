"""
Unit tests for data_enhance transforms.

Tests use synthetic images and label files created in temp directories
so no real dataset is required.

Run with:
    pytest tests/test_transforms.py -v
    pytest tests/test_transforms.py -v --tb=short   # compact tracebacks
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to create synthetic fixtures
# ---------------------------------------------------------------------------

def make_image(h: int = 200, w: int = 200, channels: int = 3, seed: int = 0) -> np.ndarray:
    """Return a random uint8 RGB image."""
    rng = np.random.default_rng(seed)
    if channels == 1:
        return rng.integers(0, 256, (h, w), dtype=np.uint8)
    return rng.integers(0, 256, (h, w, channels), dtype=np.uint8)


def save_image(path: Path, img: np.ndarray) -> None:
    """Save an RGB numpy array as a JPEG via OpenCV (converts to BGR)."""
    if img.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


def write_yolo_label(path: Path, entries: list[tuple]) -> None:
    """Write YOLO format label file. Each entry: (cls, xc, yc, w, h)."""
    with open(path, "w") as fh:
        for cls, xc, yc, bw, bh in entries:
            fh.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


@pytest.fixture()
def bg_dir(tmp_path):
    """Temp directory with 3 synthetic background images."""
    d = tmp_path / "backgrounds"
    d.mkdir()
    for i in range(3):
        img = make_image(seed=i + 10)
        save_image(d / f"bg_{i}.jpg", img)
    return d


@pytest.fixture()
def dataset_dir(tmp_path):
    """Temp directory with 4 synthetic images and their YOLO label files."""
    img_d = tmp_path / "images"
    lbl_d = tmp_path / "labels"
    img_d.mkdir()
    lbl_d.mkdir()
    entries = [
        (0, 0.5, 0.4, 0.3, 0.4),
        (1, 0.2, 0.7, 0.15, 0.2),
    ]
    for i in range(4):
        img = make_image(seed=i + 20)
        name = f"img_{i}"
        save_image(img_d / f"{name}.jpg", img)
        write_yolo_label(lbl_d / f"{name}.txt", entries)
    return img_d, lbl_d


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestLoadRandomBg:
    def test_returns_correct_shape(self, bg_dir):
        from data_enhance.background_transforms import _load_random_bg
        bg = _load_random_bg(bg_dir, 100, 80)
        assert bg.shape == (100, 80, 3)

    def test_dtype_uint8(self, bg_dir):
        from data_enhance.background_transforms import _load_random_bg
        bg = _load_random_bg(bg_dir, 50, 60)
        assert bg.dtype == np.uint8

    def test_raises_on_empty_dir(self, tmp_path):
        from data_enhance.background_transforms import _load_random_bg
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            _load_random_bg(empty, 100, 100)


class TestUnionMask:
    def test_empty_bboxes_gives_zero_mask(self):
        from data_enhance.background_transforms import _union_mask
        mask = _union_mask([], 100, 100)
        assert mask.max() == 0

    def test_full_bbox_covers_image(self):
        from data_enhance.background_transforms import _union_mask
        # bbox covering entire image
        mask = _union_mask([(0.0, 0.0, 1.0, 1.0)], 100, 100)
        assert mask.min() == 255

    def test_bbox_pixels_are_255(self):
        from data_enhance.background_transforms import _union_mask
        # a bbox in the center (normalised)
        mask = _union_mask([(0.25, 0.25, 0.75, 0.75)], 100, 100)
        # Inside the box
        assert mask[50, 50] == 255
        # Outside the box (top-left corner)
        assert mask[0, 0] == 0

    def test_two_boxes_union(self):
        from data_enhance.background_transforms import _union_mask
        mask = _union_mask(
            [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)],
            100, 100,
        )
        # Top-left quadrant
        assert mask[25, 25] == 255
        # Bottom-right quadrant
        assert mask[75, 75] == 255
        # Middle edge (boundary, could be either, skip)

    def test_shape(self):
        from data_enhance.background_transforms import _union_mask
        mask = _union_mask([(0.1, 0.1, 0.9, 0.9)], 80, 120)
        assert mask.shape == (80, 120)


class TestLoadYoloLabels:
    def test_parses_correctly(self, tmp_path):
        from data_enhance.mosaic_transforms import _load_yolo_labels
        lbl = tmp_path / "test.txt"
        write_yolo_label(lbl, [(0, 0.5, 0.5, 0.4, 0.4)])
        bboxes = _load_yolo_labels(lbl)
        assert len(bboxes) == 1
        x_min, y_min, x_max, y_max, cls_id = bboxes[0]
        assert cls_id == 0
        assert abs(x_min - 0.3) < 1e-4
        assert abs(y_min - 0.3) < 1e-4
        assert abs(x_max - 0.7) < 1e-4
        assert abs(y_max - 0.7) < 1e-4

    def test_returns_empty_for_missing_file(self, tmp_path):
        from data_enhance.mosaic_transforms import _load_yolo_labels
        bboxes = _load_yolo_labels(tmp_path / "nonexistent.txt")
        assert bboxes == []

    def test_clips_to_unit_range(self, tmp_path):
        from data_enhance.mosaic_transforms import _load_yolo_labels
        lbl = tmp_path / "clip.txt"
        # Bbox near edge that would exceed [0,1] without clipping
        write_yolo_label(lbl, [(1, 0.98, 0.98, 0.1, 0.1)])
        bboxes = _load_yolo_labels(lbl)
        for b in bboxes:
            assert b[2] <= 1.0  # x_max clipped
            assert b[3] <= 1.0  # y_max clipped

    def test_skips_degenerate_lines(self, tmp_path):
        from data_enhance.mosaic_transforms import _load_yolo_labels
        lbl = tmp_path / "degen.txt"
        lbl.write_text("0 0.5 0.5\n")  # too few fields
        assert _load_yolo_labels(lbl) == []


class TestIouMosaic:
    def test_no_overlap(self):
        from data_enhance.mosaic_transforms import _iou
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert _iou(a, b) == 0.0

    def test_identical_boxes(self):
        from data_enhance.mosaic_transforms import _iou
        a = (0, 0, 10, 10)
        assert _iou(a, a) == pytest.approx(1.0)

    def test_partial_overlap(self):
        from data_enhance.mosaic_transforms import _iou
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # Intersection = 5*5=25, Union = 100+100-25=175
        assert _iou(a, b) == pytest.approx(25 / 175)


class TestIouBboxRelocate:
    def test_no_overlap(self):
        from data_enhance.background_transforms import BboxRelocate
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert BboxRelocate._iou(a, b) == 0.0

    def test_identical_boxes(self):
        from data_enhance.background_transforms import BboxRelocate
        a = (0, 0, 10, 10)
        assert BboxRelocate._iou(a, a) == pytest.approx(1.0)

    def test_touching_edges_no_overlap(self):
        from data_enhance.background_transforms import BboxRelocate
        a = (0, 0, 10, 10)
        b = (10, 0, 20, 10)  # touching but not overlapping
        assert BboxRelocate._iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# BackgroundReplace tests
# ---------------------------------------------------------------------------

class TestBackgroundReplace:
    """Tests applied via the low-level apply() method (bypasses albumentations pipeline)."""

    def _make_transform(self, bg_dir, blend_border=0):
        from data_enhance import BackgroundReplace
        return BackgroundReplace(bg_dir=bg_dir, blend_border=blend_border, p=1.0)

    def test_output_shape_preserved(self, bg_dir):
        t = self._make_transform(bg_dir)
        img = make_image(100, 120)
        bboxes = [(0.2, 0.2, 0.8, 0.8)]
        out = t.apply(img, bboxes=bboxes)
        assert out.shape == img.shape

    def test_output_dtype_preserved(self, bg_dir):
        t = self._make_transform(bg_dir)
        img = make_image(100, 100)
        out = t.apply(img, bboxes=[(0.1, 0.1, 0.9, 0.9)])
        assert out.dtype == img.dtype

    def test_no_bboxes_replaces_entire_image(self, bg_dir):
        """With no bboxes the entire image should become the background."""
        t = self._make_transform(bg_dir)
        # Solid black image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = t.apply(img, bboxes=[])
        # The background images have random non-zero pixels
        assert out.sum() > 0

    def test_bbox_region_mostly_unchanged(self, bg_dir):
        """Pixels inside the bbox should come from the original image."""
        t = self._make_transform(bg_dir, blend_border=0)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        bbox = (0.3, 0.3, 0.7, 0.7)
        out = t.apply(img, bboxes=[bbox])
        # Center of bbox should still be ~200
        assert out[50, 50, 0] == 200

    def test_blend_border_output_valid(self, bg_dir):
        t = self._make_transform(bg_dir, blend_border=5)
        img = make_image(100, 100)
        out = t.apply(img, bboxes=[(0.2, 0.2, 0.8, 0.8)])
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_apply_to_bboxes_unchanged(self, bg_dir):
        t = self._make_transform(bg_dir)
        bboxes = [(0.1, 0.2, 0.4, 0.6, "cat")]
        result = t.apply_to_bboxes(bboxes)
        assert result == bboxes

    def test_apply_to_mask_unchanged(self, bg_dir):
        t = self._make_transform(bg_dir)
        mask = make_image(100, 100, channels=1)
        result = t.apply_to_mask(mask)
        np.testing.assert_array_equal(result, mask)


# ---------------------------------------------------------------------------
# BboxRelocate tests
# ---------------------------------------------------------------------------

class TestBboxRelocate:

    def _make_transform(self, bg_dir, allow_overlap=True, max_attempts=50):
        from data_enhance import BboxRelocate
        return BboxRelocate(
            bg_dir=bg_dir,
            allow_overlap=allow_overlap,
            max_attempts=max_attempts,
            p=1.0,
        )

    def test_output_shape_preserved(self, bg_dir):
        t = self._make_transform(bg_dir)
        img = make_image(200, 200)
        bboxes = [(0.2, 0.2, 0.5, 0.5)]
        new_positions = [(10, 10, 70, 70)]
        out = t.apply(img, bboxes=bboxes, new_positions=new_positions)
        assert out.shape == img.shape

    def test_no_bboxes_returns_background(self, bg_dir):
        t = self._make_transform(bg_dir)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = t.apply(img, bboxes=[], new_positions=None)
        # Should be a non-zero background
        assert out.sum() > 0

    def test_crop_pasted_at_new_position(self, bg_dir):
        """Solid-color crop should appear exactly at the destination location."""
        t = self._make_transform(bg_dir)
        # Solid red image; bbox covers top-left quarter
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:100, :100] = [255, 0, 0]  # red region
        bboxes_norm = [(0.0, 0.0, 0.5, 0.5)]
        new_positions = [(100, 100, 200, 200)]  # paste to bottom-right
        out = t.apply(img, bboxes=bboxes_norm, new_positions=new_positions)
        # Bottom-right quarter should be red
        assert out[150, 150, 0] == 255
        assert out[150, 150, 1] == 0

    def test_apply_to_bboxes_updates_positions(self, bg_dir):
        t = self._make_transform(bg_dir)
        bboxes = [(0.0, 0.0, 0.5, 0.5, 0)]  # original bbox with class
        new_positions = [(50, 50, 150, 150)]  # pixel coords on 200x200
        result = t.apply_to_bboxes(
            bboxes, new_positions=new_positions, image_shape=(200, 200)
        )
        assert len(result) == 1
        xmin, ymin, xmax, ymax = result[0][:4]
        assert abs(xmin - 50 / 200) < 1e-4
        assert abs(ymin - 50 / 200) < 1e-4
        assert abs(xmax - 150 / 200) < 1e-4
        assert abs(ymax - 150 / 200) < 1e-4

    def test_no_overlap_constraint(self, bg_dir):
        """With allow_overlap=False placed boxes should not overlap."""
        t = self._make_transform(bg_dir, allow_overlap=False, max_attempts=200)
        img = make_image(400, 400)
        # Small bboxes on a large canvas — easy to place without overlap
        bboxes_norm = [
            (0.0, 0.0, 0.1, 0.1),
            (0.2, 0.0, 0.3, 0.1),
            (0.4, 0.0, 0.5, 0.1),
        ]
        from data_enhance.background_transforms import BboxRelocate
        placed: list[tuple] = []
        for xmin, ymin, xmax, ymax in bboxes_norm:
            bw = max(1, int(round((xmax - xmin) * 400)))
            bh = max(1, int(round((ymax - ymin) * 400)))
            x1, y1 = t._find_position(bw, bh, 400, 400, placed)
            x2, y2 = x1 + bw, y1 + bh
            for p in placed:
                assert BboxRelocate._iou((x1, y1, x2, y2), p) == 0.0
            placed.append((x1, y1, x2, y2))

    def test_find_position_returns_within_canvas(self, bg_dir):
        t = self._make_transform(bg_dir)
        x1, y1 = t._find_position(30, 30, 100, 100, [])
        assert 0 <= x1 <= 70
        assert 0 <= y1 <= 70

    def test_find_position_oversized_crop_pins_to_zero(self, bg_dir):
        t = self._make_transform(bg_dir)
        x1, y1 = t._find_position(200, 200, 100, 100, [])
        assert x1 == 0
        assert y1 == 0


# ---------------------------------------------------------------------------
# Mosaic tests
# ---------------------------------------------------------------------------

class TestMosaic:

    def _make_transform(self, dataset_dir, img_size=200):
        from data_enhance import Mosaic
        img_d, lbl_d = dataset_dir
        return Mosaic(
            image_dir=img_d,
            label_dir=lbl_d,
            img_size=img_size,
            center_range=(0.4, 0.6),
            p=1.0,
        )

    def test_output_is_square_canvas(self, dataset_dir):
        t = self._make_transform(dataset_dir, img_size=200)
        img = make_image(100, 100)
        mosaic_data = {
            "cx": 100, "cy": 100,
            "quad_order": [0, 1, 2, 3],
            "extra_images": [make_image(100, 100, seed=i) for i in range(3)],
            "extra_bboxes": [[] for _ in range(3)],
        }
        out = t.apply(img, mosaic_data=mosaic_data)
        assert out.shape == (200, 200, 3)

    def test_none_mosaic_data_resizes_only(self, dataset_dir):
        t = self._make_transform(dataset_dir, img_size=160)
        img = make_image(100, 100)
        out = t.apply(img, mosaic_data=None)
        assert out.shape == (160, 160, 3)

    def test_bboxes_remapped_into_canvas(self, dataset_dir):
        t = self._make_transform(dataset_dir, img_size=200)
        s = 200
        # Current image goes in quadrant 0 (top-left, 0..cx, 0..cy)
        cx, cy = 100, 100
        quad_order = [0, 1, 2, 3]
        mosaic_data = {
            "cx": cx, "cy": cy,
            "quad_order": quad_order,
            "extra_images": [make_image(100, 100, seed=i) for i in range(3)],
            "extra_bboxes": [[] for _ in range(3)],
        }
        # A bbox filling the entire current image (normalised [0,1])
        bboxes = [(0.0, 0.0, 1.0, 1.0, 0)]
        result = t.apply_to_bboxes(bboxes, mosaic_data=mosaic_data)
        assert len(result) == 1
        xmin, ymin, xmax, ymax = result[0][:4]
        # Should map to top-left quadrant in normalised canvas coords
        assert abs(xmin - 0.0) < 1e-4
        assert abs(ymin - 0.0) < 1e-4
        assert abs(xmax - cx / s) < 1e-4
        assert abs(ymax - cy / s) < 1e-4

    def test_extra_bboxes_appended(self, dataset_dir):
        t = self._make_transform(dataset_dir, img_size=200)
        mosaic_data = {
            "cx": 100, "cy": 100,
            "quad_order": [0, 1, 2, 3],
            "extra_images": [make_image(100, 100, seed=i) for i in range(3)],
            "extra_bboxes": [
                [(0.0, 0.0, 1.0, 1.0, 1)],  # one box in each extra image
                [(0.0, 0.0, 1.0, 1.0, 2)],
                [(0.0, 0.0, 1.0, 1.0, 3)],
            ],
        }
        bboxes = [(0.0, 0.0, 1.0, 1.0, 0)]
        result = t.apply_to_bboxes(bboxes, mosaic_data=mosaic_data)
        # 1 current + 3 extra = 4 boxes
        assert len(result) == 4

    def test_all_result_bboxes_in_unit_range(self, dataset_dir):
        t = self._make_transform(dataset_dir, img_size=200)
        mosaic_data = {
            "cx": 100, "cy": 100,
            "quad_order": [0, 1, 2, 3],
            "extra_images": [make_image(100, 100, seed=i) for i in range(3)],
            "extra_bboxes": [[(0.1, 0.1, 0.9, 0.9, i)] for i in range(3)],
        }
        bboxes = [(0.1, 0.1, 0.9, 0.9, 0)]
        for b in t.apply_to_bboxes(bboxes, mosaic_data=mosaic_data):
            xmin, ymin, xmax, ymax = b[:4]
            assert 0.0 <= xmin < xmax <= 1.0
            assert 0.0 <= ymin < ymax <= 1.0


# ---------------------------------------------------------------------------
# CrossImageCopyPaste tests
# ---------------------------------------------------------------------------

class TestCrossImageCopyPaste:

    def _make_transform(self, dataset_dir, allow_overlap=True, max_paste=2):
        from data_enhance import CrossImageCopyPaste
        img_d, lbl_d = dataset_dir
        return CrossImageCopyPaste(
            image_dir=img_d,
            label_dir=lbl_d,
            max_paste=max_paste,
            scale_range=(1.0, 1.0),
            allow_overlap=allow_overlap,
            max_attempts=50,
            p=1.0,
        )

    def _make_paste_items(self):
        crop = make_image(20, 20, seed=99)
        return [{"crop": crop, "dst_px": (10, 10, 30, 30), "cls_id": 5}]

    def test_output_shape_unchanged(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        img = make_image(100, 100)
        out = t.apply(img, paste_items=self._make_paste_items())
        assert out.shape == img.shape

    def test_no_paste_items_returns_original(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        img = make_image(100, 100, seed=7)
        out = t.apply(img, paste_items=[])
        np.testing.assert_array_equal(out, img)

    def test_crop_visible_at_destination(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = np.full((20, 20, 3), 200, dtype=np.uint8)
        paste_items = [{"crop": crop, "dst_px": (10, 10, 30, 30), "cls_id": 0}]
        out = t.apply(img, paste_items=paste_items)
        assert out[15, 15, 0] == 200

    def test_apply_to_bboxes_appends_new_bbox(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        existing = [(0.0, 0.0, 0.1, 0.1, 0)]
        paste_items = [{"crop": make_image(20, 20), "dst_px": (50, 50, 70, 70), "cls_id": 3}]
        result = t.apply_to_bboxes(existing, paste_items=paste_items, image_shape=(100, 100))
        assert len(result) == 2
        # New box coords
        xmin, ymin, xmax, ymax, cls_id = result[1]
        assert abs(xmin - 0.5) < 1e-4
        assert abs(ymin - 0.5) < 1e-4
        assert abs(xmax - 0.7) < 1e-4
        assert abs(ymax - 0.7) < 1e-4
        assert cls_id == 3

    def test_apply_to_bboxes_no_paste_unchanged(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        existing = [(0.1, 0.1, 0.5, 0.5, 0)]
        result = t.apply_to_bboxes(existing, paste_items=[], image_shape=(100, 100))
        assert result == existing

    def test_find_position_respects_canvas_bounds(self, dataset_dir):
        t = self._make_transform(dataset_dir, allow_overlap=True)
        pos = t._find_position(30, 30, 100, 100, [])
        assert pos is not None
        x1, y1 = pos
        assert 0 <= x1 <= 70
        assert 0 <= y1 <= 70

    def test_find_position_returns_none_for_oversized_crop(self, dataset_dir):
        t = self._make_transform(dataset_dir)
        pos = t._find_position(200, 200, 100, 100, [])
        assert pos is None

    def test_no_overlap_when_disabled(self, dataset_dir):
        """Placed boxes must not overlap when allow_overlap=False."""
        from data_enhance.mosaic_transforms import _iou
        t = self._make_transform(dataset_dir, allow_overlap=False)
        placed: list[tuple] = []
        crop_size = 20
        canvas = 200
        for _ in range(10):
            pos = t._find_position(crop_size, crop_size, canvas, canvas, placed)
            if pos is None:
                break
            x1, y1 = pos
            box = (x1, y1, x1 + crop_size, y1 + crop_size)
            for p in placed:
                assert _iou(box, p) == 0.0
            placed.append(box)


# ---------------------------------------------------------------------------
# Integration: transform via Albumentations pipeline
# ---------------------------------------------------------------------------

class TestAlbumentationsPipelineIntegration:
    """Smoke tests running the transforms through A.Compose."""

    def test_background_replace_pipeline(self, bg_dir):
        import albumentations as A
        from data_enhance import BackgroundReplace

        transform = A.Compose(
            [BackgroundReplace(bg_dir=bg_dir, blend_border=0, p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )
        img = make_image(200, 200)
        result = transform(
            image=img,
            bboxes=[[0.5, 0.5, 0.3, 0.3]],
            labels=[0],
        )
        assert result["image"].shape == img.shape
        assert len(result["bboxes"]) == 1

    def test_bbox_relocate_pipeline(self, bg_dir):
        import albumentations as A
        from data_enhance import BboxRelocate

        transform = A.Compose(
            [BboxRelocate(bg_dir=bg_dir, allow_overlap=True, p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )
        img = make_image(200, 200)
        result = transform(
            image=img,
            bboxes=[[0.5, 0.5, 0.2, 0.2]],
            labels=[1],
        )
        assert result["image"].shape == img.shape
        assert len(result["bboxes"]) == 1

    def test_mosaic_pipeline(self, dataset_dir):
        import albumentations as A
        from data_enhance import Mosaic

        img_d, lbl_d = dataset_dir
        transform = A.Compose(
            [Mosaic(image_dir=img_d, label_dir=lbl_d, img_size=200, p=1.0)],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["labels"], min_visibility=0.0
            ),
        )
        img = make_image(100, 100)
        result = transform(
            image=img,
            bboxes=[[0.5, 0.5, 0.3, 0.3]],
            labels=[0],
        )
        assert result["image"].shape == (200, 200, 3)
        # At least one bbox should survive remapping
        assert len(result["bboxes"]) >= 1

    def test_cross_image_copy_paste_pipeline(self, dataset_dir):
        import albumentations as A
        from data_enhance import CrossImageCopyPaste

        img_d, lbl_d = dataset_dir
        transform = A.Compose(
            [CrossImageCopyPaste(
                image_dir=img_d, label_dir=lbl_d,
                max_paste=2, scale_range=(1.0, 1.0), p=1.0,
            )],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["labels"], min_visibility=0.0
            ),
        )
        img = make_image(200, 200)
        result = transform(
            image=img,
            bboxes=[[0.5, 0.5, 0.2, 0.2]],
            labels=[0],
        )
        assert result["image"].shape == img.shape
        # After pasting, there should be at least the original bbox
        assert len(result["bboxes"]) >= 1

    def test_empty_bboxes_pipeline(self, bg_dir):
        """Transforms should handle empty bbox lists gracefully."""
        import albumentations as A
        from data_enhance import BackgroundReplace, BboxRelocate

        transform = A.Compose(
            [
                BackgroundReplace(bg_dir=bg_dir, p=1.0),
                BboxRelocate(bg_dir=bg_dir, p=1.0),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )
        img = make_image(100, 100)
        result = transform(image=img, bboxes=[], labels=[])
        assert result["image"].shape == img.shape
        assert result["bboxes"] == []
