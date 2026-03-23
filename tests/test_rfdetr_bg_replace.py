"""
MVP 单测：验证 RF-DETR 训练流程中 BackgroundReplace (GrabCut) 是否生效。

测试分三层：
  1. _grabcut_mask        —— GrabCut mask 比矩形 mask 更精确（面积更小）
  2. BackgroundReplace    —— 背景确实被替换，前景保留；use_grabcut vs 矩形的行为差异
  3. AugmentedCocoDataset —— RF-DETR wrapper 端到端运行正常（Mock base dataset，无需安装 rfdetr）

Run:
    cd /Users/zh/Codes/local/data_enhance
    pytest tests/test_rfdetr_bg_replace.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

# 让 rfdetr_training 可被 import（位于 example/）
sys.path.insert(0, str(Path(__file__).parent.parent / "example"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def structured_image():
    """
    200x200 图片：蓝色背景 + 中心 80x80 红色前景矩形。
    bbox (albumentations 归一化 xyxy): (0.25, 0.25, 0.75, 0.75)，比红色区域略大。
    GrabCut 应能区分红色前景与蓝色背景。
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (0, 0, 200)             # 蓝色背景
    img[60:140, 60:140] = (200, 0, 0)  # 红色前景（行列 60-139）
    return img


@pytest.fixture()
def bg_dir(tmp_path):
    """3 张绿色背景图供 BackgroundReplace 使用。"""
    d = tmp_path / "bgs"
    d.mkdir()
    for i in range(3):
        bg = np.full((200, 200, 3), (0, 180 + i * 10, 0), dtype=np.uint8)
        cv2.imwrite(str(d / f"bg_{i}.jpg"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    return d


# ---------------------------------------------------------------------------
# 1. _grabcut_mask
# ---------------------------------------------------------------------------

class TestGrabcutMask:

    def test_fg_pixels_le_rect_mask(self, structured_image):
        """
        GrabCut mask 前景面积 <= 矩形 mask。
        bbox 比实际红色区域大，边缘蓝色像素应被 GrabCut 判为背景。
        """
        from data_enhance.background_transforms import _grabcut_mask, _union_mask
        img = structured_image
        h, w = img.shape[:2]
        bboxes = [(0.25, 0.25, 0.75, 0.75)]
        rect_fg = int(_union_mask(bboxes, h, w).sum()) // 255
        gc_fg   = int(_grabcut_mask(img, bboxes, h, w, iter_count=5).sum()) // 255
        assert gc_fg <= rect_fg, (
            f"GrabCut fg ({gc_fg}px) should be <= rect mask ({rect_fg}px)"
        )
        assert gc_fg > 0, "GrabCut produced empty mask"

    def test_center_pixel_is_foreground(self, structured_image):
        """图片中心（红色区域）应被 GrabCut 识别为前景。"""
        from data_enhance.background_transforms import _grabcut_mask
        img = structured_image
        h, w = img.shape[:2]
        gc_mask = _grabcut_mask(img, [(0.25, 0.25, 0.75, 0.75)], h, w, iter_count=5)
        assert gc_mask[100, 100] == 255, "Center of red region should be foreground"

    def test_tiny_bbox_fallback_no_exception(self, structured_image):
        """极小 bbox（< 4px）应回退到矩形 mask，不抛异常。"""
        from data_enhance.background_transforms import _grabcut_mask
        img = structured_image
        h, w = img.shape[:2]
        mask = _grabcut_mask(img, [(0.0, 0.0, 0.01, 0.01)], h, w)
        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8

    def test_empty_bboxes_returns_zero_mask(self, structured_image):
        from data_enhance.background_transforms import _grabcut_mask
        img = structured_image
        h, w = img.shape[:2]
        mask = _grabcut_mask(img, [], h, w)
        assert mask.max() == 0


# ---------------------------------------------------------------------------
# 2. BackgroundReplace(use_grabcut=True/False)
# ---------------------------------------------------------------------------

class TestBackgroundReplaceGrabcut:

    def _make_t(self, bg_dir, use_grabcut=True, blend_border=0):
        from data_enhance import BackgroundReplace
        return BackgroundReplace(
            bg_dir=bg_dir, blend_border=blend_border,
            use_grabcut=use_grabcut, grabcut_iters=5, p=1.0,
        )

    def test_output_shape_dtype_preserved(self, structured_image, bg_dir):
        out = self._make_t(bg_dir).apply(
            structured_image, bboxes=[(0.25, 0.25, 0.75, 0.75)]
        )
        assert out.shape == structured_image.shape
        assert out.dtype == structured_image.dtype

    def test_outer_background_replaced(self, structured_image, bg_dir):
        """
        bbox 外的像素（原为蓝色）替换后应变成绿色背景。
        G 通道值 > B 通道值 即视为绿色。
        """
        out = self._make_t(bg_dir, use_grabcut=False).apply(
            structured_image, bboxes=[(0.25, 0.25, 0.75, 0.75)]
        )
        corner = out[10, 10]  # 明确在 bbox 外
        assert int(corner[1]) > int(corner[2]), (
            f"Outer bg should be greenish after replace, got R={corner[0]} G={corner[1]} B={corner[2]}"
        )

    def test_bbox_center_foreground_preserved_rect_mode(self, structured_image, bg_dir):
        """矩形 mask 模式：bbox 内中心（红色前景）应保留原色。"""
        out = self._make_t(bg_dir, use_grabcut=False).apply(
            structured_image, bboxes=[(0.25, 0.25, 0.75, 0.75)]
        )
        center = out[100, 100]
        assert int(center[0]) > int(center[2]), (
            f"Center should remain reddish in rect mode, got {center}"
        )

    def test_grabcut_mode_runs_without_error(self, structured_image, bg_dir):
        """GrabCut 模式完整运行不报错。"""
        out = self._make_t(bg_dir, use_grabcut=True).apply(
            structured_image, bboxes=[(0.25, 0.25, 0.75, 0.75)]
        )
        assert out.shape == structured_image.shape

    def test_grabcut_bbox_corner_differs_from_rect(self, structured_image, bg_dir):
        """
        bbox 内角落（蓝色背景区域，行列约 55）：
          - 矩形模式：在 bbox 内 → 保留蓝色（B > G）
          - GrabCut 模式：蓝色被识别为背景 → 替换为绿色（G > B）
        """
        bboxes = [(0.25, 0.25, 0.75, 0.75)]
        out_rect = self._make_t(bg_dir, use_grabcut=False).apply(structured_image, bboxes=bboxes)
        out_gc   = self._make_t(bg_dir, use_grabcut=True).apply(structured_image, bboxes=bboxes)

        rect_corner = out_rect[55, 55]
        gc_corner   = out_gc[55, 55]

        # 矩形模式：bbox 内角落保留原蓝色
        assert int(rect_corner[2]) > int(rect_corner[1]), (
            f"Rect mode: bbox corner should be blue, got {rect_corner}"
        )
        # GrabCut 模式：蓝色角落应被替换为绿色背景
        assert int(gc_corner[1]) > int(gc_corner[2]), (
            f"GrabCut mode: bbox corner should be replaced to green, got {gc_corner}"
        )

    def test_pipeline_via_albumentations(self, structured_image, bg_dir):
        """通过 A.Compose 管道运行，bbox 数量不变。"""
        import albumentations as A
        from data_enhance import BackgroundReplace

        transform = A.Compose(
            [BackgroundReplace(bg_dir=bg_dir, use_grabcut=True, grabcut_iters=5, p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )
        result = transform(
            image=structured_image,
            bboxes=[[0.5, 0.5, 0.5, 0.5]],
            labels=[0],
        )
        assert result["image"].shape == structured_image.shape
        assert len(result["bboxes"]) == 1


# ---------------------------------------------------------------------------
# 3. AugmentedCocoDataset（Mock base dataset，不依赖 rfdetr 安装）
# ---------------------------------------------------------------------------

class MockCocoDataset:
    """模拟 RF-DETR 返回的 COCO dataset 结构。"""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[60:140, 60:140] = (200, 0, 0)
        img = Image.fromarray(arr)
        target = {
            "image_id": idx,
            "annotations": [
                {
                    "id": idx * 10 + 1,
                    "image_id": idx,
                    "category_id": 0,
                    "bbox": [40.0, 40.0, 120.0, 120.0],  # COCO [x,y,w,h]
                    "area": 14400.0,
                    "iscrowd": 0,
                }
            ],
        }
        return img, target


class TestAugmentedCocoDataset:

    def _make_transform(self, bg_dir):
        import albumentations as A
        from data_enhance import BackgroundReplace
        return A.Compose(
            [BackgroundReplace(bg_dir=bg_dir, use_grabcut=True, grabcut_iters=5, p=1.0)],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.3
            ),
        )

    def test_len_matches_base(self, bg_dir):
        from rfdetr_training import AugmentedCocoDataset
        base = MockCocoDataset()
        ds = AugmentedCocoDataset(base, self._make_transform(bg_dir))
        assert len(ds) == len(base)

    def test_getitem_returns_pil_image(self, bg_dir):
        from rfdetr_training import AugmentedCocoDataset
        ds = AugmentedCocoDataset(MockCocoDataset(), self._make_transform(bg_dir))
        img, _ = ds[0]
        assert isinstance(img, Image.Image)

    def test_getitem_image_size_unchanged(self, bg_dir):
        from rfdetr_training import AugmentedCocoDataset
        ds = AugmentedCocoDataset(MockCocoDataset(), self._make_transform(bg_dir))
        img, _ = ds[0]
        assert img.size == (200, 200)  # PIL: (width, height)

    def test_getitem_annotations_structure(self, bg_dir):
        """增强后 annotations 结构完整：含 bbox（4元素）、category_id、area。"""
        from rfdetr_training import AugmentedCocoDataset
        ds = AugmentedCocoDataset(MockCocoDataset(), self._make_transform(bg_dir))
        _, target = ds[0]
        assert "annotations" in target
        for ann in target["annotations"]:
            assert "bbox" in ann and len(ann["bbox"]) == 4
            assert "category_id" in ann
            assert "area" in ann

    def test_getitem_background_is_replaced(self, bg_dir):
        """
        增强后图片角落应变为绿色背景（非原始黑色/蓝色）。
        验证 BackgroundReplace 在 wrapper 内实际生效。
        """
        from rfdetr_training import AugmentedCocoDataset
        ds = AugmentedCocoDataset(MockCocoDataset(), self._make_transform(bg_dir))
        img, _ = ds[0]
        arr = np.array(img)
        corner = arr[10, 10]  # 明确在 bbox 外
        assert int(corner[1]) > int(corner[2]), (
            f"After bg replace, corner should be greenish, got R={corner[0]} G={corner[1]} B={corner[2]}"
        )
