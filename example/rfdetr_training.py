"""
RF-DETR 训练集成 Demo
=====================
将 data_enhance 的自定义增强注入到 RF-DETR（Roboflow）训练流程中。

集成方式：Dataset Wrapper（在线增强）
- Monkey-patch RFDETRDataModule.setup，在 _dataset_train 构建后立即包装
- 在每次 __getitem__ 时应用 Albumentations 增强
- 处理 COCO bbox ↔ YOLO 归一化格式转换

数据集格式（COCO）:
    dataset_dir/
      train/
        _annotations.coco.json
        image1.jpg
        ...
      valid/
        _annotations.coco.json
        ...

用法:
    python demo/rfdetr_training.py \
        --dataset-dir path/to/coco_dataset \
        --epochs 50 \
        --batch-size 4 \
        --bg-dir path/to/backgrounds
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np


# ---------------------------------------------------------------------------
# 格式转换工具
# ---------------------------------------------------------------------------

def coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """
    COCO bbox → YOLO 归一化格式。

    Args:
        bbox: [x_min, y_min, width, height]（绝对像素）
        img_w: 图片宽度
        img_h: 图片高度

    Returns:
        [x_center, y_center, width, height]（归一化 0-1）
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    return [x_center, y_center, w / img_w, h / img_h]


def yolo_to_coco(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """
    YOLO 归一化格式 → COCO bbox。

    Args:
        bbox: [x_center, y_center, width, height]（归一化 0-1）
        img_w: 图片宽度
        img_h: 图片高度

    Returns:
        [x_min, y_min, width, height]（绝对像素）
    """
    x_center, y_center, w, h = bbox
    x_min = (x_center - w / 2) * img_w
    y_min = (y_center - h / 2) * img_h
    return [x_min, y_min, w * img_w, h * img_h]


# ---------------------------------------------------------------------------
# Dataset Wrapper
# ---------------------------------------------------------------------------

class AugmentedCocoDataset:
    """
    包装 RF-DETR 内部的 COCO Dataset，注入 Albumentations 增强。

    RF-DETR 的 dataset.__getitem__ 返回:
        image (PIL.Image), target (dict with 'annotations' list)

    每个 annotation 包含:
        {'id': int, 'image_id': int, 'category_id': int,
         'bbox': [x_min, y_min, w, h],  # 绝对像素, COCO 格式
         'area': float, 'iscrowd': int, ...}

    增强后保持相同结构，更新 image 和 bbox 坐标。
    """

    def __init__(self, base_dataset, albu_transform: A.Compose):
        """
        Args:
            base_dataset: RF-DETR 原始 COCO dataset 对象
            albu_transform: Albumentations Compose 对象（YOLO bbox 格式）
        """
        self.base = base_dataset
        self.transform = albu_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        from PIL import Image

        image, target = self.base[idx]

        # PIL → numpy RGB
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        annotations = target.get("annotations", [])
        if not annotations:
            return image, target

        # COCO bbox → YOLO 归一化格式
        bboxes_yolo = []
        class_labels = []
        valid_anns = []

        for ann in annotations:
            x_min, y_min, bw, bh = ann["bbox"]
            # 过滤无效 bbox
            if bw <= 0 or bh <= 0:
                continue
            # 裁剪到图片范围
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            bw = min(bw, w - x_min)
            bh = min(bh, h - y_min)
            if bw <= 0 or bh <= 0:
                continue
            bboxes_yolo.append(coco_to_yolo([x_min, y_min, bw, bh], w, h))
            class_labels.append(ann["category_id"])
            valid_anns.append(ann)

        if not bboxes_yolo:
            return image, target

        # 应用 Albumentations 增强
        result = self.transform(
            image=img_np,
            bboxes=bboxes_yolo,
            class_labels=class_labels,
        )

        aug_img_np = result["image"]
        aug_bboxes = result["bboxes"]
        aug_labels = result["class_labels"]

        # 转回 PIL
        aug_image = Image.fromarray(aug_img_np)
        new_h, new_w = aug_img_np.shape[:2]

        # YOLO 归一化格式 → COCO bbox，重建 annotations
        new_annotations = []
        for i, (bbox_yolo, cat_id) in enumerate(zip(aug_bboxes, aug_labels)):
            coco_bbox = yolo_to_coco(list(bbox_yolo), new_w, new_h)
            bw, bh = coco_bbox[2], coco_bbox[3]
            # 复用原 annotation 的元数据（id、iscrowd 等），更新 bbox 和 area
            base_ann = valid_anns[i] if i < len(valid_anns) else {}
            new_ann = {
                **base_ann,
                "category_id": int(cat_id),
                "bbox": coco_bbox,
                "area": float(bw * bh),
            }
            new_annotations.append(new_ann)

        new_target = {**target, "annotations": new_annotations}
        return aug_image, new_target

    # 代理其他属性到 base dataset（RF-DETR 可能访问 .coco 等属性）
    def __getattr__(self, name: str):
        return getattr(self.base, name)


# ---------------------------------------------------------------------------
# 构建增强 pipeline
# ---------------------------------------------------------------------------

def build_albu_transform(
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_copy_paste: bool = True,
    imgsz: int = 560,
) -> A.Compose:
    """
    构建用于 RF-DETR 的 Albumentations transform pipeline。

    注意：不在此处使用 Mosaic，因为 RF-DETR 有自己的多尺度处理。
    如需 Mosaic，建议在离线预处理阶段使用。
    """
    from data_enhance import BackgroundReplace, BboxRelocate, CrossImageCopyPaste

    transforms = []

    if use_bg_replace and bg_dir:
        transforms.append(
            BackgroundReplace(bg_dir=bg_dir, blend_border=8, p=0.3)
        )

    if use_bbox_relocate and bg_dir:
        transforms.append(
            BboxRelocate(bg_dir=bg_dir, allow_overlap=False, max_attempts=50, p=0.2)
        )

    if use_copy_paste and image_dir and label_dir:
        transforms.append(
            CrossImageCopyPaste(
                image_dir=image_dir,
                label_dir=label_dir,
                max_paste=3,
                scale_range=(0.5, 1.2),
                allow_overlap=False,
                p=0.4,
            )
        )

    # 追加一些通用增强
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# RF-DETR 训练注入
# ---------------------------------------------------------------------------

def _patch_datamodule_setup(albu_transform: A.Compose) -> None:
    """
    Monkey-patch RFDETRDataModule.setup，在 _dataset_train 构建后立即用
    AugmentedCocoDataset 包装，train_dataloader() 创建的 DataLoader 就会
    使用包装后的数据集。

    注意：patch 在进程级别生效，调用 model.train() 后应调用 _unpatch_datamodule_setup
    还原，以免影响后续训练调用。
    """
    from rfdetr.training import RFDETRDataModule

    original_setup = RFDETRDataModule.setup

    def patched_setup(self, stage: str) -> None:
        original_setup(self, stage)
        if stage == "fit" and self._dataset_train is not None:
            print("[data_enhance] 注入增强到 train dataset")
            self._dataset_train = AugmentedCocoDataset(self._dataset_train, albu_transform)

    RFDETRDataModule.setup = patched_setup
    RFDETRDataModule._original_setup = original_setup


def _unpatch_datamodule_setup() -> None:
    """还原 RFDETRDataModule.setup 为原始实现。"""
    try:
        from rfdetr.training import RFDETRDataModule
        if hasattr(RFDETRDataModule, "_original_setup"):
            RFDETRDataModule.setup = RFDETRDataModule._original_setup
            del RFDETRDataModule._original_setup
    except Exception:
        pass


def train_with_rfdetr(
    dataset_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    resolution: int = 560,
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_copy_paste: bool = True,
    grad_accum_steps: int = 4,
    output_dir: str = "runs/rfdetr",
):
    """
    使用自定义增强训练 RF-DETR 模型。

    注入方式：在 model.train() 前 monkey-patch RFDETRDataModule.setup，
    使 _dataset_train 在构建完成后立即被 AugmentedCocoDataset 包装。

    数据集目录结构（COCO 格式）:
        dataset_dir/
          train/
            _annotations.coco.json
            *.jpg
          valid/
            _annotations.coco.json
            *.jpg

    Args:
        dataset_dir: COCO 格式数据集根目录
        epochs: 训练轮数
        batch_size: 每批次样本数
        lr: 学习率
        resolution: 输入分辨率（RF-DETR 常用 560）
        bg_dir: 背景图片目录
        image_dir: 训练图片目录（YOLO 格式，供 CopyPaste 使用）
        label_dir: YOLO 标注目录（供 CopyPaste 使用）
        use_bg_replace: 是否启用背景替换
        use_bbox_relocate: 是否启用 bbox 重定位
        use_copy_paste: 是否启用跨图复制粘贴
        grad_accum_steps: 梯度累积步数
        output_dir: 输出目录
    """
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        raise ImportError("请先安装 rfdetr: pip install rfdetr")

    # 构建增强 pipeline
    albu_transform = build_albu_transform(
        bg_dir=bg_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        use_bg_replace=use_bg_replace,
        use_bbox_relocate=use_bbox_relocate,
        use_copy_paste=use_copy_paste,
        imgsz=resolution,
    )

    print("=== data_enhance RF-DETR 训练 Demo ===")
    print(f"数据集:      {dataset_dir}")
    print(f"训练轮数:    {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"分辨率:      {resolution}")
    print(f"背景替换:    {'是' if use_bg_replace and bg_dir else '否'}")
    print(f"Bbox重定位:  {'是' if use_bbox_relocate and bg_dir else '否'}")
    print(f"跨图复制粘贴:{'是' if use_copy_paste and image_dir else '否'}")
    print()

    # 注入增强：patch DataModule.setup，train() 内部构建数据集时自动生效
    _patch_datamodule_setup(albu_transform)
    try:
        model = RFDETRBase()
        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            resolution=resolution,
            output_dir=output_dir,
        )
    finally:
        _unpatch_datamodule_setup()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="使用 data_enhance 自定义增强训练 RF-DETR"
    )
    parser.add_argument("--dataset-dir", required=True, help="COCO 格式数据集根目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--resolution", type=int, default=560, help="输入分辨率")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--output-dir", default="runs/rfdetr", help="输出目录")

    # 增强配置
    parser.add_argument("--bg-dir", default=None, help="背景图片目录")
    parser.add_argument("--image-dir", default=None,
                        help="YOLO 格式训练图片目录（CopyPaste 需要）")
    parser.add_argument("--label-dir", default=None,
                        help="YOLO 格式标注目录（CopyPaste 需要）")
    parser.add_argument("--use-bg-replace", action="store_true", default=True)
    parser.add_argument("--no-bg-replace", dest="use_bg_replace", action="store_false")
    parser.add_argument("--use-bbox-relocate", action="store_true", default=False)
    parser.add_argument("--use-copy-paste", action="store_true", default=True)
    parser.add_argument("--no-copy-paste", dest="use_copy_paste", action="store_false")

    args = parser.parse_args()

    train_with_rfdetr(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resolution=args.resolution,
        bg_dir=args.bg_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        use_bg_replace=args.use_bg_replace,
        use_bbox_relocate=args.use_bbox_relocate,
        use_copy_paste=args.use_copy_paste,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
