"""
增强效果可视化工具
==================
加载图片 + 标注，应用 data_enhance 增强 pipeline，
并排展示原图 vs 增强后的图（带 bbox 标注）。

用法:
    # 单张图片可视化
    python demo/visualize.py \
        --image path/to/image.jpg \
        --label path/to/label.txt \
        --bg-dir path/to/backgrounds \
        --output output.png

    # 批量可视化（从数据集目录随机采样）
    python demo/visualize.py \
        --image-dir path/to/images \
        --label-dir path/to/labels \
        --bg-dir path/to/backgrounds \
        --n-samples 8 \
        --output-dir vis_output/

    # 自定义 n 次增强结果网格图
    python demo/visualize.py \
        --image path/to/image.jpg \
        --label path/to/label.txt \
        --bg-dir path/to/backgrounds \
        --n-augments 6 \
        --output grid.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

# 颜色表（BGR），用于区分不同类别
_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def _color_for_class(cls_id: int) -> tuple[int, int, int]:
    return _COLORS[int(cls_id) % len(_COLORS)]


def load_image_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_yolo_labels(label_path: str | Path) -> tuple[list[list[float]], list[int]]:
    """
    读取 YOLO .txt 标注文件。

    Returns:
        bboxes: List of [x_center, y_center, w, h]（归一化）
        class_ids: List of int class ids
    """
    bboxes, class_ids = [], []
    path = Path(label_path)
    if not path.exists():
        return bboxes, class_ids
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes.append([xc, yc, w, h])
            class_ids.append(cls)
    return bboxes, class_ids


def draw_bboxes(
    image: np.ndarray,
    bboxes: list,
    class_labels: list,
    class_names: dict[int, str] | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    在图片上绘制 YOLO 格式的 bbox（归一化坐标）。

    Args:
        image: RGB numpy array
        bboxes: List of [x_center, y_center, w, h]（归一化 0-1）
        class_labels: 对应的类别 id 列表
        class_names: {class_id: name} 映射（可选）
        line_thickness: 边框线宽

    Returns:
        带 bbox 标注的图片（RGB）
    """
    img = image.copy()
    h, w = img.shape[:2]
    for bbox, cls in zip(bboxes, class_labels):
        xc, yc, bw, bh = bbox
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = _color_for_class(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        label = class_names.get(int(cls), str(cls)) if class_names else str(cls)
        font_scale = max(0.4, min(w, h) / 800)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


# ---------------------------------------------------------------------------
# 增强 pipeline 构建
# ---------------------------------------------------------------------------

def build_vis_transform(
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_mosaic: bool = False,
    use_copy_paste: bool = True,
    imgsz: int = 640,
) -> A.Compose:
    from data_enhance import BackgroundReplace, BboxRelocate, CrossImageCopyPaste, Mosaic

    transforms = []

    if use_bg_replace and bg_dir:
        transforms.append(BackgroundReplace(bg_dir=bg_dir, blend_border=8, p=0.7))
    if use_bbox_relocate and bg_dir:
        transforms.append(BboxRelocate(bg_dir=bg_dir, allow_overlap=False, p=0.5))
    if use_mosaic and image_dir and label_dir:
        transforms.append(Mosaic(image_dir=image_dir, label_dir=label_dir, img_size=imgsz, p=0.8))
    if use_copy_paste and image_dir and label_dir:
        transforms.append(CrossImageCopyPaste(
            image_dir=image_dir, label_dir=label_dir,
            max_paste=3, scale_range=(0.5, 1.5), p=0.8,
        ))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


# ---------------------------------------------------------------------------
# 可视化函数
# ---------------------------------------------------------------------------

def visualize_single(
    image: np.ndarray,
    bboxes: list,
    class_ids: list,
    transform: A.Compose,
    class_names: dict | None = None,
    output_path: str | None = None,
    n_augments: int = 1,
):
    """
    展示原图 vs n 次增强结果。

    Args:
        image: 原始图片（RGB numpy）
        bboxes: YOLO 格式 bbox 列表
        class_ids: 类别 id 列表
        transform: Albumentations Compose 对象
        class_names: 类别名称映射
        output_path: 保存路径（None 则显示）
        n_augments: 增强结果数量
    """
    orig_vis = draw_bboxes(image, bboxes, class_ids, class_names)

    aug_results = []
    for _ in range(n_augments):
        result = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        aug_img = draw_bboxes(
            result["image"], result["bboxes"], result["class_labels"], class_names
        )
        aug_results.append(aug_img)

    n_cols = n_augments + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(orig_vis)
    axes[0].set_title(f"原图\n{len(bboxes)} 个目标", fontsize=10)
    axes[0].axis("off")

    for i, aug_img in enumerate(aug_results):
        axes[i + 1].imshow(aug_img)
        n_boxes = len(transform(image=image, bboxes=bboxes, class_labels=class_ids)["bboxes"])
        axes[i + 1].set_title(f"增强 #{i + 1}", fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"保存到: {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_batch(
    image_dir: str | Path,
    label_dir: str | Path,
    transform: A.Compose,
    class_names: dict | None = None,
    n_samples: int = 4,
    output_dir: str | Path | None = None,
):
    """
    从数据集目录随机采样，批量生成可视化结果。

    每个样本生成一张"原图 vs 增强"对比图，保存到 output_dir。
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    if not candidates:
        print(f"未找到图片文件: {image_dir}")
        return

    samples = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"可视化 {len(samples)} 张样本...")

    for img_path in samples:
        image = load_image_rgb(img_path)
        label_path = label_dir / (img_path.stem + ".txt")
        bboxes, class_ids = load_yolo_labels(label_path)

        out_path = str(output_dir / f"{img_path.stem}_aug.png") if output_dir else None
        visualize_single(
            image=image,
            bboxes=bboxes,
            class_ids=class_ids,
            transform=transform,
            class_names=class_names,
            output_path=out_path,
            n_augments=2,
        )

    if output_dir:
        print(f"全部保存到: {output_dir}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="data_enhance 增强效果可视化")

    # 输入
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="单张图片路径")
    group.add_argument("--image-dir", help="批量模式：图片目录")

    parser.add_argument("--label", help="单张模式：YOLO 标注 .txt 路径")
    parser.add_argument("--label-dir", help="批量模式：标注目录")
    parser.add_argument("--class-names", default=None,
                        help="类别名称，格式: '0:cat,1:dog'（可选）")

    # 增强配置
    parser.add_argument("--bg-dir", default=None, help="背景图片目录")
    parser.add_argument("--dataset-image-dir", default=None,
                        help="数据集图片目录（Mosaic/CopyPaste 需要）")
    parser.add_argument("--dataset-label-dir", default=None,
                        help="数据集标注目录（Mosaic/CopyPaste 需要）")
    parser.add_argument("--use-bg-replace", action="store_true", default=True)
    parser.add_argument("--no-bg-replace", dest="use_bg_replace", action="store_false")
    parser.add_argument("--use-mosaic", action="store_true", default=False)
    parser.add_argument("--use-copy-paste", action="store_true", default=True)
    parser.add_argument("--no-copy-paste", dest="use_copy_paste", action="store_false")
    parser.add_argument("--imgsz", type=int, default=640)

    # 输出
    parser.add_argument("--output", default=None, help="单张模式：输出图片路径")
    parser.add_argument("--output-dir", default="vis_output", help="批量模式：输出目录")
    parser.add_argument("--n-samples", type=int, default=4, help="批量模式：采样数量")
    parser.add_argument("--n-augments", type=int, default=3, help="单张模式：增强次数")

    args = parser.parse_args()

    # 解析类别名称
    class_names = None
    if args.class_names:
        class_names = {}
        for item in args.class_names.split(","):
            k, v = item.split(":")
            class_names[int(k.strip())] = v.strip()

    # 构建增强 pipeline
    transform = build_vis_transform(
        bg_dir=args.bg_dir,
        image_dir=args.dataset_image_dir,
        label_dir=args.dataset_label_dir,
        use_bg_replace=args.use_bg_replace,
        use_mosaic=args.use_mosaic,
        use_copy_paste=args.use_copy_paste,
        imgsz=args.imgsz,
    )

    if args.image:
        # 单张模式
        image = load_image_rgb(args.image)
        label_path = args.label or str(Path(args.image).with_suffix(".txt"))
        bboxes, class_ids = load_yolo_labels(label_path)
        print(f"图片: {args.image}, 目标数: {len(bboxes)}")
        visualize_single(
            image=image,
            bboxes=bboxes,
            class_ids=class_ids,
            transform=transform,
            class_names=class_names,
            output_path=args.output,
            n_augments=args.n_augments,
        )
    else:
        # 批量模式
        label_dir = args.label_dir or args.image_dir
        visualize_batch(
            image_dir=args.image_dir,
            label_dir=label_dir,
            transform=transform,
            class_names=class_names,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
