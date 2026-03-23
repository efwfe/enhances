"""
RF-DETR 增强 pipeline 验证脚本
================================
在正式训练前，用与 rfdetr_training.py 完全相同的参数跑几张样图，
目视确认 BackgroundReplace / CrossImageCopyPaste 正常工作且 bbox 对齐。

验证路径与训练完全一致：
    COCO JSON → AugmentedCocoDataset.__getitem__
      → coco_to_yolo → Albumentations transforms → yolo_to_coco
        → 输出对比图

用法（参数与 rfdetr_training.py 保持一致）:
    python example/validate_rfdetr_pipeline.py \
        --dataset-dir ./out \
        --bg-dir ./bg \
        --image-dir ./cpt/train/images/ \
        --label-dir ./cpt/train/labels/ \
        --n-samples 4 \
        --output-dir ./vis_validate/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 复用 rfdetr_training.py 的转换函数和 dataset wrapper
# ---------------------------------------------------------------------------

def coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    return [x_center, y_center, w / img_w, h / img_h]


def yolo_to_coco(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    x_center, y_center, w, h = bbox
    x_min = (x_center - w / 2) * img_w
    y_min = (y_center - h / 2) * img_h
    return [x_min, y_min, w * img_w, h * img_h]


# ---------------------------------------------------------------------------
# 从 COCO JSON 加载样本
# ---------------------------------------------------------------------------

def load_coco_samples(coco_json: Path, n: int) -> list[dict]:
    """
    从 COCO JSON 随机抽 n 张图，返回:
        [{'img_path': Path, 'anns': [{'bbox': [...], 'category_id': int}]}]
    """
    with open(coco_json) as f:
        data = json.load(f)

    img_dir = coco_json.parent
    id2file = {img["id"]: img["file_name"] for img in data["images"]}

    anns_map: dict[int, list] = {img_id: [] for img_id in id2file}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in anns_map and ann.get("bbox"):
            anns_map[img_id].append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
            })

    # 只选有标注的图
    valid_ids = [img_id for img_id, anns in anns_map.items() if anns]
    if not valid_ids:
        raise ValueError(f"COCO JSON 中没有带标注的图片: {coco_json}")

    chosen = random.sample(valid_ids, min(n, len(valid_ids)))
    return [
        {"img_path": img_dir / id2file[img_id], "anns": anns_map[img_id]}
        for img_id in chosen
    ]


# ---------------------------------------------------------------------------
# 绘制 bbox（COCO 格式：x_min, y_min, w, h）
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 56, 56), (255, 157, 151), (72, 249, 10), (0, 194, 255),
    (132, 56, 255), (255, 178, 29), (0, 212, 187), (203, 56, 255),
]


def draw_coco_bboxes(img_bgr: np.ndarray, anns: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for ann in anns:
        x_min, y_min, bw, bh = ann["bbox"]
        cat = ann["category_id"]
        x1, y1 = int(x_min), int(y_min)
        x2, y2 = int(x_min + bw), int(y_min + bh)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = _COLORS[int(cat) % len(_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(cat)
        fs = max(0.4, min(w, h) / 800)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 单张图片的完整 pipeline（与 AugmentedCocoDataset 逻辑相同）
# ---------------------------------------------------------------------------

def run_pipeline(
    img_path: Path,
    anns: list[dict],
    albu_transform,
) -> tuple[np.ndarray, list[dict]]:
    """
    返回 (aug_img_bgr, aug_anns)，其中 aug_anns 为 COCO 格式。
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取: {img_path}")
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 过滤无效 bbox
    valid_anns = []
    bboxes_yolo = []
    class_labels = []
    for ann in anns:
        x_min, y_min, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0:
            continue
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
        return bgr, anns

    result = albu_transform(image=img_rgb, bboxes=bboxes_yolo, class_labels=class_labels)

    aug_rgb = result["image"]
    aug_bboxes = result["bboxes"]
    aug_labels = result["class_labels"]
    new_h, new_w = aug_rgb.shape[:2]

    aug_anns = []
    for i, (bbox_yolo, cat_id) in enumerate(zip(aug_bboxes, aug_labels)):
        coco_bbox = yolo_to_coco(list(bbox_yolo), new_w, new_h)
        aug_anns.append({"bbox": coco_bbox, "category_id": int(cat_id)})

    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
    return aug_bgr, aug_anns


# ---------------------------------------------------------------------------
# 生成对比图（原图 | 增强1 | 增强2）
# ---------------------------------------------------------------------------

def make_comparison(
    img_path: Path,
    anns: list[dict],
    albu_transform,
    n_aug: int = 2,
) -> np.ndarray:
    orig_bgr = cv2.imread(str(img_path))
    orig_vis = draw_coco_bboxes(orig_bgr, anns)

    cols = [orig_vis]
    for i in range(n_aug):
        try:
            aug_bgr, aug_anns = run_pipeline(img_path, anns, albu_transform)
            vis = draw_coco_bboxes(aug_bgr, aug_anns)
        except Exception as e:
            h, w = orig_bgr.shape[:2]
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(vis, f"ERROR: {e}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cols.append(vis)

    # 统一高度后水平拼接
    target_h = max(c.shape[0] for c in cols)
    resized = []
    for c in cols:
        ch, cw = c.shape[:2]
        if ch != target_h:
            scale = target_h / ch
            c = cv2.resize(c, (int(cw * scale), target_h))
        resized.append(c)

    grid = np.concatenate(resized, axis=1)

    # 标注列标题
    labels = ["原图"] + [f"增强 #{i+1}" for i in range(n_aug)]
    col_w = grid.shape[1] // len(labels)
    for i, label in enumerate(labels):
        cv2.putText(grid, label, (i * col_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return grid


# ---------------------------------------------------------------------------
# 统计报告（帮助快速发现问题）
# ---------------------------------------------------------------------------

def run_stats(samples: list[dict], albu_transform, n_runs: int = 10) -> None:
    """对第一张图跑 n_runs 次，统计 bbox 数量变化和异常。"""
    sample = samples[0]
    img_path = sample["img_path"]
    anns = sample["anns"]
    orig_n = len(anns)

    print(f"\n--- 统计校验（{n_runs} 次增强，图片: {img_path.name}）---")
    print(f"原始 bbox 数: {orig_n}")

    counts = []
    errors = 0
    for _ in range(n_runs):
        try:
            _, aug_anns = run_pipeline(img_path, anns, albu_transform)
            counts.append(len(aug_anns))
        except Exception as e:
            errors += 1
            print(f"  [错误] {e}")

    if counts:
        print(f"增强后 bbox 数: min={min(counts)}, max={max(counts)}, "
              f"avg={sum(counts)/len(counts):.1f}")
        if min(counts) == 0:
            print("  [警告] 有增强结果 bbox 全部丢失，检查 min_visibility 参数")
        if max(counts) > orig_n * 5:
            print(f"  [警告] bbox 数量膨胀过多（max={max(counts)}），检查 CopyPaste max_paste")
        else:
            print("  [OK] bbox 数量在合理范围内")
    if errors:
        print(f"  [错误] {errors}/{n_runs} 次运行抛出异常")
    else:
        print(f"  [OK] {n_runs} 次运行无异常")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证 RF-DETR 增强 pipeline（与 rfdetr_training.py 参数相同）"
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="COCO 格式数据集根目录（含 train/_annotations.coco.json）")
    parser.add_argument("--bg-dir", default=None)
    parser.add_argument("--image-dir", default=None,
                        help="YOLO 格式图片目录（CopyPaste 用）")
    parser.add_argument("--label-dir", default=None,
                        help="YOLO 格式标注目录（CopyPaste 用）")
    parser.add_argument("--use-bg-replace", action="store_true", default=True)
    parser.add_argument("--no-bg-replace", dest="use_bg_replace", action="store_false")
    parser.add_argument("--use-bbox-relocate", action="store_true", default=False)
    parser.add_argument("--use-copy-paste", action="store_true", default=True)
    parser.add_argument("--no-copy-paste", dest="use_copy_paste", action="store_false")
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="从 train 集随机抽取的图片数")
    parser.add_argument("--n-aug", type=int, default=2,
                        help="每张图生成的增强结果列数")
    parser.add_argument("--output-dir", default="vis_validate",
                        help="对比图保存目录")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    coco_json = dataset_dir / "train" / "_annotations.coco.json"
    if not coco_json.exists():
        raise FileNotFoundError(f"找不到 COCO 标注: {coco_json}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 与 rfdetr_training.py 完全相同的 pipeline 构建
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from example.rfdetr_training import build_albu_transform

    albu_transform = build_albu_transform(
        bg_dir=args.bg_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        use_bg_replace=args.use_bg_replace,
        use_bbox_relocate=args.use_bbox_relocate,
        use_copy_paste=args.use_copy_paste,
        imgsz=args.resolution,
    )

    print(f"COCO JSON : {coco_json}")
    print(f"bg_dir    : {args.bg_dir}")
    print(f"image_dir : {args.image_dir}")
    print(f"label_dir : {args.label_dir}")
    print(f"输出目录   : {output_dir}")

    samples = load_coco_samples(coco_json, args.n_samples)
    print(f"\n抽取 {len(samples)} 张样本，生成对比图...")

    for i, sample in enumerate(samples):
        img_path = sample["img_path"]
        anns = sample["anns"]
        print(f"  [{i+1}/{len(samples)}] {img_path.name}  bbox={len(anns)}")

        grid = make_comparison(img_path, anns, albu_transform, n_aug=args.n_aug)
        out_path = output_dir / f"{img_path.stem}_validate.jpg"
        cv2.imwrite(str(out_path), grid)

    # 统计校验
    run_stats(samples, albu_transform, n_runs=20)

    print(f"\n完成。对比图保存在: {output_dir}/")
    print("目视检查要点：")
    print("  1. 原图与增强后的 bbox 是否紧贴目标（无明显偏移）")
    print("  2. BackgroundReplace：背景区域是否被替换，目标区域是否保留")
    print("  3. CrossImageCopyPaste：增强图是否出现额外粘贴的目标且带正确 bbox")


if __name__ == "__main__":
    main()
