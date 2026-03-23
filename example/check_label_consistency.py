"""
COCO JSON ↔ YOLO .txt 标签一致性检查（混淆矩阵）
=================================================

用途：验证 YOLO .txt 文件中的 class_id 与 COCO JSON 里的 category_id 是否对齐。
CrossImageCopyPaste 从 YOLO .txt 读取 class_id 后直接塞回 RF-DETR 的 bbox 列表，
若两者编号不一致，训练时类别会错乱——本脚本通过 IoU 匹配逐框对比，输出混淆矩阵。

输入数据结构（RF-DETR COCO 格式）:
    dataset_dir/
      train/
        _annotations.coco.json
        img001.jpg
        ...

用法:
    # 只检查 COCO JSON（无 YOLO txt，查看类别分布）
    python example/check_label_consistency.py \
        --coco-json /path/to/train/_annotations.coco.json

    # 检查 COCO ↔ YOLO 一致性（有 YOLO txt 时）
    python example/check_label_consistency.py \
        --coco-json /path/to/train/_annotations.coco.json \
        --label-dir /path/to/train/labels \
        --iou-thresh 0.5

解读输出：
    - 混淆矩阵行 = COCO category_id，列 = YOLO class_id
    - 对角线全为非零、非对角线全为零 → 完全一致
    - 非对角线有值 → 存在类别 ID 错位，需要检查标注
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# COCO JSON 加载
# ---------------------------------------------------------------------------

def load_coco(coco_json: Path) -> tuple[dict, dict, dict]:
    """
    返回:
        images   : {image_id -> file_name}
        anns_map : {image_id -> [{'bbox': [x,y,w,h], 'category_id': int}]}
        cats     : {category_id -> name}
    """
    with open(coco_json) as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}

    anns_map: dict[int, list] = {img_id: [] for img_id in images}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in anns_map:
            anns_map[img_id].append({
                "bbox": ann["bbox"],          # [x_min, y_min, w, h] 绝对像素
                "category_id": ann["category_id"],
            })

    return images, anns_map, cats


# ---------------------------------------------------------------------------
# YOLO .txt 加载
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    读取 YOLO .txt，返回绝对像素 bbox 列表:
        [{'bbox': [x_min, y_min, w, h], 'class_id': int}]
    """
    results = []
    if not label_path.exists():
        return results
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            x_min = (xc - bw / 2) * img_w
            y_min = (yc - bh / 2) * img_h
            results.append({
                "bbox": [x_min, y_min, bw * img_w, bh * img_h],
                "class_id": cls_id,
            })
    return results


# ---------------------------------------------------------------------------
# IoU（xywh 格式）
# ---------------------------------------------------------------------------

def iou_xywh(a: list[float], b: list[float]) -> float:
    """计算两个 [x_min, y_min, w, h] bbox 的 IoU。"""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 混淆矩阵构建
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    coco_json: Path,
    label_dir: Path | None,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, list[int], list[int], dict[int, str]]:
    """
    遍历所有图片，对每张图：
      - 若有 label_dir：按 IoU 匹配 COCO 框和 YOLO 框，统计 (coco_cat_id, yolo_cls_id) 对
      - 若无 label_dir：只统计 COCO 类别分布（对角矩阵）

    返回:
        matrix    : confusion matrix，shape (n_coco_cats, n_yolo_cls)
        coco_ids  : 行对应的 category_id 列表
        yolo_ids  : 列对应的 class_id 列表
        cats      : {category_id -> name}
    """
    images, anns_map, cats = load_coco(coco_json)
    img_dir = coco_json.parent

    # 收集所有出现的 ID
    all_coco_ids: set[int] = set()
    all_yolo_ids: set[int] = set()

    # 先扫一遍收集 ID 范围
    for img_id, anns in anns_map.items():
        for ann in anns:
            all_coco_ids.add(ann["category_id"])

    matched_pairs: list[tuple[int, int]] = []   # (coco_cat_id, yolo_cls_id)
    unmatched_coco = 0   # COCO 框找不到对应 YOLO 框
    unmatched_yolo = 0   # YOLO 框找不到对应 COCO 框
    missing_label_files = 0

    if label_dir is not None:
        for img_id, anns in anns_map.items():
            if not anns:
                continue
            file_name = images[img_id]
            stem = Path(file_name).stem

            # 需要图片尺寸来反算 YOLO 归一化坐标
            img_path = img_dir / file_name
            if img_path.exists():
                import cv2
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                img_h, img_w = bgr.shape[:2]
            else:
                # 从 COCO bbox 估算（仅在图片缺失时用）
                all_x2 = [a["bbox"][0] + a["bbox"][2] for a in anns]
                all_y2 = [a["bbox"][1] + a["bbox"][3] for a in anns]
                img_w, img_h = int(max(all_x2)) + 1, int(max(all_y2)) + 1

            label_path = label_dir / (stem + ".txt")
            if not label_path.exists():
                missing_label_files += 1
                continue

            yolo_anns = load_yolo_labels(label_path, img_w, img_h)
            for ya in yolo_anns:
                all_yolo_ids.add(ya["class_id"])

            # 贪心 IoU 匹配（最高 IoU 优先）
            matched_coco = set()
            matched_yolo_idx = set()

            score_matrix = [
                [iou_xywh(ca["bbox"], ya["bbox"]) for ya in yolo_anns]
                for ca in anns
            ]

            # 按 IoU 降序匹配
            pairs_sorted = sorted(
                [(i, j, score_matrix[i][j])
                 for i in range(len(anns))
                 for j in range(len(yolo_anns))],
                key=lambda x: x[2],
                reverse=True,
            )
            for ci, yi, iou in pairs_sorted:
                if iou < iou_thresh:
                    break
                if ci in matched_coco or yi in matched_yolo_idx:
                    continue
                matched_coco.add(ci)
                matched_yolo_idx.add(yi)
                matched_pairs.append((anns[ci]["category_id"], yolo_anns[yi]["class_id"]))

            unmatched_coco += len(anns) - len(matched_coco)
            unmatched_yolo += len(yolo_anns) - len(matched_yolo_idx)
    else:
        # 无 YOLO txt：只展示 COCO 类别分布（自对自，对角线）
        for img_id, anns in anns_map.items():
            for ann in anns:
                cid = ann["category_id"]
                matched_pairs.append((cid, cid))
                all_yolo_ids.add(cid)

    coco_ids = sorted(all_coco_ids)
    yolo_ids = sorted(all_yolo_ids)
    coco_idx = {v: i for i, v in enumerate(coco_ids)}
    yolo_idx = {v: i for i, v in enumerate(yolo_ids)}

    matrix = np.zeros((len(coco_ids), len(yolo_ids)), dtype=int)
    for cid, yid in matched_pairs:
        if cid in coco_idx and yid in yolo_idx:
            matrix[coco_idx[cid], yolo_idx[yid]] += 1

    return matrix, coco_ids, yolo_ids, cats, unmatched_coco, unmatched_yolo, missing_label_files


# ---------------------------------------------------------------------------
# 打印输出
# ---------------------------------------------------------------------------

def print_report(
    matrix: np.ndarray,
    coco_ids: list[int],
    yolo_ids: list[int],
    cats: dict[int, str],
    unmatched_coco: int,
    unmatched_yolo: int,
    missing_label_files: int,
    label_dir: Path | None,
) -> None:
    n_coco = len(coco_ids)
    n_yolo = len(yolo_ids)

    print("\n" + "=" * 60)
    if label_dir is not None:
        print("COCO ↔ YOLO 标签一致性混淆矩阵")
        print("行 = COCO category_id，列 = YOLO class_id")
    else:
        print("COCO 类别分布（无 YOLO 标注目录，仅展示 COCO 统计）")
        print("行 = COCO category_id，列 = category_id（同轴）")
    print("=" * 60)

    # 表头
    col_w = 8
    cat_w = 20
    header = f"{'COCO cat':<{cat_w}}"
    for yid in yolo_ids:
        header += f"{'y=' + str(yid):>{col_w}}"
    header += f"{'total':>{col_w}}"
    print(header)
    print("-" * (cat_w + col_w * (n_yolo + 1)))

    for i, cid in enumerate(coco_ids):
        cat_name = cats.get(cid, "?")
        row_label = f"{cid}:{cat_name}"[:cat_w - 1]
        row = f"{row_label:<{cat_w}}"
        total = int(matrix[i].sum())
        for j in range(n_yolo):
            val = matrix[i, j]
            marker = "*" if (label_dir is not None and i != j and val > 0) else " "
            row += f"{str(val) + marker:>{col_w}}"
        row += f"{total:>{col_w}}"
        print(row)

    print("-" * (cat_w + col_w * (n_yolo + 1)))
    col_totals = f"{'total':<{cat_w}}"
    for j in range(n_yolo):
        col_totals += f"{int(matrix[:, j].sum()):>{col_w}}"
    col_totals += f"{int(matrix.sum()):>{col_w}}"
    print(col_totals)

    # 类别映射表
    print("\n类别名称映射：")
    for cid in coco_ids:
        print(f"  category_id={cid:3d}  →  {cats.get(cid, '(未知)')}")

    # 一致性判断（仅在有 YOLO txt 时）
    if label_dir is not None:
        print(f"\n未匹配框：COCO={unmatched_coco}，YOLO={unmatched_yolo}")
        if missing_label_files > 0:
            print(f"缺失 YOLO label 文件数：{missing_label_files}")

        is_diagonal = all(
            coco_ids[i] == yolo_ids[j] if (i < n_yolo and j < n_coco) else True
            for i in range(n_coco)
            for j in range(n_yolo)
            if matrix[i, j] > 0 and i != j
        )
        off_diag = int((matrix - np.diag(np.diag(matrix))).sum()) if n_coco == n_yolo else -1

        print()
        if n_coco != n_yolo:
            print("[警告] COCO 类别数与 YOLO 类别数不同，存在 ID 错位风险")
        elif off_diag == 0:
            print("[OK] 对角线匹配完美，COCO category_id 与 YOLO class_id 完全一致")
        else:
            print(f"[警告] 非对角线共 {off_diag} 个匹配对（标 * 处），存在类别 ID 错位！")
            print("       建议检查 COCO JSON 与 YOLO txt 的类别编号是否使用相同起始值（0 vs 1）")
    print()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 COCO JSON 与 YOLO .txt 标注的类别 ID 一致性"
    )
    parser.add_argument(
        "--coco-json", required=True,
        help="COCO 格式标注文件，例如 train/_annotations.coco.json",
    )
    parser.add_argument(
        "--label-dir", default=None,
        help="YOLO .txt 标注目录（与图片同名 .txt）；不传则仅展示 COCO 类别分布",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.5,
        help="IoU 匹配阈值（默认 0.5）",
    )
    args = parser.parse_args()

    coco_json = Path(args.coco_json)
    label_dir = Path(args.label_dir) if args.label_dir else None

    if not coco_json.exists():
        raise FileNotFoundError(f"找不到 COCO JSON：{coco_json}")
    if label_dir is not None and not label_dir.exists():
        raise FileNotFoundError(f"找不到 YOLO label 目录：{label_dir}")

    print(f"COCO JSON : {coco_json}")
    if label_dir:
        print(f"YOLO dir  : {label_dir}")
        print(f"IoU 阈值  : {args.iou_thresh}")

    matrix, coco_ids, yolo_ids, cats, unc, uny, miss = build_confusion_matrix(
        coco_json, label_dir, args.iou_thresh
    )
    print_report(matrix, coco_ids, yolo_ids, cats, unc, uny, miss, label_dir)


if __name__ == "__main__":
    main()
