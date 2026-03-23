"""
RF-DETR ONNX 推理脚本
=====================
支持将 RF-DETR PyTorch 模型导出为 ONNX，并使用 onnxruntime 进行推理。

功能：
  1. 导出：将训练好的 RF-DETR 权重导出为 ONNX 格式
  2. 推理：加载 ONNX 模型对单张图片或目录批量推理
  3. 可视化：推理结果绘制并保存

ONNX 模型 I/O：
  输入  image  [1, 3, H, W]  float32，ImageNet 归一化
  输出  logits [1, Q, C]     float32，类别 logits（sigmoid 后为概率）
       boxes  [1, Q, 4]     float32，[cx, cy, w, h] 归一化坐标

用法示例：
  # 导出（分辨率必须被 32 整除，推荐 448）
  python rfdetr_onnx_infer.py export \
      --weights runs/rfdetr/checkpoint.pth \
      --output model.onnx \
      --resolution 448

  # 单张推理
  python rfdetr_onnx_infer.py infer \
      --model model.onnx \
      --source image.jpg \
      --conf 0.5 \
      --labels-file labels.txt

  # 目录批量推理
  python rfdetr_onnx_infer.py infer \
      --model model.onnx \
      --source images/ \
      --conf 0.5 \
      --output-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ImageNet 归一化参数
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DINOv2 backbone 要求输入尺寸同时能被 patch_size=14 和 stride=32 整除
# 即必须是 LCM(14, 32)=224 的倍数：224, 448, 672, 896...
_BACKBONE_STRIDE = 32
_DINOV2_PATCH    = 14


def _validate_resolution(resolution: int) -> int:
    """
    检查分辨率是否满足 DINOv2 backbone 约束（需被 32 整除）。
    如果不满足则报错，并提示最近的合法值。
    """
    if resolution % _BACKBONE_STRIDE != 0:
        lo = (resolution // _BACKBONE_STRIDE) * _BACKBONE_STRIDE
        hi = lo + _BACKBONE_STRIDE
        raise ValueError(
            f"分辨率 {resolution} 不能被 {_BACKBONE_STRIDE} 整除，"
            f"DINOv2 backbone 会报 AssertionError。\n"
            f"请改用 {lo} 或 {hi}（推荐 448）。"
        )
    return resolution

# 支持的图片扩展名
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray, resolution: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    将 BGR 图像预处理为模型输入 tensor。

    Args:
        img_bgr: OpenCV 读取的 BGR 图像 [H, W, 3]
        resolution: 模型输入分辨率（正方形边长）

    Returns:
        blob: float32 数组 [1, 3, resolution, resolution]，已归一化
        orig_size: 原始图像尺寸 (orig_h, orig_w)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - _MEAN) / _STD          # HWC
    blob = img_norm.transpose(2, 0, 1)[np.newaxis]  # → 1CHW
    return blob, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------------------

def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    orig_size: tuple[int, int],
    conf_threshold: float = 0.5,
) -> list[dict]:
    """
    解码 RF-DETR ONNX 模型输出。

    Args:
        logits: [1, Q, C] 类别 logits
        boxes:  [1, Q, 4] 归一化 [cx, cy, w, h]
        orig_size: 原始图像尺寸 (orig_h, orig_w)
        conf_threshold: 置信度阈值

    Returns:
        检测结果列表，每项为：
            {'bbox': [x1, y1, x2, y2],  # 绝对像素坐标
             'score': float,
             'class_id': int}
    """
    orig_h, orig_w = orig_size

    # logits → 概率（RF-DETR 使用 sigmoid 多标签分类）
    scores_all = 1.0 / (1.0 + np.exp(-logits[0]))  # [Q, C]

    # 取每个 query 的最大类别分数
    class_ids = scores_all.argmax(axis=-1)           # [Q]
    scores = scores_all[np.arange(len(class_ids)), class_ids]  # [Q]

    # 过滤低置信度
    keep = scores >= conf_threshold
    if not keep.any():
        return []

    scores    = scores[keep]
    class_ids = class_ids[keep]
    bboxes_n  = boxes[0][keep]  # [K, 4] cxcywh 归一化

    # cxcywh 归一化 → xyxy 绝对像素
    cx, cy, bw, bh = bboxes_n[:, 0], bboxes_n[:, 1], bboxes_n[:, 2], bboxes_n[:, 3]
    x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
    y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
    x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
    y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)

    results = []
    for i in range(len(scores)):
        results.append({
            "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
        })

    # 按置信度降序排列
    results.sort(key=lambda d: d["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def draw_detections(
    img_bgr: np.ndarray,
    detections: list[dict],
    class_names: list[str] | None = None,
) -> np.ndarray:
    """
    在图像上绘制检测框和标签。

    Args:
        img_bgr: BGR 图像
        detections: postprocess 返回的检测结果列表
        class_names: 类别名称列表，为 None 时显示类别 ID

    Returns:
        绘制了检测结果的 BGR 图像（副本）
    """
    vis = img_bgr.copy()
    np.random.seed(42)
    color_table = [
        tuple(int(c) for c in np.random.randint(50, 220, 3))
        for _ in range(max(len(class_names) if class_names else 0, 80) + 1)
    ]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid   = det["class_id"]
        score = det["score"]
        color = color_table[cid % len(color_table)]
        label = class_names[cid] if (class_names and cid < len(class_names)) else f"cls{cid}"
        text  = f"{label} {score:.2f}"

        # 边框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 4, th + baseline)
        cv2.rectangle(vis, (x1, label_y - th - baseline), (x1 + tw, label_y), color, -1)
        cv2.putText(vis, text, (x1, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# ONNX 推理器
# ---------------------------------------------------------------------------

class RFDETROnnxInfer:
    """
    RF-DETR ONNX 推理封装。

    Args:
        model_path: ONNX 模型路径
        resolution: 模型输入分辨率（正方形边长，应与导出时一致）
        conf_threshold: 置信度阈值
        class_names: 类别名称列表
        providers: onnxruntime 执行提供者列表
    """

    def __init__(
        self,
        model_path: str,
        resolution: int = 448,
        conf_threshold: float = 0.5,
        class_names: list[str] | None = None,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请先安装 onnxruntime: pip install onnxruntime  或  pip install onnxruntime-gpu")

        _validate_resolution(resolution)
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.resolution    = resolution
        self.conf_threshold = conf_threshold
        self.class_names   = class_names

        # 检查输入输出节点名称
        self._input_name  = self.session.get_inputs()[0].name
        output_names      = [o.name for o in self.session.get_outputs()]
        # RF-DETR 通常输出 pred_logits 和 pred_boxes
        if len(output_names) >= 2:
            self._logits_name = output_names[0]
            self._boxes_name  = output_names[1]
        else:
            raise RuntimeError(f"ONNX 模型输出节点数量不足，期望至少 2 个，实际: {output_names}")

        print(f"[ONNX] 加载模型: {model_path}")
        print(f"[ONNX] 输入节点: {self._input_name}  输出节点: {self._logits_name}, {self._boxes_name}")
        print(f"[ONNX] 执行提供者: {self.session.get_providers()}")

    def predict(self, img_bgr: np.ndarray) -> list[dict]:
        """
        对单张图像进行推理。

        Args:
            img_bgr: OpenCV 读取的 BGR 图像

        Returns:
            检测结果列表，参见 postprocess 说明
        """
        blob, orig_size = preprocess(img_bgr, self.resolution)
        logits, boxes = self.session.run(
            [self._logits_name, self._boxes_name],
            {self._input_name: blob},
        )
        return postprocess(logits, boxes, orig_size, self.conf_threshold)

    def predict_and_draw(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[list[dict], np.ndarray]:
        """
        推理并返回带检测框的可视化图像。

        Returns:
            (detections, vis_img_bgr)
        """
        detections = self.predict(img_bgr)
        vis = draw_detections(img_bgr, detections, self.class_names)
        return detections, vis


# ---------------------------------------------------------------------------
# 导出功能
# ---------------------------------------------------------------------------

def export_onnx(
    weights: str,
    output: str,
    resolution: int = 448,
    opset: int = 17,
    simplify: bool = True,
) -> None:
    """
    将 RF-DETR PyTorch 权重导出为 ONNX 格式。

    Args:
        weights: 训练权重路径（.pth / .pt）
        output: 导出 ONNX 文件路径
        resolution: 输入分辨率
        opset: ONNX opset 版本
        simplify: 是否使用 onnx-simplifier 简化模型
    """
    import torch

    try:
        from rfdetr import RFDETRBase
    except ImportError:
        raise ImportError("请先安装 rfdetr: pip install rfdetr")

    _validate_resolution(resolution)

    print(f"[export] 加载权重: {weights}")
    model = RFDETRBase(pretrain_weights=weights)
    model.model.eval()

    dummy = torch.zeros(1, 3, resolution, resolution)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] 导出 ONNX → {output_path}  (opset={opset}, resolution={resolution})")
    torch.onnx.export(
        model.model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "image":      {0: "batch"},
            "pred_logits":{0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        do_constant_folding=True,
    )

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            model_onnx = onnx.load(str(output_path))
            model_simplified, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(model_simplified, str(output_path))
                print("[export] 模型简化成功")
            else:
                print("[export] 模型简化失败，使用原始 ONNX 文件")
        except ImportError:
            print("[export] 未安装 onnxsim，跳过简化（pip install onnxsim）")

    print(f"[export] 完成: {output_path}")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_class_names(labels_file: str) -> list[str]:
    """
    从文件加载类别名称。
    支持两种格式：
      - 每行一个类别名称的纯文本文件
      - COCO JSON 格式（包含 'categories' 键）
    """
    path = Path(labels_file)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cats = sorted(data["categories"], key=lambda c: c["id"])
        return [c["name"] for c in cats]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]


def infer_directory(
    infer_engine: RFDETROnnxInfer,
    source_dir: str,
    output_dir: str | None,
    save_json: bool = False,
) -> None:
    """批量推理目录下的所有图片。"""
    source = Path(source_dir)
    img_paths = [p for p in sorted(source.iterdir()) if p.suffix.lower() in _IMG_EXTS]
    if not img_paths:
        print(f"[infer] 目录 {source} 中未找到图片")
        return

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[infer] 跳过（无法读取）: {img_path}")
            continue

        detections, vis = infer_engine.predict_and_draw(img)
        print(f"[infer] {img_path.name}: {len(detections)} 个检测结果")
        for d in detections:
            cid = d["class_id"]
            name = infer_engine.class_names[cid] if (infer_engine.class_names and cid < len(infer_engine.class_names)) else f"cls{cid}"
            print(f"         {name}  score={d['score']:.3f}  bbox={[round(v) for v in d['bbox']]}")

        if output_dir:
            save_path = out / img_path.name
            cv2.imwrite(str(save_path), vis)

        all_results[img_path.name] = detections

    if save_json and output_dir:
        json_path = Path(output_dir) / "detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[infer] 检测结果已保存: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> None:
    export_onnx(
        weights=args.weights,
        output=args.output,
        resolution=args.resolution,
        opset=args.opset,
        simplify=not args.no_simplify,
    )


def cmd_infer(args: argparse.Namespace) -> None:
    class_names: list[str] | None = None
    if args.labels_file:
        class_names = load_class_names(args.labels_file)
        print(f"[infer] 类别 ({len(class_names)}): {class_names[:10]}{'...' if len(class_names) > 10 else ''}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if not args.cpu_only
        else ["CPUExecutionProvider"]
    )

    engine = RFDETROnnxInfer(
        model_path=args.model,
        resolution=args.resolution,
        conf_threshold=args.conf,
        class_names=class_names,
        providers=providers,
    )

    source = Path(args.source)

    if source.is_dir():
        infer_directory(engine, str(source), args.output_dir, save_json=args.save_json)
    else:
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {source}")

        detections, vis = engine.predict_and_draw(img)
        print(f"[infer] 检测到 {len(detections)} 个目标:")
        for d in detections:
            cid = d["class_id"]
            name = class_names[cid] if (class_names and cid < len(class_names)) else f"cls{cid}"
            print(f"  {name}  score={d['score']:.3f}  bbox={[round(v) for v in d['bbox']]}")

        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_path = out / source.name
            cv2.imwrite(str(save_path), vis)
            print(f"[infer] 结果已保存: {save_path}")
        else:
            cv2.imshow("RF-DETR ONNX Inference", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if args.save_json and args.output_dir:
            json_path = Path(args.output_dir) / (source.stem + "_detections.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detections, f, ensure_ascii=False, indent=2)
            print(f"[infer] JSON 已保存: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RF-DETR ONNX 推理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- export 子命令 ----
    p_export = sub.add_parser("export", help="导出 RF-DETR 模型为 ONNX 格式")
    p_export.add_argument("--weights",    required=True, help="RF-DETR 训练权重路径（.pth）")
    p_export.add_argument("--output",     default="rfdetr.onnx", help="ONNX 输出路径")
    p_export.add_argument("--resolution", type=int, default=448, help="输入分辨率，需被 32 整除（默认 448）")
    p_export.add_argument("--opset",      type=int, default=17,  help="ONNX opset 版本（默认 17）")
    p_export.add_argument("--no-simplify", action="store_true",  help="跳过 onnxsim 简化")

    # ---- infer 子命令 ----
    p_infer = sub.add_parser("infer", help="使用 ONNX 模型推理")
    p_infer.add_argument("--model",       required=True, help="ONNX 模型路径")
    p_infer.add_argument("--source",      required=True, help="输入图片路径或目录")
    p_infer.add_argument("--conf",        type=float, default=0.5, help="置信度阈值（默认 0.5）")
    p_infer.add_argument("--resolution",  type=int, default=448,   help="模型输入分辨率，需被 32 整除（默认 448）")
    p_infer.add_argument("--labels-file", default=None,
                         help="类别名称文件（每行一个类别名，或 COCO JSON）")
    p_infer.add_argument("--output-dir",  default=None, help="可视化结果保存目录")
    p_infer.add_argument("--save-json",   action="store_true",     help="同时保存 JSON 检测结果")
    p_infer.add_argument("--cpu-only",    action="store_true",     help="仅使用 CPU 推理")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        cmd_export(args)
    elif args.command == "infer":
        cmd_infer(args)


if __name__ == "__main__":
    main()
