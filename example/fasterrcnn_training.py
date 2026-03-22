"""
Faster R-CNN 训练集成 Demo
==========================
将 data_enhance 的自定义增强注入到 torchvision Faster R-CNN 训练流程中。

集成方式：自定义 Dataset，在 __getitem__ 中应用 Albumentations 增强。

支持数据集格式：
  - COCO JSON（推荐）
  - Pascal VOC XML

用法:
    python demo/fasterrcnn_training.py \
        --data demo/configs/fasterrcnn_dataset.yaml \
        --epochs 20 \
        --bg-dir path/to/backgrounds

数据集配置文件见 demo/configs/fasterrcnn_dataset.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Albumentations → Faster R-CNN 格式转换
# ---------------------------------------------------------------------------
# Albumentations 内部使用 pascal_voc 格式: [x_min, y_min, x_max, y_max]（归一化 0-1）
# Faster R-CNN 需要: boxes [x1, y1, x2, y2]（绝对像素，float32）

def norm_to_abs(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """归一化 pascal_voc → 绝对像素坐标。"""
    x1, y1, x2, y2 = bbox
    return [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]


def abs_to_norm(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """绝对像素坐标 → 归一化 pascal_voc。"""
    x1, y1, x2, y2 = bbox
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


# ---------------------------------------------------------------------------
# 数据集：COCO JSON 格式
# ---------------------------------------------------------------------------

class CocoDetectionDataset(torch.utils.data.Dataset):
    """
    读取 COCO JSON 格式标注，应用 Albumentations 增强，
    返回 torchvision Faster R-CNN 所需格式。

    数据集目录结构:
        images/          ← 图片文件
        annotations.json ← COCO JSON 标注

    COCO JSON 格式:
        {
          "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                           "bbox": [x, y, w, h], "area": ..., "iscrowd": 0}],
          "categories": [{"id": 1, "name": "cat"}]
        }
    """

    def __init__(
        self,
        img_dir: str | Path,
        annotation_file: str | Path,
        transforms: A.Compose | None = None,
    ):
        import json

        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(annotation_file) as f:
            coco = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

        # image_id → file info
        self._images = {img["id"]: img for img in coco["images"]}

        # image_id → list of annotations
        self._annotations: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            self._annotations.setdefault(img_id, []).append(ann)

        # 有效的 image id 列表（只保留 COCO 中存在的图片）
        self._ids = [
            img_id for img_id in self._images
            if (self.img_dir / self._images[img_id]["file_name"]).exists()
        ]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        img_id = self._ids[idx]
        img_info = self._images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        # 读取图片
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 解析标注
        anns = self._annotations.get(img_id, [])
        bboxes_abs = []   # pascal_voc: [x1, y1, x2, y2]（绝对像素）
        class_labels = []

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
            if x2 > x1 and y2 > y1:
                bboxes_abs.append([x1, y1, x2, y2])
                class_labels.append(ann["category_id"])

        # 应用 Albumentations 增强
        if self.transforms and bboxes_abs:
            # 转成归一化坐标传给 Albumentations
            bboxes_norm = [abs_to_norm(b, w, h) for b in bboxes_abs]
            result = self.transforms(
                image=image,
                bboxes=bboxes_norm,
                class_labels=class_labels,
            )
            image = result["image"]
            new_h, new_w = image.shape[:2]
            # 转回绝对坐标
            bboxes_abs = [norm_to_abs(list(b), new_w, new_h) for b in result["bboxes"]]
            class_labels = list(result["class_labels"])
        elif self.transforms:
            result = self.transforms(image=image, bboxes=[], class_labels=[])
            image = result["image"]

        # 转为 Faster R-CNN 期望的 tensor 格式
        image_tensor = torch.from_numpy(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        if bboxes_abs:
            boxes = torch.tensor(bboxes_abs, dtype=torch.float32)
            labels = torch.tensor(class_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

        return image_tensor, target


# ---------------------------------------------------------------------------
# 数据集：Pascal VOC XML 格式
# ---------------------------------------------------------------------------

class VOCDetectionDataset(torch.utils.data.Dataset):
    """
    读取 Pascal VOC XML 格式标注，应用 Albumentations 增强。

    数据集目录结构:
        JPEGImages/      ← 图片
        Annotations/     ← XML 标注（与图片同名）
        ImageSets/Main/train.txt  ← 训练集文件列表

    XML 标注格式（VOC）:
        <annotation>
          <object>
            <name>cat</name>
            <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>200</ymax></bndbox>
          </object>
        </annotation>
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        class_names: list[str],
        transforms: A.Compose | None = None,
    ):
        import xml.etree.ElementTree as ET

        self.root = Path(root)
        self.transforms = transforms
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # 0 = background
        self._ET = ET

        # 读取文件列表
        with open(split_file) as f:
            self._stems = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int):
        stem = self._stems[idx]
        img_path = self.img_dir / f"{stem}.jpg"
        ann_path = self.ann_dir / f"{stem}.xml"

        # 尝试多种图片扩展名
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.root / "JPEGImages" / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 解析 XML
        ann_path = self.root / "Annotations" / f"{stem}.xml"
        bboxes_abs = []
        class_labels = []

        if ann_path.exists():
            tree = self._ET.parse(str(ann_path))
            root_xml = tree.getroot()
            for obj in root_xml.findall("object"):
                name = obj.find("name").text.strip()
                if name not in self.class_to_idx:
                    continue
                bb = obj.find("bndbox")
                x1 = max(0.0, float(bb.find("xmin").text))
                y1 = max(0.0, float(bb.find("ymin").text))
                x2 = min(float(w), float(bb.find("xmax").text))
                y2 = min(float(h), float(bb.find("ymax").text))
                if x2 > x1 and y2 > y1:
                    bboxes_abs.append([x1, y1, x2, y2])
                    class_labels.append(self.class_to_idx[name])

        # 应用增强
        if self.transforms and bboxes_abs:
            bboxes_norm = [abs_to_norm(b, w, h) for b in bboxes_abs]
            result = self.transforms(
                image=image, bboxes=bboxes_norm, class_labels=class_labels
            )
            image = result["image"]
            new_h, new_w = image.shape[:2]
            bboxes_abs = [norm_to_abs(list(b), new_w, new_h) for b in result["bboxes"]]
            class_labels = list(result["class_labels"])
        elif self.transforms:
            result = self.transforms(image=image, bboxes=[], class_labels=[])
            image = result["image"]

        image_tensor = torch.from_numpy(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        if bboxes_abs:
            boxes = torch.tensor(bboxes_abs, dtype=torch.float32)
            labels = torch.tensor(class_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return image_tensor, target


# ---------------------------------------------------------------------------
# 增强 pipeline 构建
# ---------------------------------------------------------------------------

def build_train_transforms(
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_copy_paste: bool = True,
    min_visibility: float = 0.3,
) -> A.Compose:
    """
    构建 Faster R-CNN 训练用 Albumentations pipeline。

    注意: 使用 pascal_voc 格式（归一化）作为 bbox 格式，
    这样与 Faster R-CNN 的 [x1, y1, x2, y2] 转换最直接。
    """
    from data_enhance import BackgroundReplace, BboxRelocate, CrossImageCopyPaste

    transforms = []

    if use_bg_replace and bg_dir:
        transforms.append(BackgroundReplace(bg_dir=bg_dir, blend_border=8, p=0.3))

    if use_bbox_relocate and bg_dir:
        transforms.append(BboxRelocate(bg_dir=bg_dir, allow_overlap=False, p=0.2))

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

    # 通用几何 & 颜色增强
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.RandomScale(scale_limit=0.2, p=0.3),
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="albumentations",  # 归一化 [x_min, y_min, x_max, y_max]
            label_fields=["class_labels"],
            min_visibility=min_visibility,
        ),
    )


def build_val_transforms() -> A.Compose:
    """验证集不做数据增强，仅做格式转换。"""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
        ),
    )


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

def build_faster_rcnn(num_classes: int, pretrained: bool = True):
    """
    构建 Faster R-CNN 模型（ResNet-50 + FPN backbone）。

    Args:
        num_classes: 类别数（含背景类，即实际类别数 + 1）
        pretrained: 是否使用 COCO 预训练权重

    Returns:
        torchvision Faster R-CNN 模型
    """
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError:
        raise ImportError("请先安装 torchvision: pip install torchvision")

    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # 替换分类头为自定义类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = len(data_loader)

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  Epoch {epoch} [{i+1}/{n_batches}]  loss: {losses.item():.4f}  "
                  f"(cls: {loss_dict.get('loss_classifier', 0):.3f}, "
                  f"box: {loss_dict.get('loss_box_reg', 0):.3f}, "
                  f"rpn_cls: {loss_dict.get('loss_objectness', 0):.3f}, "
                  f"rpn_box: {loss_dict.get('loss_rpn_box_reg', 0):.3f})")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data_loader, device):
    """简单评估：统计平均推理 loss（需要标注）。"""
    model.train()  # 保持 train 模式以获取 loss
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    return total_loss / max(1, len(data_loader))


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(config: dict):
    """
    从配置字典启动 Faster R-CNN 训练。

    配置键见 demo/configs/fasterrcnn_dataset.yaml。
    """
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 构建增强
    aug_cfg = config.get("augmentation", {})
    train_transforms = build_train_transforms(
        bg_dir=aug_cfg.get("bg_dir"),
        image_dir=aug_cfg.get("image_dir"),
        label_dir=aug_cfg.get("label_dir"),
        use_bg_replace=aug_cfg.get("use_bg_replace", True),
        use_bbox_relocate=aug_cfg.get("use_bbox_relocate", False),
        use_copy_paste=aug_cfg.get("use_copy_paste", True),
        min_visibility=aug_cfg.get("min_visibility", 0.3),
    )
    val_transforms = build_val_transforms()

    # 构建数据集
    dataset_cfg = config["dataset"]
    fmt = dataset_cfg.get("format", "coco").lower()

    if fmt == "coco":
        train_dataset = CocoDetectionDataset(
            img_dir=dataset_cfg["train_img_dir"],
            annotation_file=dataset_cfg["train_ann"],
            transforms=train_transforms,
        )
        val_dataset = CocoDetectionDataset(
            img_dir=dataset_cfg["val_img_dir"],
            annotation_file=dataset_cfg["val_ann"],
            transforms=val_transforms,
        )
        num_classes = len(train_dataset.categories) + 1  # +1 for background
        print(f"类别数（含背景）: {num_classes}")
        print(f"类别: {train_dataset.categories}")
    elif fmt == "voc":
        class_names = dataset_cfg["class_names"]
        train_dataset = VOCDetectionDataset(
            root=dataset_cfg["root"],
            split_file=dataset_cfg["train_split"],
            class_names=class_names,
            transforms=train_transforms,
        )
        val_dataset = VOCDetectionDataset(
            root=dataset_cfg["root"],
            split_file=dataset_cfg["val_split"],
            class_names=class_names,
            transforms=val_transforms,
        )
        num_classes = len(class_names) + 1
    else:
        raise ValueError(f"不支持的数据集格式: {fmt}，请使用 'coco' 或 'voc'")

    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"训练样本: {len(train_dataset)}  验证样本: {len(val_dataset)}")

    # 构建模型
    model_cfg = config.get("model", {})
    model = build_faster_rcnn(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
    )
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=train_cfg.get("lr", 0.005),
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg.get("lr_step_size", 5),
        gamma=train_cfg.get("lr_gamma", 0.1),
    )

    epochs = train_cfg.get("epochs", 20)
    output_dir = Path(config.get("output_dir", "runs/fasterrcnn"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 开始训练 Faster R-CNN，共 {epochs} 轮 ===\n")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        lr_scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs}  "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}  "
              f"time: {elapsed:.1f}s")

        # 保存最优模型
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_classes": num_classes,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  → 保存最优模型 (val_loss: {best_val_loss:.4f})")

    print(f"\n训练完成，模型保存至: {output_dir}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="使用 data_enhance 增强训练 Faster R-CNN")
    parser.add_argument("--data", required=True,
                        help="数据集配置文件路径（YAML），见 demo/configs/fasterrcnn_dataset.yaml")
    # 允许通过 CLI 覆盖部分配置
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（覆盖配置文件）")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size（覆盖配置文件）")
    parser.add_argument("--bg-dir", default=None, help="背景图片目录（覆盖配置文件）")
    parser.add_argument("--device", default=None, help="训练设备（覆盖配置文件）")
    parser.add_argument("--output-dir", default=None, help="输出目录（覆盖配置文件）")

    args = parser.parse_args()

    with open(args.data) as f:
        config = yaml.safe_load(f)

    # CLI 参数覆盖配置文件
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.bg_dir is not None:
        config.setdefault("augmentation", {})["bg_dir"] = args.bg_dir
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    train(config)


if __name__ == "__main__":
    main()
