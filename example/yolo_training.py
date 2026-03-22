"""
YOLO 训练集成 Demo
==================
将 data_enhance 的自定义增强注入到 Ultralytics YOLO 训练流程中。

集成方式：子类化 DetectionTrainer + YOLODataset（最灵活，推荐生产使用）

用法:
    python demo/yolo_training.py \
        --data path/to/dataset.yaml \
        --model yolo11n.pt \
        --epochs 100 \
        --bg-dir path/to/backgrounds \
        --imgsz 640

数据集 YAML 示例 (dataset.yaml):
    path: /path/to/dataset
    train: images/train
    val: images/val
    names:
        0: cat
        1: dog

注意: 如果启用 --use-mosaic，需要在 dataset.yaml 中禁用内置 mosaic:
    mosaic: 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A


def build_custom_transforms(
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_mosaic: bool = False,
    use_copy_paste: bool = True,
    imgsz: int = 640,
) -> list:
    """
    根据配置构建 data_enhance transform 列表。

    Args:
        bg_dir: 背景图片目录（BackgroundReplace / BboxRelocate 需要）
        image_dir: 训练图片目录（Mosaic / CrossImageCopyPaste 需要）
        label_dir: YOLO 标注目录（Mosaic / CrossImageCopyPaste 需要）
        use_bg_replace: 是否启用背景替换
        use_bbox_relocate: 是否启用 bbox 随机重定位
        use_mosaic: 是否启用自定义 Mosaic（需同时禁用 YOLO 内置 mosaic）
        use_copy_paste: 是否启用跨图复制粘贴
        imgsz: 输入图片尺寸

    Returns:
        Albumentations transform 对象列表
    """
    from data_enhance import BackgroundReplace, BboxRelocate, CrossImageCopyPaste, Mosaic

    transforms = []

    if use_bg_replace and bg_dir:
        transforms.append(
            BackgroundReplace(bg_dir=bg_dir, blend_border=8, p=0.3)
        )

    if use_bbox_relocate and bg_dir:
        transforms.append(
            BboxRelocate(bg_dir=bg_dir, allow_overlap=False, max_attempts=50, p=0.2)
        )

    if use_mosaic and image_dir and label_dir:
        transforms.append(
            Mosaic(
                image_dir=image_dir,
                label_dir=label_dir,
                img_size=imgsz,
                center_range=(0.25, 0.75),
                p=0.5,
            )
        )

    if use_copy_paste and image_dir and label_dir:
        transforms.append(
            CrossImageCopyPaste(
                image_dir=image_dir,
                label_dir=label_dir,
                max_paste=3,
                scale_range=(0.5, 1.5),
                allow_overlap=False,
                p=0.4,
            )
        )

    return transforms


class CustomAlbumentations:
    """
    扩展 Ultralytics 内置的 Albumentations 包装器，追加 data_enhance transforms。

    在原有的 Ultralytics augmentation pipeline 末尾注入自定义增强，
    保留原有增强（HSV、翻转等）的同时叠加背景替换、复制粘贴等。
    """

    def __init__(
        self,
        bg_dir: str | None = None,
        image_dir: str | None = None,
        label_dir: str | None = None,
        use_bg_replace: bool = True,
        use_bbox_relocate: bool = False,
        use_mosaic: bool = False,
        use_copy_paste: bool = True,
        imgsz: int = 640,
    ):
        try:
            from ultralytics.data.augment import Albumentations
        except ImportError:
            raise ImportError("请先安装 ultralytics: pip install ultralytics")

        # 初始化 Ultralytics 原有的 Albumentations wrapper
        base = Albumentations(p=1.0)

        extra = build_custom_transforms(
            bg_dir=bg_dir,
            image_dir=image_dir,
            label_dir=label_dir,
            use_bg_replace=use_bg_replace,
            use_bbox_relocate=use_bbox_relocate,
            use_mosaic=use_mosaic,
            use_copy_paste=use_copy_paste,
            imgsz=imgsz,
        )

        if extra and base.transform:
            # 将自定义 transforms 追加到原有 pipeline 中
            self.transform = A.Compose(
                list(base.transform.transforms) + extra,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.3,
                ),
            )
        elif extra:
            # 原有 pipeline 为空（未安装 albumentations 依赖），仅用自定义 transforms
            self.transform = A.Compose(
                extra,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.3,
                ),
            )
        else:
            self.transform = base.transform

        # 保留原有 wrapper 的其他属性（用于兼容 Ultralytics 调用）
        self._base = base

    def __call__(self, labels):
        """与 Ultralytics Albumentations.__call__ 接口兼容。"""
        if self.transform is None:
            return labels

        image = labels["img"]
        bboxes = labels.get("bboxes", [])
        cls = labels.get("cls", [])

        # Ultralytics 内部 bboxes 格式: (N, 4) numpy array, YOLO 格式
        # class_labels: (N,) numpy array 整数
        import numpy as np
        if len(bboxes) == 0:
            return labels

        bboxes_list = bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
        cls_list = cls.flatten().tolist() if hasattr(cls, "tolist") else list(cls)

        result = self.transform(
            image=image,
            bboxes=bboxes_list,
            class_labels=cls_list,
        )

        labels["img"] = result["image"]
        out_bboxes = result["bboxes"]
        out_cls = result["class_labels"]

        if out_bboxes:
            labels["bboxes"] = np.array(out_bboxes, dtype=np.float32)
            labels["cls"] = np.array(out_cls, dtype=np.float32).reshape(-1, 1)
        else:
            labels["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            labels["cls"] = np.zeros((0, 1), dtype=np.float32)

        return labels


def build_custom_trainer(
    bg_dir: str | None = None,
    image_dir: str | None = None,
    label_dir: str | None = None,
    use_bg_replace: bool = True,
    use_bbox_relocate: bool = False,
    use_mosaic: bool = False,
    use_copy_paste: bool = True,
):
    """
    工厂函数：返回配置好的 CustomTrainer 类。

    通过工厂函数动态绑定增强参数，避免 Ultralytics 内部调用
    无参数构造函数时的兼容性问题。
    """
    try:
        from ultralytics.data.augment import Albumentations
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError:
        raise ImportError("请先安装 ultralytics: pip install ultralytics")

    _bg_dir = bg_dir
    _image_dir = image_dir
    _label_dir = label_dir
    _use_bg_replace = use_bg_replace
    _use_bbox_relocate = use_bbox_relocate
    _use_mosaic = use_mosaic
    _use_copy_paste = use_copy_paste

    class _CustomDataset(YOLODataset):
        """注入自定义 Albumentations transforms 的 YOLODataset 子类。"""

        def build_transforms(self, hyp=None):
            transforms = super().build_transforms(hyp)
            if not self.augment:
                return transforms
            # 找到并替换 Albumentations 实例
            for i, t in enumerate(transforms.transforms):
                if isinstance(t, Albumentations):
                    transforms.transforms[i] = CustomAlbumentations(
                        bg_dir=_bg_dir,
                        image_dir=_image_dir or str(Path(self.img_path).parent),
                        label_dir=_label_dir,
                        use_bg_replace=_use_bg_replace,
                        use_bbox_relocate=_use_bbox_relocate,
                        use_mosaic=_use_mosaic,
                        use_copy_paste=_use_copy_paste,
                        imgsz=self.imgsz,
                    )
                    break
            return transforms

    class _CustomTrainer(DetectionTrainer):
        """使用自定义数据集的 DetectionTrainer 子类。"""

        def build_dataset(self, img_path, mode="train", batch=None):
            from ultralytics.data.utils import check_det_dataset
            from ultralytics.utils import DEFAULT_CFG

            gs = max(int(self.model.stride.max() if self.model else 0), 32)
            return _CustomDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect if mode == "val" else False,
                cache=self.args.cache or False,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0 if mode == "train" else 0.5,
                prefix=f"{mode}: ",
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )

    return _CustomTrainer


def main():
    parser = argparse.ArgumentParser(
        description="使用 data_enhance 自定义增强训练 Ultralytics YOLO"
    )
    # 必填参数
    parser.add_argument("--data", required=True, help="数据集 YAML 路径")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型或配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", default="", help="训练设备，如 0 / 0,1 / cpu")

    # 增强相关
    parser.add_argument("--bg-dir", default=None, help="背景图片目录（BackgroundReplace/BboxRelocate 需要）")
    parser.add_argument("--image-dir", default=None, help="训练图片目录（Mosaic/CopyPaste 需要，默认从 data yaml 推断）")
    parser.add_argument("--label-dir", default=None, help="YOLO 标注目录（Mosaic/CopyPaste 需要）")
    parser.add_argument("--use-bg-replace", action="store_true", default=True, help="启用背景替换")
    parser.add_argument("--no-bg-replace", dest="use_bg_replace", action="store_false")
    parser.add_argument("--use-bbox-relocate", action="store_true", default=False, help="启用 bbox 重定位")
    parser.add_argument("--use-mosaic", action="store_true", default=False,
                        help="启用自定义 Mosaic（需在 dataset.yaml 中设置 mosaic: 0.0）")
    parser.add_argument("--use-copy-paste", action="store_true", default=True, help="启用跨图复制粘贴")
    parser.add_argument("--no-copy-paste", dest="use_copy_paste", action="store_false")

    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics: pip install ultralytics")

    print("=== data_enhance YOLO 训练 Demo ===")
    print(f"模型:        {args.model}")
    print(f"数据集:      {args.data}")
    print(f"训练轮数:    {args.epochs}")
    print(f"图片尺寸:    {args.imgsz}")
    print(f"背景替换:    {'是' if args.use_bg_replace and args.bg_dir else '否'}")
    print(f"Bbox重定位:  {'是' if args.use_bbox_relocate and args.bg_dir else '否'}")
    print(f"自定义Mosaic:{'是' if args.use_mosaic and args.image_dir else '否'}")
    print(f"跨图复制粘贴:{'是' if args.use_copy_paste and args.image_dir else '否'}")
    print()

    if args.use_mosaic:
        print("⚠️  已启用自定义 Mosaic，请确认 dataset.yaml 中已设置 mosaic: 0.0")

    CustomTrainer = build_custom_trainer(
        bg_dir=args.bg_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        use_bg_replace=args.use_bg_replace,
        use_bbox_relocate=args.use_bbox_relocate,
        use_mosaic=args.use_mosaic,
        use_copy_paste=args.use_copy_paste,
    )

    model = YOLO(args.model)
    model.train(
        trainer=CustomTrainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
    )


if __name__ == "__main__":
    main()
