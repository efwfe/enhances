# Demo：接入 YOLO、Faster R-CNN 和 RF-DETR 训练流程

将 `data_enhance` 库的自定义增强注入到主流目标检测训练框架中。

## 文件说明

| 文件 | 说明 |
|---|---|
| `yolo_training.py` | Ultralytics YOLO 训练集成（子类化 Trainer + Dataset） |
| `fasterrcnn_training.py` | torchvision Faster R-CNN 训练集成（自定义 Dataset） |
| `rfdetr_training.py` | RF-DETR 训练集成（Dataset Wrapper 在线增强） |
| `visualize.py` | 增强效果可视化工具 |
| `configs/yolo_dataset.yaml` | YOLO 数据集配置模板 |
| `configs/fasterrcnn_dataset.yaml` | Faster R-CNN 数据集配置模板（COCO/VOC） |

---

## 环境准备

```bash
# 安装核心库
pip install -e ..

# YOLO demo 依赖
pip install ultralytics

# RF-DETR demo 依赖
pip install rfdetr

# Faster R-CNN demo 依赖
pip install torch torchvision

# 可视化依赖
pip install matplotlib
```

---

## 数据集格式

### YOLO 格式（yolo_training.py）

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt   # 每行: <class_id> <xc> <yc> <w> <h>
│   │   └── ...
│   └── val/
│       └── ...
└── dataset.yaml
```

`dataset.yaml` 示例：
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog
```

> 如果启用 `--use-mosaic`，需额外在 yaml 中添加 `mosaic: 0.0` 禁用内置 mosaic。

### COCO 格式（rfdetr_training.py）

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── img001.jpg
│   └── ...
└── valid/
    ├── _annotations.coco.json
    └── ...
```

---

## 背景图片目录

`BackgroundReplace` 和 `BboxRelocate` 需要一个背景图片目录：

```
backgrounds/
├── bg001.jpg
├── bg002.png
└── ...
```

---

## 使用方法

### 1. 可视化增强效果（建议先运行，验证配置）

```bash
# 单张图片：展示原图 vs 3 次增强结果
python demo/visualize.py \
    --image dataset/images/train/sample.jpg \
    --label dataset/labels/train/sample.txt \
    --bg-dir backgrounds/ \
    --dataset-image-dir dataset/images/train \
    --dataset-label-dir dataset/labels/train \
    --n-augments 3 \
    --output vis_result.png

# 批量可视化（从数据集随机采样 8 张）
python demo/visualize.py \
    --image-dir dataset/images/train \
    --label-dir dataset/labels/train \
    --bg-dir backgrounds/ \
    --n-samples 8 \
    --output-dir vis_output/

# 带类别名称
python demo/visualize.py \
    --image sample.jpg \
    --bg-dir backgrounds/ \
    --class-names "0:cat,1:dog"
```

### 2. YOLO 训练

```bash
# 基础训练（启用背景替换 + 跨图复制粘贴）
python demo/yolo_training.py \
    --data dataset/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --bg-dir backgrounds/ \
    --image-dir dataset/images/train \
    --label-dir dataset/labels/train

# 启用所有增强（含自定义 Mosaic，需在 yaml 中禁用内置 mosaic）
python demo/yolo_training.py \
    --data dataset/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --bg-dir backgrounds/ \
    --image-dir dataset/images/train \
    --label-dir dataset/labels/train \
    --use-mosaic \
    --use-bbox-relocate \
    --imgsz 640 \
    --batch 16

# 快速验证（2 轮）
python demo/yolo_training.py \
    --data coco128.yaml \
    --model yolo11n.pt \
    --epochs 2 \
    --bg-dir backgrounds/
```

### 3. Faster R-CNN 训练

```bash
# 基础训练（COCO 格式数据集）
python demo/fasterrcnn_training.py \
    --data demo/configs/fasterrcnn_dataset.yaml \
    --bg-dir backgrounds/

# 覆盖配置文件中的部分参数
python demo/fasterrcnn_training.py \
    --data demo/configs/fasterrcnn_dataset.yaml \
    --epochs 30 \
    --batch-size 8 \
    --device cuda \
    --output-dir runs/my_experiment

# CPU 快速验证（2 轮）
python demo/fasterrcnn_training.py \
    --data demo/configs/fasterrcnn_dataset.yaml \
    --epochs 2 \
    --batch-size 2 \
    --device cpu
```

**配置文件路径**: `demo/configs/fasterrcnn_dataset.yaml`
- 支持 COCO JSON 和 Pascal VOC XML 两种格式
- 所有增强开关和超参数均在配置文件中设置
- CLI 参数可覆盖配置文件中对应的值

### 4. RF-DETR 训练

```bash
# 基础训练
python demo/rfdetr_training.py \
    --dataset-dir dataset/ \
    --epochs 50 \
    --batch-size 4 \
    --bg-dir backgrounds/

# 完整配置
python demo/rfdetr_training.py \
    --dataset-dir dataset/ \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --resolution 560 \
    --bg-dir backgrounds/ \
    --image-dir dataset/train \
    --label-dir dataset/train/labels \
    --use-copy-paste \
    --output-dir runs/rfdetr_custom

# CPU 训练（测试用）
python demo/rfdetr_training.py \
    --dataset-dir dataset/ \
    --epochs 2 \
    --batch-size 2 \
    --device cpu
```

---

## 参数说明

### 通用增强参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--bg-dir` | 背景图片目录（BackgroundReplace/BboxRelocate 需要） | `None` |
| `--image-dir` | 训练图片目录（Mosaic/CopyPaste 需要） | `None` |
| `--label-dir` | YOLO 标注目录（Mosaic/CopyPaste 需要） | `None` |
| `--use-bg-replace` / `--no-bg-replace` | 背景替换开关 | 开 |
| `--use-bbox-relocate` | bbox 随机重定位 | 关 |
| `--use-mosaic` | 自定义 Mosaic（YOLO 专用） | 关 |
| `--use-copy-paste` / `--no-copy-paste` | 跨图复制粘贴开关 | 开 |

### 各增强说明

| 增强 | 作用 | 适用场景 |
|---|---|---|
| `BackgroundReplace` | 替换 bbox 外区域为随机背景 | 减少背景过拟合 |
| `BboxRelocate` | 将目标裁剪后贴到新背景的随机位置 | 增加目标位置多样性 |
| `Mosaic` | 4 图拼接，丰富尺度和上下文 | 小目标检测 |
| `CrossImageCopyPaste` | 将其他图片的目标粘贴到当前图片 | 增加类别多样性、小目标 |

---

## 代码集成方式说明

### YOLO 集成架构

```
YOLO.train()
  └── CustomTrainer.build_dataset()
        └── CustomDataset.build_transforms()
              └── CustomAlbumentations（替换原有 Albumentations 实例）
                    ├── 原有 Ultralytics 增强（HSV、翻转等）
                    └── data_enhance 增强（追加）
```

### RF-DETR 集成架构

```
RFDETRBase.train()
  └── on_train_start 回调
        └── 找到内部 train_dataset
              └── AugmentedCocoDataset（Dataset Wrapper）
                    ├── 原始 __getitem__（PIL image + COCO annotations）
                    ├── COCO bbox → YOLO 归一化格式
                    ├── Albumentations 增强
                    └── YOLO 格式 → COCO bbox 还原
```
