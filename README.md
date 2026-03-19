# data-enhance

Custom [Albumentations](https://albumentations.ai/) transforms for object-detection data augmentation. Designed as drop-in additions to any `A.Compose` pipeline that uses bounding boxes.

## Transforms

| Transform | What it does |
|---|---|
| `BackgroundReplace` | Replaces pixels **outside** all bounding boxes with a random background image |
| `BboxRelocate` | Cuts every bbox crop and pastes it at a random position on a new background |
| `Mosaic` | Tiles 4 images into a 2×2 grid; remaps all bboxes to canvas coordinates |
| `CrossImageCopyPaste` | Pastes object crops from other dataset images onto the current image |

## Installation

```bash
pip install -e .
# or with uv
uv pip install -e .
```

**Dependencies:** `albumentations>=1.3.0`, `opencv-python>=4.5.0`, `numpy>=1.21.0`

## Usage

### BackgroundReplace

Replaces the background (area outside bboxes) with a random image from a directory. Bboxes and labels are unchanged.

```python
import cv2
import albumentations as A
from data_enhance import BackgroundReplace

transform = A.Compose(
    [
        BackgroundReplace(
            bg_dir="backgrounds",   # directory of background images
            blend_border=8,         # px of soft feathering at bbox edges (0 = hard cut)
            p=0.6,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

img = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)
result = transform(image=img, bboxes=[[0.5, 0.4, 0.3, 0.5]], class_labels=["cat"])
```

### BboxRelocate

Cuts each bbox crop from the source image and pastes it at a random position on a new background. Output bboxes reflect the new positions.

```python
from data_enhance import BboxRelocate

transform = A.Compose(
    [
        BboxRelocate(
            bg_dir="backgrounds",
            allow_overlap=False,  # prevent boxes from overlapping each other
            max_attempts=50,      # placement attempts before forced placement
            p=0.4,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
```

### Mosaic

Combines the current image with 3 randomly sampled images from your dataset into a 2×2 grid. All bboxes from all 4 images are remapped. Requires YOLO `.txt` label files alongside the images.

```python
from data_enhance import Mosaic

transform = A.Compose(
    [
        Mosaic(
            image_dir="dataset/images",
            label_dir="dataset/labels",   # YOLO .txt files with same stem as images
            img_size=640,                 # output canvas size (square)
            center_range=(0.25, 0.75),    # fractional range for the mosaic split point
            p=0.5,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
```

**Label file format** (one object per line):
```
<class_id> <x_center> <y_center> <width> <height>
```

### CrossImageCopyPaste

Pastes object crops sampled from other dataset images onto the current image, appending bboxes for the pasted objects. Unlike `BboxRelocate`, crops come from entirely different images, increasing diversity.

```python
from data_enhance import CrossImageCopyPaste

transform = A.Compose(
    [
        CrossImageCopyPaste(
            image_dir="dataset/images",
            label_dir="dataset/labels",
            max_paste=3,              # max crops to paste per call
            scale_range=(0.5, 1.5),   # random scale applied to each crop
            allow_overlap=False,      # avoid overlapping existing boxes
            max_attempts=50,
            p=0.5,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
```

> **Note:** Pasted objects carry integer `class_id` labels from the YOLO file. If your pipeline uses string labels for the current image, the output will contain mixed types. Keep labels as integers throughout to avoid this.

### Combining transforms

All transforms are standard Albumentations `DualTransform` subclasses and compose freely:

```python
import albumentations as A
from data_enhance import BackgroundReplace, BboxRelocate, CrossImageCopyPaste

transform = A.Compose(
    [
        BackgroundReplace(bg_dir="backgrounds", blend_border=8, p=0.5),
        BboxRelocate(bg_dir="backgrounds", allow_overlap=False, p=0.3),
        CrossImageCopyPaste(
            image_dir="dataset/images",
            label_dir="dataset/labels",
            max_paste=2,
            p=0.4,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)

result = transform(image=img, bboxes=yolo_bboxes, class_labels=class_labels)
out_img    = result["image"]
out_bboxes = result["bboxes"]
out_labels = result["class_labels"]
```

See `example_usage.py` for a runnable end-to-end example.

## Supported bbox formats

All transforms accept any bbox format supported by Albumentations (`yolo`, `pascal_voc`, `coco`, `albumentations`). Pass the format via `A.BboxParams`.
