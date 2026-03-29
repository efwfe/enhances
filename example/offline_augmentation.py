"""
Offline generative augmentation example.

Assumes:
  - ComfyUI is running locally on port 8188
  - You have exported two workflow JSON templates (API format) from ComfyUI:
      workflows/multiview.json    — multi-view generation (e.g. Zero123++)
      workflows/inpainting.json   — inpainting (e.g. SD / SDXL inpainting)
  - Source YOLO dataset:
      dataset/images/   *.jpg
      dataset/labels/   *.txt
  - Background images (optional):
      backgrounds/      *.jpg

Workflow template format
------------------------
Export your workflow from ComfyUI via Save → Save (API format).
Then replace the dynamic values with __PLACEHOLDER__ markers:

  multiview.json
    - In the LoadImage node: set "image" to "__INPUT_IMAGE__"
    - In the Zero123++ node: set "azimuth" to "__AZIMUTH__", "elevation" to "__ELEVATION__"

  inpainting.json
    - In the LoadImage node for the scene: set "image" to "__INPUT_IMAGE__"
    - In the LoadImage node for the mask:  set "image" to "__MASK_IMAGE__"
    - In the CLIPTextEncode (positive) node: set "text" to "__PROMPT__"

Output
------
Augmented images and labels are written to:
    output/multiview/images/   output/multiview/labels/
    output/scale/images/       output/scale/labels/
"""

from pathlib import Path

from data_enhance import ComfyUIClient, MultiViewAugmentor, BboxScaleAugmentor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188

DATASET_IMAGES = Path("dataset/images")
DATASET_LABELS = Path("dataset/labels")
BACKGROUNDS    = Path("backgrounds")         # optional; set to None to skip

MULTIVIEW_WORKFLOW  = Path("workflows/multiview.json")
INPAINTING_WORKFLOW = Path("workflows/inpainting.json")

OUTPUT_MV_IMAGES    = Path("output/multiview/images")
OUTPUT_MV_LABELS    = Path("output/multiview/labels")
OUTPUT_SCALE_IMAGES = Path("output/scale/images")
OUTPUT_SCALE_LABELS = Path("output/scale/labels")

# ---------------------------------------------------------------------------
# Setup client
# ---------------------------------------------------------------------------

client = ComfyUIClient(host=COMFYUI_HOST, port=COMFYUI_PORT, timeout=300.0)

# ---------------------------------------------------------------------------
# Multi-view augmentation
# ---------------------------------------------------------------------------
# Generates views at azimuths [-90, -45, 45, 90] degrees.
# Each original image × each labelled box → up to 4 new samples.

mv_augmentor = MultiViewAugmentor(
    client=client,
    workflow_path=MULTIVIEW_WORKFLOW,
    azimuths=[-90.0, -45.0, 45.0, 90.0],
    elevations=[0.0, 0.0, 0.0, 0.0],
    bg_dir=BACKGROUNDS,          # composite views onto random backgrounds
    crop_padding=0.15,            # 15% padding around bbox when cropping for the model
    use_grabcut=False,            # enable for cleaner foreground extraction
)

n_mv = mv_augmentor.augment_dataset(
    image_dir=DATASET_IMAGES,
    label_dir=DATASET_LABELS,
    output_image_dir=OUTPUT_MV_IMAGES,
    output_label_dir=OUTPUT_MV_LABELS,
    target_classes=None,          # None = all classes; or e.g. [0, 1]
    max_per_image=4,              # max new samples per source image
)
print(f"Multi-view: wrote {n_mv} new samples")

# ---------------------------------------------------------------------------
# Scale augmentation
# ---------------------------------------------------------------------------
# Resizes each labelled object to 50%, 75%, 150%, 200% of its original size.
# The hole left at the original position is filled by ComfyUI inpainting.

scale_augmentor = BboxScaleAugmentor(
    client=client,
    workflow_path=INPAINTING_WORKFLOW,
    scale_values=[0.5, 0.75, 1.5, 2.0],
    bg_dir=BACKGROUNDS,
    use_inpainting=True,          # False = fast background-patch fill (no ComfyUI)
    use_grabcut=True,             # refine hole mask with GrabCut
    inpaint_prompt="background, seamless, high quality",
)

n_scale = scale_augmentor.augment_dataset(
    image_dir=DATASET_IMAGES,
    label_dir=DATASET_LABELS,
    output_image_dir=OUTPUT_SCALE_IMAGES,
    output_label_dir=OUTPUT_SCALE_LABELS,
    target_classes=None,
    max_per_image=8,
)
print(f"Scale: wrote {n_scale} new samples")

# ---------------------------------------------------------------------------
# Single-sample example (useful for debugging)
# ---------------------------------------------------------------------------

import cv2
import numpy as np

sample_image = cv2.cvtColor(cv2.imread("dataset/images/sample.jpg"), cv2.COLOR_BGR2RGB)
sample_boxes = [(0.3, 0.2, 0.7, 0.8, 0)]  # (x_min, y_min, x_max, y_max, cls_id)

# Multi-view: returns list[(image, boxes)] — one per azimuth
views = mv_augmentor.augment_sample(sample_image, sample_boxes, target_idx=0)
for i, (img, boxes) in enumerate(views):
    cv2.imwrite(f"debug_view_{i}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Scale: returns (image, boxes) | None
result = scale_augmentor.augment_sample(sample_image, sample_boxes, target_idx=0, target_scale=2.0)
if result:
    aug_img, aug_boxes = result
    cv2.imwrite("debug_scale.jpg", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    print("Scale result boxes:", aug_boxes)
