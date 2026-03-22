"""
Usage example for BackgroundReplace and BboxRelocate transforms.

Directory layout assumed:
    backgrounds/   ← folder with background images
    sample.jpg     ← source image

Bboxes are in YOLO format: [x_center, y_center, width, height] (normalised).
"""

import cv2
import albumentations as A
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations

from data_enhance import BackgroundReplace, BboxRelocate

# ── load image ──────────────────────────────────────────────────────────────
img_bgr = cv2.imread("sample.jpg")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# YOLO bboxes: [[x_center, y_center, w, h], ...]  — all normalised [0, 1]
yolo_bboxes = [
    [0.5, 0.4, 0.3, 0.5],   # box 1
    [0.2, 0.7, 0.15, 0.2],  # box 2
]
class_labels = ["cat", "dog"]

# ── build pipeline ───────────────────────────────────────────────────────────
transform = A.Compose(
    [
        # Replace background (pixels outside bboxes) — 60 % chance
        BackgroundReplace(
            bg_dir="backgrounds",
            blend_border=8,   # soft feathering at bbox edges
            p=0.6,
        ),
        # Relocate bboxes to random positions on a background image — 40 % chance
        BboxRelocate(
            bg_dir="backgrounds",
            allow_overlap=False,  # try to prevent boxes from overlapping
            max_attempts=50,
            p=0.4,
        ),
        # Any other standard augmentations can follow
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,   # drop bboxes that become < 30 % visible
    ),
)

# ── apply ────────────────────────────────────────────────────────────────────
result = transform(image=img, bboxes=yolo_bboxes, class_labels=class_labels)

out_img    = result["image"]
out_bboxes = result["bboxes"]
out_labels = result["class_labels"]

print("Output bboxes (YOLO):", out_bboxes)
print("Output labels:", out_labels)

# save result
out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", out_bgr)
print("Saved → output.jpg")
