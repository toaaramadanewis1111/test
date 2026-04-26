"""
Mask Generation Module for Face Image Inpainting
=================================================
"""

from pathlib import Path
import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Default image dimensions
# ---------------------------------------------------------------------------
IMAGE_SIZE = 64

MIN_MASK_RATIO = 0.10
MAX_MASK_RATIO = 0.40


def _mask_ratio(mask: np.ndarray) -> float:
    return 1.0 - mask.mean()


def _validate_mask(mask: np.ndarray) -> bool:
    ratio = _mask_ratio(mask)
    return MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO


# ========================= CENTER MASK =====================================

def generate_center_mask(h=IMAGE_SIZE, w=IMAGE_SIZE):
    total_pixels = h * w

    min_side = int(np.ceil(np.sqrt(MIN_MASK_RATIO * total_pixels)))
    max_side = int(np.floor(np.sqrt(MAX_MASK_RATIO * total_pixels)))

    min_side = max(min_side, 1)
    max_side = min(max_side, min(h, w))

    side = np.random.randint(min_side, max_side + 1)

    y1 = (h - side) // 2
    x1 = (w - side) // 2

    mask = np.ones((h, w), dtype=np.float32)
    mask[y1:y1 + side, x1:x1 + side] = 0.0

    return mask


# ========================= RANDOM MASK =====================================

def generate_random_square_mask(h=IMAGE_SIZE, w=IMAGE_SIZE):
    total_pixels = h * w

    target_ratio = np.random.uniform(MIN_MASK_RATIO, MAX_MASK_RATIO)
    target_area = int(target_ratio * total_pixels)

    aspect = np.random.uniform(0.7, 1.4)

    rect_h = int(np.clip(np.sqrt(target_area / aspect), 1, h))
    rect_w = int(np.clip(np.sqrt(target_area * aspect), 1, w))

    y1 = np.random.randint(0, h - rect_h + 1)
    x1 = np.random.randint(0, w - rect_w + 1)

    mask = np.ones((h, w), dtype=np.float32)
    mask[y1:y1 + rect_h, x1:x1 + rect_w] = 0.0

    return mask


# ========================= IRREGULAR MASK ==================================

def generate_irregular_mask(h=IMAGE_SIZE, w=IMAGE_SIZE, max_attempts=20):
    for _ in range(max_attempts):
        canvas = np.zeros((h, w), dtype=np.uint8)

        num_strokes = np.random.randint(3, 10)

        for _ in range(num_strokes):
            x, y = np.random.randint(0, w), np.random.randint(0, h)

            num_vertices = np.random.randint(4, 12)
            points = [(x, y)]

            for _ in range(num_vertices):
                dx = np.random.randint(-20, 21)
                dy = np.random.randint(-20, 21)

                nx = np.clip(points[-1][0] + dx, 0, w - 1)
                ny = np.clip(points[-1][1] + dy, 0, h - 1)
                points.append((int(nx), int(ny)))

            thickness = np.random.randint(2, 8)

            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, 1, thickness)

        mask = 1.0 - canvas.astype(np.float32)

        if _validate_mask(mask):
            return mask

    return generate_random_square_mask(h, w)


# ========================= PUBLIC API ======================================

_MASK_GENERATORS = {
    "center": generate_center_mask,
    "random_square": generate_random_square_mask,
    "irregular": generate_irregular_mask,
}

MASK_TYPES = list(_MASK_GENERATORS.keys())


def generate_mask(img: np.ndarray, mask_type="random"):
    h, w = img.shape[:2]

    if mask_type == "random":
        mask_type = np.random.choice(MASK_TYPES)

    mask = _MASK_GENERATORS[mask_type](h, w)

    masked_img = img * mask[:, :, np.newaxis]

    return masked_img, mask


# ========================= TEST ============================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(r"C:\Users\toaa ramadan\Desktop\comp vision project\u-net")

    img_path = PROJECT_ROOT / "data" / "processed" / "train" / "000001.jpg"

    img = Image.open(img_path).convert("RGB")
    img = np.array(img) / 255.0

    for _ in range(3):
        masked, mask = generate_mask(img, "random")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.imshow(masked)

    plt.show()