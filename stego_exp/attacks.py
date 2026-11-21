"""Image attacks used for robustness evaluation."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def attack_image(img: Image.Image, attack: str) -> Image.Image:
    attack = attack.lower()
    if attack == "none":
        return img.copy()

    if attack.startswith("jpeg"):
        # jpeg_70 / jpeg_50
        q = int(attack.split("_")[1])
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if attack.startswith("noise"):
        # noise_10 => sigma ~ 10
        sigma = float(attack.split("_")[1])
        arr = np.asarray(img).astype(np.float32)
        arr = arr + np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if attack.startswith("scale"):
        # scale_0.5 / scale_1.5 then resize back to original for transmission size parity
        ratio = float(attack.split("_")[1])
        w, h = img.size
        small = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    if attack == "hist_eq":
        return ImageOps.equalize(img.convert("RGB"))

    if attack.startswith("crop"):
        # crop_0.2 => edge crop 20%
        r = float(attack.split("_")[1])
        w, h = img.size
        dx, dy = int(w * r / 2), int(h * r / 2)
        cropped = img.crop((dx, dy, w - dx, h - dy))
        return cropped.resize((w, h), Image.BILINEAR)

    if attack.startswith("translate"):
        # translate_0.1 => shift by 10% width/height
        r = float(attack.split("_")[1])
        arr = pil_to_cv(img)
        h, w = arr.shape[:2]
        M = np.float32([[1, 0, r * w], [0, 1, r * h]])
        out = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return cv_to_pil(out)

    if attack.startswith("rotate"):
        angle = float(attack.split("_")[1])
        return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    if attack == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))

    if attack == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.5)

    raise ValueError(f"Unknown attack: {attack}")


# Paper-like + geometry-focused set
DEFAULT_ATTACKS = [
    "jpeg_70",
    "noise_10",
    "scale_0.5",
    "hist_eq",
    "crop_0.2",
    "translate_0.1",
    "rotate_15",
    "rotate_30",
]
