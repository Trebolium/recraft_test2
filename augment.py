"""
Image-transform logic for the augmentation pipeline.

Each `*_modest` function takes a PIL Image and returns a new, lightly
perturbed PIL Image -- "modest" meaning the parameter ranges are kept small
so the result still clearly resembles the original. `augment_image()` picks
a random combination of these and overwrites the file in place.
"""

import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

# Modest parameter ranges -- tweak here if augmentations look too strong/weak.
CROP_FRACTION_RANGE = (0.05, 0.15)     # crop away 5-15% of each dimension
ROTATE_DEGREES_RANGE = (-10, 10)       # small rotation, no wild angles
BLUR_RADIUS_RANGE = (0.5, 2.0)
CONTRAST_FACTOR_RANGE = (0.85, 1.15)   # 1.0 = unchanged
BRIGHTNESS_FACTOR_RANGE = (0.85, 1.15)


def crop_modest(img: Image.Image) -> Image.Image:
    """Crop a small random border off the image, then resize back to the
    original size so all augmented images keep the dataset's dimensions."""
    width, height = img.size
    frac = random.uniform(*CROP_FRACTION_RANGE)
    dx, dy = int(width * frac), int(height * frac)

    left = random.randint(0, dx)
    top = random.randint(0, dy)
    right = width - random.randint(0, dx)
    bottom = height - random.randint(0, dy)

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.BICUBIC)


def rotate_modest(img: Image.Image) -> Image.Image:
    """Rotate by a small angle. expand=False keeps the canvas size fixed;
    any corners revealed by the rotation are filled with the image's
    average color so we don't introduce jarring black triangles."""
    angle = random.uniform(*ROTATE_DEGREES_RANGE)
    avg_color = tuple(int(c) for c in img.resize((1, 1)).getpixel((0, 0))[:3])
    return img.convert("RGB").rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=avg_color)


def blur_modest(img: Image.Image) -> Image.Image:
    """Apply a light Gaussian blur."""
    radius = random.uniform(*BLUR_RADIUS_RANGE)
    return img.filter(ImageFilter.GaussianBlur(radius))


def adjust_contrast_modest(img: Image.Image) -> Image.Image:
    """Nudge contrast up or down slightly."""
    factor = random.uniform(*CONTRAST_FACTOR_RANGE)
    return ImageEnhance.Contrast(img).enhance(factor)


def adjust_brightness_modest(img: Image.Image) -> Image.Image:
    """Nudge brightness up or down slightly."""
    factor = random.uniform(*BRIGHTNESS_FACTOR_RANGE)
    return ImageEnhance.Brightness(img).enhance(factor)


# All available augmentations, used by augment_image() to build a random combo.
_AUGMENTATIONS = [crop_modest, rotate_modest, blur_modest, adjust_contrast_modest, adjust_brightness_modest]


def augment_image(path: Path) -> list[str]:
    """
    Apply a random combination of 1-3 modest augmentations to the image at
    `path`, overwriting it in place. Returns the names of the augmentations
    applied (useful for logging/summary output).
    """
    chosen = random.sample(_AUGMENTATIONS, k=random.randint(1, 3))

    img = Image.open(path)
    for fn in chosen:
        img = fn(img)

    img.save(path)
    return [fn.__name__ for fn in chosen]
