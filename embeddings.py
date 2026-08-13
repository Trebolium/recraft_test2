"""
Real embedding generation using DINOv2 ViT-S/14 (Meta, self-supervised),
run locally on CPU.

Why DINOv2: it's trained so that different augmented views of the same
image (crops, rotations, blur, color jitter -- the same family of edits our
augment.py applies) end up close together in embedding space, while
different images stay far apart. That makes cosine similarity between two
vectors a meaningful "are these near-duplicates" signal, unlike the earlier
placeholder (random noise) or naive perceptual hashes (which are fragile to
rotation/cropping specifically).

ViT-S/14 (the "small" variant, 21M params) is used because it comfortably
runs on CPU -- no GPU needed for a dataset this size.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

EMBEDDING_DIM = 384  # ViT-S/14's native output width
EMBEDDING_DTYPE = np.float16

# DINOv2 uses patch size 14, so inputs must be a multiple of 14 per side.
_IMAGE_SIZE = 224

_PREPROCESS = transforms.Compose([
    transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # standard ImageNet stats
])

# Loaded lazily (and only once) on first use, since downloading/initializing
# the model is slow and most callers just want to embed a handful of images.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _model.eval()
    return _model


def embed_image(path) -> np.ndarray:
    """Return a 384-dim, L2-normalized, fp16 DINOv2 embedding for the image
    at `path`."""
    model = _get_model()

    with Image.open(path) as img:
        img = img.convert("RGB")  # normalize away grayscale/RGBA/palette modes
        tensor = _PREPROCESS(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        embedding = model(tensor).squeeze(0)  # (384,) CLS-token feature

    # L2-normalize so cosine similarity is well-behaved and comparable across images.
    embedding = embedding / embedding.norm()

    return embedding.numpy().astype(EMBEDDING_DTYPE)
