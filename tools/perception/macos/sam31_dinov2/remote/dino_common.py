"""DINOv2 crops and features shared by the remote DINOv2 scripts."""

import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from sam_session import bbox

MODELS = {
    "small": "vit_small_patch14_dinov2.lvd142m",
    "base": "vit_base_patch14_dinov2.lvd142m",
    "large": "vit_large_patch14_dinov2.lvd142m",
}
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
GREY = (MEAN * 255).astype(np.uint8)
MIN_PIXELS = 50  # same as common.MIN_PIXELS on the Mac side


def load_models(names):
    return {
        k: timm.create_model(MODELS[k], pretrained=True, num_classes=0, img_size=SIZE)
        .cuda()
        .eval()
        for k in names
    }


def object_crop(rgb, mask, pad=0.08):
    """The object alone: background greyed, padded to a square, SIZE x SIZE."""
    x0, y0, x1, y1 = bbox(mask)
    p = int(max(x1 - x0, y1 - y0) * pad)
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
    crop = np.where(mask[y0:y1, x0:x1, None], rgb[y0:y1, x0:x1], GREY)
    side = max(crop.shape[:2])
    square = np.empty((side, side, 3), np.uint8)
    square[:] = GREY
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    return np.asarray(Image.fromarray(square).resize((SIZE, SIZE), Image.BICUBIC))


def reference_crops(work, references):
    """View names and crops of the reference views (files with rgb + mask)."""
    views = list(references)
    crops = []
    for view in views:
        ref = np.load(work / references[view])
        crops.append(object_crop(ref["rgb"], ref["mask"]))
        Image.fromarray(crops[-1]).save(work / "out" / f"ref_{view}.png")
    return views, crops


def embed(model, crops):
    """L2-normalized global (CLS) and mean patch features."""
    x = np.stack([(c / 255.0 - MEAN) / STD for c in crops])
    x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
    with torch.inference_mode():
        tokens = model.forward_features(x)
    return {
        "cls": F.normalize(tokens[:, 0], dim=-1),
        "patch": F.normalize(tokens[:, model.num_prefix_tokens :].mean(1), dim=-1),
    }


def timed(fn):
    """fn() and its GPU-synchronized wall time."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start
