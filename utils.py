#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum

from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res

# IoU metric
def iou_coef(pred, gt):
    inter = intersection(pred, gt).sum(dim=(2, 3))
    uni   = union(pred, gt).sum(dim=(2, 3))
    return inter / (uni + 1e-8)

def iou_3d(pred: Tensor, gt: Tensor) -> Tensor:
    """
    pred, gt: one-hot segmentations of shape (B, K, W, H, D)
    Returns: Tensor of shape (B, K)
    """
    inter = intersection(pred, gt).sum(dim=(2, 3, 4))
    uni = union(pred, gt).sum(dim=(2, 3, 4))

    return inter / (uni + 1e-8)

def hausdorff_distance(pred: Tensor, gt: Tensor) -> Tensor:
    """
    pred, gt: one-hot segmentations of shape (B, K, W, H)
    Returns: Tensor of shape (B, K) with symmetric Hausdorff distances.
    NaN when a class is absent in either pred or gt.
    """
    B, K, W, H = pred.shape
    hd = torch.zeros((B, K), dtype=torch.float32)

    pred_np = pred.cpu().numpy()
    gt_np   = gt.cpu().numpy()

    for b in range(B):
        for c in range(K):
            pred_pts = np.argwhere(pred_np[b, c] > 0)
            gt_pts   = np.argwhere(gt_np[b, c] > 0)

            # If class absent → undefined Hausdorff
            if len(pred_pts) == 0 or len(gt_pts) == 0:
                hd[b, c] = float("nan")
                continue

            # Directed Hausdorff both ways
            hd_fwd = directed_hausdorff(pred_pts, gt_pts)[0]
            hd_bwd = directed_hausdorff(gt_pts, pred_pts)[0]

            hd[b, c] = max(hd_fwd, hd_bwd)

    return hd

# ---------------------------------------------------------------------------
# 3D boundary metrics (spacing-aware), following Metrics Reloaded recommendations
# for organ segmentation. From a single pass over each class' surface voxels we
# derive several *separate* boundary scores so they can be compared:
#   - HD    : max (classic) Hausdorff  -> worst-case error, outlier-sensitive
#   - HD95  : 95th-percentile Hausdorff -> robust worst-case
#   - ASSD  : average symmetric surface distance -> mean boundary error
#   - NSD   : Normalised Surface Dice at tolerance tau -> fraction within tau mm
# Distances are in millimetres, using the voxel spacing, so anisotropic slice
# thickness is handled correctly (e.g. SegTHOR ~0.98mm in-plane vs 2.5mm in z).
# ---------------------------------------------------------------------------

def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Coordinates (voxel indices) of the surface voxels of a binary mask."""
    if not mask.any():
        return np.empty((0, mask.ndim), dtype=np.int64)
    # A voxel is on the surface if it is foreground but not fully interior.
    surface = mask & ~binary_erosion(mask)
    return np.argwhere(surface)


def _symmetric_surface_distances(pred_mask: np.ndarray,
                                 gt_mask: np.ndarray,
                                 spacing) -> Tuple:
    """
    Nearest-surface distances (in mm) between the two masks, both directions.
    Returns (d_pred_to_gt, d_gt_to_pred) or (None, None) when undefined
    (i.e. the class is absent from either mask).
    """
    pred_surf = _surface_voxels(pred_mask)
    gt_surf = _surface_voxels(gt_mask)
    if len(pred_surf) == 0 or len(gt_surf) == 0:
        return None, None

    spacing = np.asarray(spacing, dtype=np.float64)
    pred_mm = pred_surf * spacing
    gt_mm = gt_surf * spacing

    d_pred_to_gt, _ = cKDTree(gt_mm).query(pred_mm)
    d_gt_to_pred, _ = cKDTree(pred_mm).query(gt_mm)
    return d_pred_to_gt, d_gt_to_pred


def boundary_metrics_3d(pred: Tensor, gt: Tensor, spacing,
                        nsd_tau: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    pred, gt: one-hot segmentations of shape (B, K, W, H, D).
    spacing:  (s0, s1, s2) voxel size in mm for the (W, H, D) axes.
    nsd_tau:  tolerance (mm) for the Normalised Surface Dice.

    Returns (hd, hd95, assd, nsd), each a Tensor of shape (B, K). All four come
    from the SAME surface-distance pass, so they are cheap to report together
    and can be compared side by side. Entries are NaN when the class is absent
    from pred or gt (boundary distance undefined), so aggregate with np.nanmean.
    """
    B, K = pred.shape[:2]
    hd = torch.full((B, K), float("nan"), dtype=torch.float32)
    hd95 = torch.full((B, K), float("nan"), dtype=torch.float32)
    assd = torch.full((B, K), float("nan"), dtype=torch.float32)
    nsd = torch.full((B, K), float("nan"), dtype=torch.float32)

    pred_np = pred.cpu().numpy().astype(bool)
    gt_np = gt.cpu().numpy().astype(bool)

    for b in range(B):
        for k in range(K):
            d_pg, d_gp = _symmetric_surface_distances(pred_np[b, k], gt_np[b, k], spacing)
            if d_pg is None:
                continue

            hd[b, k] = max(d_pg.max(), d_gp.max())
            hd95[b, k] = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
            assd[b, k] = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
            nsd[b, k] = ((d_pg <= nsd_tau).sum() + (d_gp <= nsd_tau).sum()) / (len(d_pg) + len(d_gp))

    return hd, hd95, assd, nsd