"""3D evaluation of a run's best-epoch validation predictions."""

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from segpipe.data import CLASS_NAMES, K
from stitch import merge_patient

# Tolerance (mm) for the Normalised Surface Dice.
NSD_TAU_MM: float = 1.0


def dice(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    # Volumetric overlap: weight voxel counts by the physical voxel volume (mm^3).
    # For Dice this cancels out (every voxel shares the same volume), so the number
    # is identical to the unweighted form, but the metric is now genuinely computed
    # in physical units and serves as the template for spacing-dependent metrics.
    voxel_volume = float(np.prod(spacing))
    intersection = float((pred & gt).sum()) * voxel_volume
    total = float(pred.sum() + gt.sum()) * voxel_volume
    return 1.0 if total == 0 else 2 * intersection / total


# ---------------------------------------------------------------------------
# Boundary metrics (Metrics Reloaded recommendations for organ segmentation).
# Distances are computed in millimetres, using the voxel spacing, on the surface
# voxels of each class -- so anisotropic slice thickness (e.g. SegTHOR ~0.98mm
# in-plane vs 2.5mm through-plane) is handled correctly. HD (max) is kept next to
# HD95 so the two can be compared: max-HD is dominated by a single outlier voxel,
# HD95 is its robust cousin.
#   hd   : max (classic) Hausdorff  -> worst-case boundary error (outlier-sensitive)
#   hd95 : 95th-percentile Hausdorff -> robust worst-case
#   assd : average symmetric surface distance -> mean boundary error
#   nsd  : Normalised Surface Dice at NSD_TAU_MM -> fraction of surface within tau
# All return NaN when the class is absent from pred or gt (boundary undefined).
# ---------------------------------------------------------------------------

def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing):
    """Symmetric nearest-surface distances (mm) between two binary masks.
    Returns (d_pred_to_gt, d_gt_to_pred), or (None, None) if either mask is empty."""
    def surface(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.empty((0, mask.ndim))
        return np.argwhere(mask & ~binary_erosion(mask))

    pred_surf, gt_surf = surface(pred), surface(gt)
    if len(pred_surf) == 0 or len(gt_surf) == 0:
        return None, None

    sp = np.asarray(spacing, dtype=np.float64)
    d_pred_to_gt, _ = cKDTree(gt_surf * sp).query(pred_surf * sp)
    d_gt_to_pred, _ = cKDTree(pred_surf * sp).query(gt_surf * sp)
    return d_pred_to_gt, d_gt_to_pred


def hd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return float("nan")
    return float(max(d_pg.max(), d_gp.max()))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def assd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return float("nan")
    return float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))


def nsd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return float("nan")
    within = (d_pg <= NSD_TAU_MM).sum() + (d_gp <= NSD_TAU_MM).sum()
    return float(within / (len(d_pg) + len(d_gp)))


# name -> fn(pred_mask, gt_mask, spacing) -> float
METRICS: dict = {
    "dice": dice,
    "hd": hd,
    "hd95": hd95,
    "assd": assd,
    "nsd": nsd,
}


def evaluate_3d(run_dir: Path, cfg, patient_ids: list[str], metric_names) -> dict:
    """Stitch best_epoch/val into volumes (stitch.py) and score them against the GT volumes."""
    for name in metric_names:
        if name not in METRICS:
            raise KeyError(f"unknown metric '{name}'. Known: {sorted(METRICS)}")

    images = sorted((run_dir / "best_epoch" / "val").glob("*.png"))
    volume_dir = run_dir / "volumes" / "val"
    volume_dir.mkdir(parents=True, exist_ok=True)
    source_pattern = str(Path(cfg.data.gt) / "train" / "{id_}" / "GT.nii.gz")

    scores = {name: {} for name in metric_names}  # metric -> patient -> K values
    for pid in patient_ids:
        idxes = [i for i, p in enumerate(images) if p.stem.rsplit("_", 1)[0] == pid]
        merge_patient(pid, str(volume_dir), images, idxes, 256, source_pattern)

        pred = np.asarray(nib.load(str(volume_dir / f"{pid}.nii.gz")).dataobj)
        gt_nib = nib.load(source_pattern.format(id_=pid))
        gt = np.asarray(gt_nib.dataobj)
        spacing = gt_nib.header.get_zooms()[:3]
        for name in metric_names:
            scores[name][pid] = np.array([METRICS[name](pred == k, gt == k, spacing) for k in range(K)])

    out_dir = run_dir / "metrics_3d"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for name, per_patient in scores.items():
        np.savez(out_dir / f"{name}.npz", **per_patient)  # patient -> K values
        table = np.stack(list(per_patient.values()))  # patients x K; averages leave out the background
        # nanmean: boundary metrics are NaN for organs absent in a patient, so a
        # single missing organ must not poison the mean (no-op for dice, which
        # never returns NaN).
        summary[name] = {
            "mean": round(float(np.nanmean(table[:, 1:])), 4),
            "per_class": {CLASS_NAMES[k]: round(float(np.nanmean(table[:, k])), 4) for k in range(1, K)},
            "per_patient": {pid: {CLASS_NAMES[k]: round(float(v[k]), 4) for k in range(1, K)}
                            for pid, v in per_patient.items()},
        }
    return summary
