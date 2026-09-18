import numpy as np
import time
import argparse
import os
import pickle
from pathlib import Path
import torch
from PIL import Image

from utils import dice_batch, iou_3d, boundary_metrics_3d

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--nsd-tau", type=float, default=1.0,
                    help="Tolerance (mm) for the Normalised Surface Dice.")
args = parser.parse_args()

BASE = os.path.join("data", "experiments", args.experiment)
PRED_DIR = os.path.join(
    BASE, "results", f"iter{args.epoch:03d}", "val_3d"
)
GT_DIR = os.path.join(BASE, "val", "gt")

# Number of classes (SegTHOR: background + 4 organs)
K = 5
CLASS_NAMES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}


# ---------------------------------------------------------
# Voxel spacing (mm), needed for physical boundary distances
# ---------------------------------------------------------

spacing_path = os.path.join(BASE, "spacing.pkl")
if os.path.exists(spacing_path):
    with open(spacing_path, "rb") as f:
        spacings = pickle.load(f)
else:
    spacings = {}
    print(
        f"[warning] {spacing_path} not found; boundary distances will be "
        f"reported in VOXELS, not mm."
    )


# ---------------------------------------------------------
# Collect prediction slices per patient
# ---------------------------------------------------------

pred_files = sorted(Path(PRED_DIR).glob("*.npy"))

patients = {}

for file in pred_files:
    stem = file.stem
    patient = "_".join(stem.split("_")[:2])

    patients.setdefault(patient, []).append(file)


# ---------------------------------------------------------
# Evaluate every patient as one 3D volume
# ---------------------------------------------------------

all_dice = []
all_iou = []
all_hd = []
all_hd95 = []
all_assd = []
all_nsd = []

for patient, files in sorted(patients.items()):

    # Sort slices by slice number
    files = sorted(
        files,
        key=lambda x: int(x.stem.split("_")[-1])
    )

    stems = [f.stem for f in files]

    # Predictions: [W,H,D] (raw class labels 0..K-1)
    pred = np.stack(
        [np.load(f) for f in files],
        axis=-1
    )

    # Ground truth: [W,H,D]
    gt = np.stack(
        [
            np.array(Image.open(Path(GT_DIR) / f"{stem}.png"))
            for stem in stems
        ],
        axis=-1
    )

    # GT pngs are stored scaled for visualisation (mult = 63 for K == 5),
    # so decode them back to raw class labels 0..K-1 before one-hot encoding.
    mult = 63 if K == 5 else 255 // (K - 1)
    if gt.max() > K - 1:
        gt = np.rint(gt / mult).astype(np.int64)

    # Convert to one-hot: [K,W,H,D]
    pred = np.stack(
        [pred == k for k in range(K)]
    )

    gt = np.stack(
        [gt == k for k in range(K)]
    )

    # Add batch dimension: [1,K,W,H,D]
    pred = torch.from_numpy(
        pred.astype(bool)
    ).unsqueeze(0)

    gt = torch.from_numpy(
        gt.astype(bool)
    ).unsqueeze(0)

    # Physical voxel spacing for this patient (mm); axes are (W, H, D).
    spacing = spacings.get(patient, (1.0, 1.0, 1.0))

    # -----------------------------------------------------
    # All actual metric calculations happen in utils.py
    # -----------------------------------------------------

    # Overlap metrics
    dice = dice_batch(pred, gt)   # shape (K,)
    iou = iou_3d(pred, gt)        # shape (1, K)

    # Boundary metrics (all four from one surface pass), in mm
    hd, hd95, assd, nsd = boundary_metrics_3d(
        pred, gt, spacing, nsd_tau=args.nsd_tau
    )

    all_dice.append(dice.numpy())
    all_iou.append(iou[0].numpy())
    all_hd.append(hd[0].numpy())
    all_hd95.append(hd95[0].numpy())
    all_assd.append(assd[0].numpy())
    all_nsd.append(nsd[0].numpy())


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

dice = np.asarray(all_dice)
iou = np.asarray(all_iou)
hd = np.asarray(all_hd)
hd95 = np.asarray(all_hd95)
assd = np.asarray(all_assd)
nsd = np.asarray(all_nsd)

# Ignore background (class 0)
dice = dice[:, 1:]
iou = iou[:, 1:]
hd = hd[:, 1:]
hd95 = hd95[:, 1:]
assd = assd[:, 1:]
nsd = nsd[:, 1:]


def class_label(c: int) -> str:
    # c is 0-based over the foreground classes; organ index is c + 1
    name = CLASS_NAMES.get(c + 1)
    return f"Class {c + 1} ({name})" if name else f"Class {c + 1}"


print("=== 3D Validation Metrics ===")
print(f"(boundary distances in mm; NSD tolerance = {args.nsd_tau} mm)\n")

print(f"Mean Dice : {np.nanmean(dice):.4f}")
print(f"Mean IoU  : {np.nanmean(iou):.4f}")
print(f"Mean HD   : {np.nanmean(hd):.4f} mm")
print(f"Mean HD95 : {np.nanmean(hd95):.4f} mm")
print(f"Mean ASSD : {np.nanmean(assd):.4f} mm")
print(f"Mean NSD  : {np.nanmean(nsd):.4f}\n")

for c in range(dice.shape[1]):
    print(
        f"{class_label(c)} "
        f"Dice: {np.nanmean(dice[:, c]):.4f} "
        f"IoU: {np.nanmean(iou[:, c]):.4f} "
        f"HD: {np.nanmean(hd[:, c]):.2f} "
        f"HD95: {np.nanmean(hd95[:, c]):.2f} "
        f"ASSD: {np.nanmean(assd[:, c]):.2f} "
        f"NSD: {np.nanmean(nsd[:, c]):.4f}"
    )

# ---------------------------------------------------------
# Save the 3D metrics
# ---------------------------------------------------------

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = os.path.join(
    BASE,
    f"metrics_3d_summary_epoch{args.epoch}_{timestamp}.txt"
)

with open(filename, "w") as f:

    f.write("=== 3D Validation Metrics ===\n\n")

    f.write(f"Experiment: {args.experiment}\n")
    f.write(f"Epoch: {args.epoch}\n")
    f.write(f"NSD tolerance: {args.nsd_tau} mm\n")
    f.write("Boundary distances (HD, HD95, ASSD) are in mm.\n\n")

    f.write(f"Mean Dice : {np.nanmean(dice):.4f}\n")
    f.write(f"Mean IoU  : {np.nanmean(iou):.4f}\n")
    f.write(f"Mean HD   : {np.nanmean(hd):.4f} mm\n")
    f.write(f"Mean HD95 : {np.nanmean(hd95):.4f} mm\n")
    f.write(f"Mean ASSD : {np.nanmean(assd):.4f} mm\n")
    f.write(f"Mean NSD  : {np.nanmean(nsd):.4f}\n\n")

    for c in range(dice.shape[1]):
        f.write(f"{class_label(c)}\n")
        f.write(f"  Dice: {np.nanmean(dice[:, c]):.4f}\n")
        f.write(f"  IoU : {np.nanmean(iou[:, c]):.4f}\n")
        f.write(f"  HD  : {np.nanmean(hd[:, c]):.4f} mm\n")
        f.write(f"  HD95: {np.nanmean(hd95[:, c]):.4f} mm\n")
        f.write(f"  ASSD: {np.nanmean(assd[:, c]):.4f} mm\n")
        f.write(f"  NSD : {np.nanmean(nsd[:, c]):.4f}\n\n")

# Also save raw per-patient arrays so the metrics can be compared / plotted later.
npz_name = os.path.join(
    BASE, f"metrics_3d_perpatient_epoch{args.epoch}_{timestamp}.npz"
)
np.savez(
    npz_name,
    patients=np.array(sorted(patients.keys())),
    dice=dice, iou=iou, hd=hd, hd95=hd95, assd=assd, nsd=nsd,
)

print(
    f"\nThe 3D evaluation has been saved as "
    f"'{os.path.basename(filename)}'"
)
print(f"Per-patient arrays saved as '{os.path.basename(npz_name)}'")
print(f"Location: {os.path.abspath(filename)}")
