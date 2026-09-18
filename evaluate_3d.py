import numpy as np
import time
import argparse
import os
from pathlib import Path
import torch
from PIL import Image

from utils import dice_batch, iou_3d #hausdorff_distance_3d

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--epoch", type=int, default=0)
args = parser.parse_args()

BASE = os.path.join("data", "experiments", args.experiment)
PRED_DIR = os.path.join(
    BASE, "results", f"iter{args.epoch:03d}", "val_3d"
)
GT_DIR = os.path.join(BASE, "val", "gt")


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

for patient, files in sorted(patients.items()):

    # Sort slices by slice number
    files = sorted(
        files,
        key=lambda x: int(x.stem.split("_")[-1])
    )

    stems = [f.stem for f in files]

    # Predictions: [W,H,D]
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

    # Convert to one-hot: [K,W,H,D]
    K = 5

    pred = np.stack(
        [pred == k for k in range(K)]
    )

    gt = np.stack(
        [gt == k for k in range(K)]
    )

    # Add batch dimension
    pred = torch.from_numpy(
        pred.astype(bool)
    ).unsqueeze(0)

    gt = torch.from_numpy(
        gt.astype(bool)
    ).unsqueeze(0)
    # -----------------------------------------------------
    # All actual metric calculations happen in utils.py
    # -----------------------------------------------------

    dice = dice_batch(pred, gt)
    iou = iou_3d(pred, gt)
    # hd = hausdorff_distance_3d(pred, gt)

    all_dice.append(dice[0].numpy())
    all_iou.append(iou[0].numpy())
    # all_hd.append(hd[0].numpy())


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

dice = np.asarray(all_dice)
iou = np.asarray(all_iou)
# hd = np.asarray(all_hd)

# Ignore background
dice = dice[:, 1:]
iou = iou[:, 1:]
# hd = hd[:, 1:]


print("=== 3D Validation Metrics ===")

print(f"Mean Dice: {np.nanmean(dice):.4f}")
print(f"Mean IoU : {np.nanmean(iou):.4f}")
# print(f"Mean HD  : {np.nanmean(hd):.4f}")


for c in range(dice.shape[1]):
    print(
        f"Class {c+1} "
        f"Dice: {np.nanmean(dice[:, c]):.4f} "
        f"IoU: {np.nanmean(iou[:, c]):.4f} "
        # f"HD: {np.nanmean(hd[:, c]):.4f}"
    )

# Save the 3D metrics
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = os.path.join(
    BASE,
    f"metrics_3d_summary_epoch{args.epoch}_{timestamp}.txt"
)

with open(filename, "w") as f:

    f.write("=== 3D Validation Metrics ===\n\n")

    f.write(
        f"Experiment: {args.experiment}\n"
    )
    f.write(
        f"Epoch: {args.epoch}\n\n"
    )

    f.write(
        f"Mean Dice: {np.nanmean(dice):.4f}\n"
    )
    f.write(
        f"Mean IoU : {np.nanmean(iou):.4f}\n"
    )
    #f.write(
        #f"Mean HD  : {np.nanmean(hd):.4f}\n\n"
    #)

    for c in range(dice.shape[1]):

        f.write(
            f"Class {c+1} Dice: "
            f"{np.nanmean(dice[:, c]):.4f}\n"
        )

        f.write(
            f"Class {c+1} IoU : "
            f"{np.nanmean(iou[:, c]):.4f}\n"
        )

        #f.write(
         #   f"Class {c+1} HD  : "
          #  f"{np.nanmean(hd[:, c]):.4f}\n"
        #)

print(
    f"\nThe 3D evaluation has been saved as "
    f"'{os.path.basename(filename)}'"
)

print(
    f"Location: {os.path.abspath(filename)}"
)