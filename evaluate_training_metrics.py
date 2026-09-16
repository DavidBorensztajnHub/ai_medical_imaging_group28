import numpy as np
import time
import os

BASE = r"C:\Users\britt\Documents\ai_medical_imaging_group28\results\toy2\ce"

dice_val = np.load(BASE + "/dice_val.npy")
iou_val  = np.load(BASE + "/iou_val.npy")

# Background class = 0 → skip
dice_no_bg = dice_val[:, :, 1:]
iou_no_bg  = iou_val[:, :, 1:]

print("=== Validation Metrics ===")
print(f"Mean Dice (all classes): {dice_no_bg.mean():.4f}")
print(f"Mean IoU  (all classes): {iou_no_bg.mean():.4f}")

# Per class
num_classes = dice_no_bg.shape[-1]

for c in range(num_classes):
    print(f"Class {c+1} Dice: {dice_no_bg[:, :, c].mean():.4f}")
    print(f"Class {c+1} IoU : {iou_no_bg[:, :, c].mean():.4f}")

# Per epoch (optional)
print("\n=== Per Epoch ===")
for e in range(dice_no_bg.shape[0]):
    print(f"Epoch {e}: Dice={dice_no_bg[e].mean():.4f}, IoU={iou_no_bg[e].mean():.4f}")

# Save the metrics
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = os.path.join(BASE, f"metrics_summary_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"Mean Dice: {dice_no_bg.mean():.4f}\n")
    f.write(f"Mean IoU : {iou_no_bg.mean():.4f}\n\n")

    for c in range(num_classes):
        f.write(f"Class {c+1} Dice: {dice_no_bg[:, :, c].mean():.4f}\n")
        f.write(f"Class {c+1} IoU : {iou_no_bg[:, :, c].mean():.4f}\n")

    f.write("\nPer Epoch:\n")
    for e in range(dice_no_bg.shape[0]):
        f.write(f"Epoch {e}: Dice={dice_no_bg[e].mean():.4f}, IoU={iou_no_bg[e].mean():.4f}\n")

    
