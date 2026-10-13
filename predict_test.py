#!/usr/bin/env python3
"""Predict the (unlabelled) test set with a trained run and stitch the predictions to 3D volumes.

    python predict_test.py results/<experiment>/<run> [--source data/SEGTHOR] [--device auto]

Slices every <source>/test/Patient_XX.nii.gz with the run's own preprocessing (window, target_spacing,
shape, extra windows), predicts each slice with <run>/bestweights.pt and writes:
  <run>/best_epoch/test/Patient_XX_zzzz.png   2D predictions (class k stored as k * 63, like best_epoch/val)
  <run>/volumes/test/Patient_XX.nii.gz        raw prediction on the native CT grid
  <run>/volumes/test_post/Patient_XX.nii.gz   the same after the run's eval.postprocess
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from main import img_transform
from slice_segthor import slice_patient
from segpipe.config import load_config
from segpipe.data import K, cache_dir, context_slices, extra_windows, n_channels, parse_stem
from segpipe.evaluate import stitch_to_native
from segpipe.models import build_model
from segpipe.postprocess import build_postprocess
from segpipe.train import pick_device
from utils import tqdm_

LABEL_STEP = 63  # class k is stored as k * 63 in the PNGs, as in the slice cache and best_epoch/val


def build_test_cache(cfg, source: Path, patient_ids: list[str]) -> Path:
    """Slice the test CTs once with the run's preprocessing, into data/cache/test_<train cache name>."""
    dest = cache_dir(cfg).with_name("test_" + cache_dir(cfg).name)
    window = tuple(cfg.data.window) if cfg.data.get("window") is not None else None
    target_spacing = tuple(cfg.data.target_spacing) if cfg.data.get("target_spacing") is not None else None
    done = {p.stem.rsplit("_", 1)[0] for p in (dest / "img").glob("*.png")}
    for pid in tqdm_([p for p in patient_ids if p not in done]):
        slice_patient(pid, dest_path=dest, source_path=source, shape=tuple(cfg.data.shape), test_mode=True,
                      window=window, target_spacing=target_spacing, extra_windows=extra_windows(cfg))
    return dest


def load_patient(cache: Path, cfg, pid: str) -> tuple[list[str], torch.Tensor]:
    """All slices of one patient as (Z, n_channels, H, W), with the same 2.5D stacking as SliceDataset."""
    channel_dirs = [cache / "img"] + [cache / f"img{i}" for i in range(1, len(extra_windows(cfg)) + 1)]
    stems = sorted((p.stem for p in (cache / "img").glob(f"{pid}_*.png")), key=lambda s: parse_stem(s)[1])
    slices = {parse_stem(s)[1]: torch.cat([img_transform(Image.open(d / f"{s}.png")) for d in channel_dirs])
              for s in stems}
    lo, hi = min(slices), max(slices)
    context = context_slices(cfg)
    stacks = [torch.cat([slices[min(max(parse_stem(s)[1] + o, lo), hi)] for o in range(-context, context + 1)])
              for s in stems]
    return stems, torch.stack(stacks)


def save_volume(volume: np.ndarray, ct_nib, path: Path) -> None:
    # A fresh header from the CT affine: the CT header may carry intensity scaling (scl_slope/inter)
    # that must not be applied to the labels.
    img = nib.nifti1.Nifti1Image(volume.astype(np.uint8), affine=ct_nib.affine)
    img.set_qform(ct_nib.affine, code=1)
    img.set_sform(ct_nib.affine, code=1)
    img.header.set_zooms(ct_nib.header.get_zooms()[:3])
    nib.save(img, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", type=Path, default=Path("data/SEGTHOR"), help="folder containing test/")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg = load_config(run_dir / "config.yaml")
    device = pick_device(args.device)
    patient_ids = sorted(p.name.removesuffix(".nii.gz") for p in (args.source / "test").glob("Patient_*.nii.gz"))
    assert patient_ids, f"no Patient_*.nii.gz in {args.source / 'test'}"
    print(f">>> {run_dir} on {device}: {len(patient_ids)} test patients")

    cache = build_test_cache(cfg, args.source, patient_ids)

    model = build_model(cfg.model, n_channels(cfg), K)
    model.load_state_dict(torch.load(run_dir / "bestweights.pt", map_location="cpu"))
    model.to(device).eval()

    pred_dir = run_dir / "best_epoch" / "test"
    pred_dir.mkdir(parents=True, exist_ok=True)
    postprocess = build_postprocess((cfg.get("eval") or {}).get("postprocess"))
    out_dirs = {"test": None, "test_post": postprocess} if postprocess is not None else {"test": None}
    for name in out_dirs:
        (run_dir / "volumes" / name).mkdir(parents=True, exist_ok=True)

    for pid in tqdm_(patient_ids):
        stems, imgs = load_patient(cache, cfg, pid)
        with torch.no_grad():
            classes = torch.cat([model(imgs[i:i + args.batch_size].to(device)).argmax(dim=1).cpu()
                                 for i in range(0, len(imgs), args.batch_size)])
        for stem, seg in zip(stems, classes):
            Image.fromarray((seg.numpy() * LABEL_STEP).astype(np.uint8)).save(pred_dir / f"{stem}.png")

        ct_nib = nib.load(str(args.source / "test" / f"{pid}.nii.gz"))
        spacing = ct_nib.header.get_zooms()[:3]
        images = [pred_dir / f"{stem}.png" for stem in stems]
        pred = stitch_to_native(images, list(range(len(images))), ct_nib.shape[:3], spacing, cfg)
        for name, step in out_dirs.items():
            volume = pred if step is None else step(pred, spacing)
            save_volume(volume, ct_nib, run_dir / "volumes" / name / f"{pid}.nii.gz")

    print(f">>> Wrote {', '.join(str(run_dir / 'volumes' / n) for n in out_dirs)}")


if __name__ == "__main__":
    main()
