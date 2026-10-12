"""Extra candidate figures for the final presentation (the group picks favourites).

    ai4mi/bin/python figures/make_presentation_figs_extra.py

Writes figures/presentation/extra/<speaker>_<nn>_<name>.png (+ .svg). Sources:
- results/<experiment>/<run>/summary.json (the repo; `git pull` first),
- figures/presentation/data/wandb_cache.json (runs that are only in W&B: final model seed 42, k32 + cosine50,
  the CV folds, training curves),
- the preprocessed slices in data/cache/ and saved predictions in results/*/best_epoch/val (image figures).
Style and colours come from make_presentation_figs.py (blue = mean Dice, orange = esophagus).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import make_presentation_figs as base  # noqa: E402
from make_presentation_figs import (AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, SURFACE, VIOLET,  # noqa: E402
                                    rounded_bar, style_axes)

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "presentation" / "extra"
CACHE_N = ROOT / "data/cache/segthor_full_window40_400_sp0.98_0.98_2_384x384"
CACHE_M = ROOT / "data/cache/segthor_full_window40_400_sp1.95_1.95_2.5_256x256"
WB = json.loads((Path(__file__).parent / "presentation/data/wandb_cache.json").read_text())

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
# organ identity colours (anatomy images and per-organ charts); esophagus is always orange
ORGAN_C = {"esophagus": ORANGE, "heart": "#e87ba4", "trachea": "#1baf7a", "aorta": "#4a3aa7"}
RED = "#e34948"
HOLDOUT = ["Patient_01", "Patient_13", "Patient_16", "Patient_22", "Patient_28", "Patient_30", "Patient_35", "Patient_39"]


# ---------------------------------------------------------------- data access
class Run:
    """One run's summary, from the repo (nested json) or the W&B cache (flat keys)."""

    def __init__(self, d, flat):
        self.d, self.flat = d, flat

    def get(self, metric="dice", cls=None, patient=None, key="metrics_3d_post"):
        if self.flat:
            k = f"{key}/{metric}/" + (f"per_patient/{patient}/{cls}" if patient else f"per_class/{cls}" if cls else "mean")
            return self.d[k]
        m = self.d[key][metric]
        return m["per_patient"][patient][cls] if patient else m["per_class"][cls] if cls else m["mean"]

    def pmean(self, patient, metric="dice", key="metrics_3d_post"):
        return float(np.mean([self.get(metric, o, patient, key) for o in ORGANS]))


def repo(exp, seed=42, run=None):
    run = run or f"holdout40-f0-s{seed}"
    return Run(json.loads((ROOT / "results" / exp / run / "summary.json").read_text()), False)


def wb(name):
    return Run(WB["summaries"][name], True)


def final_runs():
    return [wb("combo_core_ce_k16_cosine50/holdout40-f0-s42"), repo("combo_core_ce_k16_cosine50", 43)]


def mean_of(runs, **kw):
    return float(np.mean([r.get(**kw) for r in runs]))


# ---------------------------------------------------------------- helpers
def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def legend(ax, items, kind="o", ncol=None, y=1.1, x=0):
    hs = [plt.Line2D([0], [0], marker=kind, linestyle="", markersize=9, markerfacecolor=c, markeredgecolor=SURFACE,
                     markeredgewidth=1.5 if kind == "o" else 0, label=l) for l, c in items]
    lg = ax.legend(handles=hs, loc="upper left", frameon=False, ncol=ncol or len(items), bbox_to_anchor=(x, y),
                   borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
    for t in lg.get_texts():
        t.set_color(INK_2)


def dot(ax, x, y, c, s=110, z=3, **kw):
    ax.scatter(x, y, s=s, color=c, edgecolors=SURFACE, linewidths=2, zorder=z, **kw)


def hgrid(ax):
    style_axes(ax, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def note(ax, text, y=-0.18):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def load_slice(cache, patient, z):
    # stored as (x, y); transposed = radiological view: anterior up, patient's left on the image's right
    img = np.array(Image.open(cache / "img" / f"{patient}_{z:04d}.png")).astype(float).T
    gt = (np.array(Image.open(cache / "gt" / f"{patient}_{z:04d}.png")) // 63).T
    return img, gt


def contours(ax, lab, extent=None, lw=1.8, organs=ORGANS, style="-"):
    from skimage import measure
    h, w = lab.shape
    for o in organs:
        m = lab == ORGANS.index(o) + 1
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ys, xs = c[:, 0], c[:, 1]
            if extent is not None:  # pixel -> mm
                x0, x1, y1, y0 = extent
                xs = x0 + (xs + 0.5) * (x1 - x0) / w
                ys = y0 + (ys + 0.5) * (y1 - y0) / h
            ax.plot(xs, ys, color=ORGAN_C[o], linewidth=lw, linestyle=style)


def extent_mm(shape, spacing):
    h, w = shape
    return (-w / 2 * spacing, w / 2 * spacing, h / 2 * spacing, -h / 2 * spacing)  # centre-cropped grids


def eso_slices(cache, patient):
    out = []
    for f in sorted(glob.glob(str(cache / "gt" / f"{patient}_*.png"))):
        lab = np.array(Image.open(f)) // 63
        out.append((int(Path(f).stem.rsplit("_", 1)[1]), lab))
    return out


# ================================================================ ELENA
def e01_midterm_per_organ():
    r = repo("current")
    fig, ax = plt.subplots(figsize=(6, 4))
    style_axes(ax)
    ax.set_ylim(0, 1)
    vals = [r.get("dice", o, key="metrics_3d") for o in ORGANS]
    for i, (o, v) in enumerate(zip(ORGANS, vals)):
        rounded_bar(ax, i, 0.5, v, ORANGE if o == "esophagus" else AXIS)
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                color=INK if o == "esophagus" else INK_2, fontweight="bold" if o == "esophagus" else None)
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks(range(4))
    tl = ax.set_xticklabels([o.capitalize() for o in ORGANS], fontsize=11)
    tl[0].set_fontweight("bold")
    ax.set_ylabel("3D Dice")
    ax.set_title("Midterm pipeline: per-organ Dice (8 holdout patients)", loc="left", fontsize=11, color=INK_2)
    save(fig, "elena_01_midterm_per_organ")


def e02_ct_slices():
    """Two axial slices of one holdout patient with the GT contours of the 4 organs."""
    p = "Patient_28"
    sl = eso_slices(CACHE_N, p)
    upper = [z for z, l in sl if (l == 3).any() and (l == 1).any() and (l == 4).any()]
    lower = [z for z, l in sl if (l == 2).sum() > 4000 and (l == 1).any() and (l == 4).any()]
    zs = [upper[len(upper) // 2], lower[len(lower) // 2]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, z, t in zip(axes, zs, ("Upper thorax: trachea level", "Lower thorax: heart level")):
        img, lab = load_slice(CACHE_N, p, z)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        contours(ax, lab, lw=1.6)
        ax.set_xlim(55, 329)
        ax.set_ylim(320, 60)
        ax.axis("off")
        ax.set_title(t, fontsize=11, color=INK_2, loc="left")
    legend(axes[0], [(o.capitalize(), ORGAN_C[o]) for o in ORGANS], kind="s", y=-0.02)
    fig.subplots_adjust(wspace=0.04, bottom=0.1)
    save(fig, "elena_02_ct_slices_organs")


def e03_esophagus_resolution():
    """The same anatomy at 1.95 mm (midterm) and 0.98 mm (native), pixels visible, esophagus outlined."""
    p = "Patient_28"
    sn = [(z, (l == 1).sum()) for z, l in eso_slices(CACHE_N, p) if (l == 1).sum() > 0]
    areas = [a for _, a in sn]
    zn = min(sn, key=lambda t: abs(t[1] - np.median(areas)))[0]  # a typical (median-area) esophagus slice
    zm = int(round(zn * 2.0 / 2.5))
    imn, labn = load_slice(CACHE_N, p, zn)
    imm, labm = load_slice(CACHE_M, p, zm)
    rr, cc = np.nonzero(labn == 1)
    cy, cx = (rr.mean() - 192) * 0.98, (cc.mean() - 192) * 0.98  # esophagus centre in mm
    half = 32
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, img, lab, sp, t in ((axes[0], imm, labm, 1.95, "1.95 mm (midterm)"),
                                (axes[1], imn, labn, 0.98, "0.98 mm (native, final)")):
        ext = extent_mm(img.shape, sp)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=ext, interpolation="nearest")
        contours(ax, lab, extent=ext, organs=["esophagus"], lw=2.4)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        n = int((lab == 1).sum())
        ax.set_title(f"{t}\nesophagus = {n} pixels → ~{max(1, round(n / 64))} at ENet's 1/8 bottleneck",
                     fontsize=11, color=INK, loc="left")
    axes[0].plot([cx - half + 4, cx - half + 14], [cy + half - 4] * 2, color="white", linewidth=3)
    import matplotlib.patheffects as pe
    axes[0].text(cx - half + 9, cy + half - 6, "10 mm", color="white", ha="center", fontsize=10,
                 path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    fig.subplots_adjust(wspace=0.05)
    save(fig, "elena_03_esophagus_at_two_resolutions")


def e04_preprocessing_ablation():
    base_d = repo("current").get("dice", key="metrics_3d")
    rows = [("No HU window (min-max)", "no_hu_window", [42]), ("No resampling (resize)", "no_resampling", [42]),
            ("Narrow window [40, 300]", "window_narrow", [42]), ("Slice thickness 2.0 mm only", "spacing_1.95_z2", [42]),
            ("Lung window channel", "lung_window_channel", [42]), ("1.5 mm in-plane", "spacing_1.5_256", [42]),
            ("Tight crop at 1.95 mm", "spacing_1.95_crop192", [42]),
            ("Native 0.98 mm, 384×384", "spacing_native_384", [42, 43])]
    deltas = [(lab, np.mean([repo(e, s).get("dice", key="metrics_3d") for s in seeds]) - base_d, len(seeds))
              for lab, e, seeds in rows]
    deltas.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    hgrid(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    for y, (lab, d, n) in enumerate(deltas):
        ax.barh(y, d, height=0.55, color=BLUE if d > 0 else RED, alpha=1.0 if "Native" in lab else 0.75)
        ax.text(d + (0.002 if d > 0 else -0.002), y, f"{d:+.3f}" + (" (2 seeds)" if n > 1 else ""),
                va="center", ha="left" if d > 0 else "right", fontsize=10, color=INK_2)
    ax.set_yticks(range(len(deltas)))
    tl = ax.set_yticklabels([l for l, *_ in deltas], fontsize=10.5)
    for t in tl:
        t.set_color(INK)
        if "Native" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlim(-0.05, 0.065)
    ax.set_xlabel("Δ mean Dice vs midterm pipeline (0.787, raw)")
    ax.set_title("One change at a time on the midterm pipeline", loc="left", fontsize=11, color=INK_2)
    save(fig, "elena_04_preprocessing_ablation")


def e05_resolution_per_organ():
    steps = [("1.95 mm", [repo("current")]), ("1.5 mm", [repo("spacing_1.5_256")]),
             ("0.98 mm\n(native)", [repo("spacing_native_384", 42), repo("spacing_native_384", 43)])]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    style_axes(ax)
    for o in ORGANS:
        ys = [mean_of(rs, metric="dice", cls=o, key="metrics_3d") for _, rs in steps]
        ax.plot(range(3), ys, color=ORGAN_C[o], linewidth=2.5 if o == "esophagus" else 2, zorder=2)
        dot(ax, range(3), ys, ORGAN_C[o])
        ax.text(2.12, ys[-1], f"{o.capitalize()}  {ys[-1]:.3f}  ({ys[-1] - ys[0]:+.3f})", va="center", fontsize=10,
                color=INK, fontweight="bold" if o == "esophagus" else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s for s, _ in steps])
    ax.set_xlim(-0.2, 3.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("3D Dice (raw)")
    ax.set_title("Finer in-plane resolution helps the small organs most", loc="left", fontsize=11, color=INK_2)
    save(fig, "elena_05_resolution_per_organ")


def e06_qualitative_midterm_vs_native():
    """GT vs midterm vs native prediction on the same slice (physical mm, aligned)."""
    p = "Patient_28"
    sn = [(z, (l == 1).sum()) for z, l in eso_slices(CACHE_N, p) if (l == 1).sum() > 0]
    zn = min(sn, key=lambda t: abs(t[1] - np.median([a for _, a in sn])))[0]
    zm = int(round(zn * 2.0 / 2.5))
    imn, labn = load_slice(CACHE_N, p, zn)
    imm, labm = load_slice(CACHE_M, p, zm)
    pm = (np.array(Image.open(ROOT / f"results/current/holdout40-f0-s42/best_epoch/val/{p}_{zm:04d}.png")) // 63).T
    pn = (np.array(Image.open(ROOT / f"results/spacing_native_384/holdout40-f0-s42/best_epoch/val/{p}_{zn:04d}.png")) // 63).T
    rr, cc = np.nonzero(labn > 0)
    cy, cx = (np.median(rr) - 192) * 0.98, (np.median(cc) - 192) * 0.98
    half = 55
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, img, sp, lab, t in ((axes[0], imn, 0.98, labn, "Ground truth"),
                                (axes[1], imm, 1.95, pm, "Midterm pipeline (1.95 mm)"),
                                (axes[2], imn, 0.98, pn, "Native resolution (0.98 mm)")):
        ext = extent_mm(img.shape, sp)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=ext)
        contours(ax, lab, extent=ext, lw=2)
        if t != "Ground truth":  # GT esophagus as a dashed white reference
            ref_lab, ref_ext = (labm, extent_mm(labm.shape, 1.95)) if sp == 1.95 else (labn, extent_mm(labn.shape, 0.98))
            from skimage import measure
            for c in measure.find_contours((ref_lab == 1).astype(float), 0.5):
                h, w = ref_lab.shape
                x0, x1, y1, y0 = ref_ext
                ax.plot(x0 + (c[:, 1] + 0.5) * (x1 - x0) / w, y0 + (c[:, 0] + 0.5) * (y1 - y0) / h,
                        color="white", linewidth=1.2, linestyle="--")
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.axis("off")
        ax.set_title(t, fontsize=11, color=INK, loc="left")
    hs = [plt.Line2D([0], [0], color=ORGAN_C[o], linewidth=2, label=o.capitalize()) for o in ORGANS]
    hs.append(plt.Line2D([0], [0], color=MUTED, linewidth=1.5, linestyle="--", label="Ground-truth esophagus"))
    lg = axes[0].legend(handles=hs, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=5, frameon=False)
    for t in lg.get_texts():
        t.set_color(INK_2)
    fig.subplots_adjust(wspace=0.03, bottom=0.12)
    save(fig, "elena_06_qualitative_midterm_vs_native")


# ================================================================ GITHA
def g07_augmentation_2x2():
    data = {("1.95 mm", "No augmentation"): [repo("current").get(key="metrics_3d")],
            ("1.95 mm", "Augmentation"): [repo("augmentation-new-data").get(key="metrics_3d"),
                                          repo("augmentation-new-data2.0").get(key="metrics_3d")],
            ("Native 0.98 mm", "No augmentation"): [repo("spacing_native_384", s).get(key="metrics_3d") for s in (42, 43)],
            ("Native 0.98 mm", "Augmentation"): [repo("augmentation-native384", s).get(key="metrics_3d")
                                                 for s in (42, 43, 44)]}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    style_axes(ax)
    for gi, g in enumerate(("1.95 mm", "Native 0.98 mm")):
        for si, (s, c) in enumerate((("No augmentation", MUTED), ("Augmentation", BLUE))):
            x = gi + (si - 0.5) * 0.36
            vals = data[(g, s)]
            ax.plot([x - 0.12, x + 0.12], [np.mean(vals)] * 2, color=c, linewidth=3, solid_capstyle="round")
            dot(ax, [x] * len(vals), vals, c, s=70)
            ax.text(x, max(vals) + 0.006, f"{np.mean(vals):.3f}", ha="center", fontsize=10.5, color=INK_2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1.95 mm (midterm)", "Native 0.98 mm"], fontsize=11)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.76, 0.89)
    ax.set_ylabel("3D Dice (raw)")
    legend(ax, [("No augmentation", MUTED), ("Augmentation", BLUE)])
    note(ax, "Dots = individual runs (seeds); line = mean. 1.95 mm augmentation: two different versions, 1 seed each.",
         y=-0.16)
    save(fig, "githa_07_augmentation_by_resolution")


def g08_seed_spread():
    no = [repo("spacing_native_384", s).get(key="metrics_3d") for s in (42, 43)]
    yes = [repo("augmentation-native384", s).get(key="metrics_3d") for s in (42, 43, 44)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    hgrid(ax)
    for y, (vals, c, lab) in enumerate(((yes, BLUE, "With augmentation (3 seeds)"), (no, MUTED, "Without (2 seeds)"))):
        ax.plot([min(vals), max(vals)], [y, y], color=c, linewidth=6, alpha=0.25, solid_capstyle="round")
        dot(ax, vals, [y] * len(vals), c)
        ax.text(max(vals) + 0.002, y, f"spread {max(vals) - min(vals):.3f}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks([0, 1])
    for t in ax.set_yticklabels(["With augmentation\n(3 seeds)", "Without augmentation\n(2 seeds)"], fontsize=10.5):
        t.set_color(INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0.815, 0.885)
    ax.set_xlabel("3D Dice, native resolution (raw); each dot = one seed")
    ax.set_title("Augmentation makes runs reproducible", loc="left", fontsize=11, color=INK_2)
    save(fig, "githa_08_seed_spread")


def g09_augmentation_examples():
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    p, z = "Patient_28", 0
    sl = eso_slices(CACHE_N, p)
    z = max(((s, (l == 2).sum()) for s, l in sl if (l == 1).any() and (l == 4).any()), key=lambda t: t[1])[0]
    img, lab = load_slice(CACHE_N, p, z)
    ti = torch.tensor(img / 255.0, dtype=torch.float32)[None, None]
    tl = torch.tensor(lab, dtype=torch.float32)[None, None]
    NEAR, BIL = TF.InterpolationMode.NEAREST, TF.InterpolationMode.BILINEAR
    torch.manual_seed(3)
    disp = T.ElasticTransform.get_params([35.0, 35.0], [5.0, 5.0], [384, 384])
    variants = [
        ("Original", ti, tl),
        ("Rotation + scale\n(±10°, 0.9–1.1×)", TF.affine(ti, angle=10, translate=[0, 0], scale=1.1, shear=0, interpolation=BIL),
         TF.affine(tl, angle=10, translate=[0, 0], scale=1.1, shear=0, interpolation=NEAR)),
        ("Elastic deformation", TF.elastic_transform(ti, disp, BIL), TF.elastic_transform(tl, disp, NEAR)),
        ("Brightness ×1.2", TF.adjust_brightness(ti, 1.2), tl),
        ("Contrast ×0.8", TF.adjust_contrast(ti, 0.8), tl),
        ("Roll (wraps around!)", torch.roll(ti, shifts=(70, 60), dims=(-2, -1)), torch.roll(tl, shifts=(70, 60), dims=(-2, -1))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for ax, (t, i, l) in zip(axes.flat, variants):
        ax.imshow(i[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        contours(ax, l[0, 0].numpy().round().astype(int), lw=1.3)
        ax.set_xlim(40, 344)
        ax.set_ylim(344, 40)
        ax.axis("off")
        ax.set_title(t, fontsize=10.5, color=RED if "wraps" in t else INK, loc="left")
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    save(fig, "githa_09_augmentation_examples")


# ================================================================ PUCK
def p10_pixel_share():
    from segpipe.data import load_split
    train, _ = load_split("holdout40", 0)
    counts = np.zeros(5)
    for p in train:
        for f in glob.glob(str(CACHE_N / "gt" / f"{p}_*.png")):
            counts += np.bincount((np.array(Image.open(f)) // 63).ravel(), minlength=5)[:5]
    share = counts / counts.sum()
    fg = counts[1:] / counts[1:].sum()
    order = [1, 3, 4, 0]  # heart, aorta, trachea, esophagus by size (indices into ORGANS order below)
    names = ["esophagus", "heart", "trachea", "aorta"]
    idx = sorted(range(4), key=lambda i: -fg[i])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hgrid(ax)
    for y, i in enumerate(idx[::-1]):
        ax.barh(y, fg[i] * 100, height=0.55, color=ORGAN_C[names[i]])
        ax.text(fg[i] * 100 + 0.8, y, f"{fg[i] * 100:.1f}% of organ voxels  ·  {share[i + 1] * 100:.2f}% of all",
                va="center", fontsize=10, color=INK if names[i] == "esophagus" else INK_2,
                fontweight="bold" if names[i] == "esophagus" else None)
    ax.set_yticks(range(4))
    for t in ax.set_yticklabels([names[i].capitalize() for i in idx[::-1]], fontsize=11):
        t.set_color(INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of organ voxels, % (32 training patients, native resolution)")
    ax.set_title(f"Background is {share[0] * 100:.1f}% of all voxels; the esophagus "
                 f"{share[1] * 100:.2f}% (1 in {round(1 / share[1])})", loc="left", fontsize=11, color=INK_2)
    save(fig, "puck_10_voxel_share")
    print("   voxel share:", dict(zip(["bg"] + names, np.round(share * 100, 3))))


def p11_loss_comparison():
    rows = [("CE (baseline)", "augmentation-native384", (42, 43, 44)), ("CE + Tversky", "loss_ce_tversky", (42, 43)),
            ("Focal + Dice", "loss_focal_dice", (42, 43)), ("CE + Dice", "loss_ce_dice", (42, 43))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (cls, lim, title, c) in zip(axes, ((None, (0.855, 0.885), "Mean Dice", BLUE),
                                               ("esophagus", (0.70, 0.79), "Esophagus Dice", ORANGE))):
        hgrid(ax)
        for y, (lab, e, seeds) in enumerate(rows[::-1]):
            vals = [repo(e, s).get("dice", cls) for s in seeds]
            ax.plot([min(vals), max(vals)], [y, y], color=c, linewidth=5, alpha=0.2, solid_capstyle="round")
            dot(ax, vals, [y] * len(vals), c, s=50)
            dot(ax, [np.mean(vals)], [y], c, s=150, z=4)
            ax.text(np.mean(vals), y + 0.3, f"{np.mean(vals):.3f}", ha="center", fontsize=10, color=INK_2)
        ax.set_xlim(*lim)
        ax.set_title(title, loc="left", fontsize=11, color=INK_2)
    axes[0].set_yticks(range(len(rows)))
    for t in axes[0].set_yticklabels([r[0] for r in rows[::-1]], fontsize=10.5):
        t.set_color(INK)
        if t.get_text() == "CE + Dice":
            t.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(rows) - 0.3)
    note(axes[0], "Big dot = mean, small dots = seeds. Post-processed, 8 holdout patients.", y=-0.22)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "puck_11_loss_comparison")


def p12_per_patient_esophagus():
    b = [repo("augmentation-native384", s) for s in (42, 43, 44)]
    d = [repo("loss_ce_dice", s) for s in (42, 43)]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    style_axes(ax)
    for p in HOLDOUT:
        y0 = np.mean([r.get("dice", "esophagus", p) for r in b])
        y1 = np.mean([r.get("dice", "esophagus", p) for r in d])
        ax.plot([0, 1], [y0, y1], color=ORANGE if y1 > y0 else RED, linewidth=2, alpha=0.8)
        dot(ax, [0, 1], [y0, y1], ORANGE, s=60)
        ax.text(1.05, y1, p.replace("Patient_", "P"), va="center", fontsize=9, color=INK_2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CE\n(3 seeds)", "CE + Dice\n(2 seeds)"], fontsize=11)
    ax.set_xlim(-0.3, 1.35)
    ax.set_ylabel("Esophagus Dice per patient")
    ax.set_title("Esophagus per holdout patient", loc="left", fontsize=11, color=INK_2)
    save(fig, "puck_12_per_patient_esophagus")


# ================================================================ DAVID
def d13_sampler_share():
    before = {"Empty": 36.6, "Other organs only": 10.1, "Esophagus": 53.3}
    w = {"Empty": 0.7, "Other organs only": 1.0, "Esophagus": 2.0}
    tot = sum(before[k] * w[k] for k in before)
    after = {k: before[k] * w[k] / tot * 100 for k in before}
    cols = {"Empty": AXIS, "Other organs only": MUTED, "Esophagus": ORANGE}
    fig, ax = plt.subplots(figsize=(8, 2.8))
    style_axes(ax, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    for y, (lab, d) in enumerate((("After: weighted\n(empty ×0.7, esophagus ×2)", after), ("Before: every slice once", before))):
        left = 0
        for k in ("Esophagus", "Other organs only", "Empty"):
            ax.barh(y, d[k], left=left, height=0.55, color=cols[k], edgecolor=SURFACE, linewidth=2)
            if d[k] > 6:
                ax.text(left + d[k] / 2, y, f"{d[k]:.0f}%", ha="center", va="center", fontsize=10.5,
                        color="white" if k != "Empty" else INK)
            left += d[k]
    ax.set_yticks([0, 1])
    for t in ax.set_yticklabels(["After: weighted sampler", "Before: each slice once"], fontsize=10.5):
        t.set_color(INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    legend(ax, [(k, cols[k]) for k in ("Esophagus", "Other organs only", "Empty")], kind="s", y=1.2)
    ax.set_title("Share of training slices per epoch", loc="left", fontsize=11, color=INK_2, pad=28)
    save(fig, "david_13_sampler_slice_share")


def d14_sampler_hd95():
    rows = [("CE + Dice", "loss_ce_dice"), ("CE + Dice + sampler", "combo_core_ce")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (metric, lim, lab) in zip(axes, (("dice", (0.86, 0.89), "Mean Dice"), ("hd95", (6, 11), "HD95, mm (lower is better)"))):
        hgrid(ax)
        for y, (n, e) in enumerate(rows[::-1]):
            vals = [repo(e, s).get(metric) for s in (42, 43)]
            c = BLUE if metric == "dice" else VIOLET
            dot(ax, vals, [y] * 2, c, s=60)
            dot(ax, [np.mean(vals)], [y], c, s=150, z=4)
            ax.text(np.mean(vals), y + 0.28, f"{np.mean(vals):.3f}" if metric == "dice" else f"{np.mean(vals):.1f}",
                    ha="center", fontsize=10.5, color=INK_2)
        ax.set_xlim(*lim)
        ax.set_title(lab, loc="left", fontsize=11, color=INK_2)
        ax.set_ylim(-0.6, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r[0] for r in rows[::-1]] if ax is axes[0] else ["", ""], fontsize=10.5)
    note(axes[0], "Big dot = mean of 2 seeds, small dots = seeds. Post-processed.", y=-0.22)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "david_14_sampler_fixes_hd95")


# ================================================================ JUNIS
def j15_width_curve():
    const = [repo("combo_core_ce", s) for s in (42, 43)], [repo("combo_core_ce_k16", s) for s in (42, 43)], \
            [repo("combo_core_ce_k32", s) for s in (42, 43)]
    cos = [repo("combo_core_ce_cosine50", s) for s in (42, 43)], final_runs(), \
          [wb(f"combo_core_ce_k32_cosine50/holdout40-f0-s{s}") for s in (42, 43)]
    xs = [0.28, 1.12, 4.44]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cls, title, lim in ((axes[0], None, "Mean Dice", (0.875, 0.9)), (axes[1], "esophagus", "Esophagus Dice", (0.75, 0.81))):
        style_axes(ax)
        for runs, c, sign in ((const, MUTED, -1), (cos, BLUE if cls is None else ORANGE, 1)):
            ys = [mean_of(r, metric="dice", cls=cls) for r in runs]
            ax.plot(xs, ys, color=c, linewidth=2)
            dot(ax, xs, ys, c)
            for x, y in zip(xs, ys):
                ax.text(x, y + sign * (lim[1] - lim[0]) * 0.035, f"{y:.3f}" if abs(y - 0.8945) > 1e-4 else "0.8945",
                        ha="center", va="bottom" if sign > 0 else "top", fontsize=9.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["k8\n0.28M", "k16\n1.1M", "k32\n4.4M"])
        ax.minorticks_off()
        ax.set_ylim(*lim)
        ax.set_title(title, loc="left", fontsize=11, color=INK_2)
    legend(axes[0], [("25 epochs, constant lr", MUTED), ("50 epochs, cosine", BLUE)], y=1.2)
    note(axes[0], "ENet width (kernels, parameters). CE + Dice + sampler; post-processed; 2 seeds each.", y=-0.25)
    fig.subplots_adjust(wspace=0.2)
    save(fig, "junis_15_width_curve")


def j16_transformer():
    k32 = [repo("combo_core_ce_k32", s) for s in (42, 43)]
    rows = [("Plain ENet k32 (2 seeds)", k32), ("+ transformer, stage 2, pos. emb.", [repo("tf_st2_combo_core_ce_k32_res_pos")]),
            ("+ transformer, stage 2, no pos. emb.", [repo("tf_st2_combo_core_ce_k32_res_nopos")]),
            ("+ transformer, no resampling", [repo("tf_st2_combo_core_ce_k32_nores")]),
            ("+ transformer + pos. emb., no resampling", [repo("tf_st2_combo_core_ce_k32_nores_pos")])]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    hgrid(ax)
    ref = mean_of(k32, metric="dice")
    ax.axvline(ref, color=MUTED, linewidth=1.5)
    for y, (lab, rs) in enumerate(rows[::-1]):
        v = mean_of(rs, metric="dice")
        dot(ax, [v], [y], BLUE if y == len(rows) - 1 else (RED if v < ref - 0.01 else MUTED))
        ax.text(v + 0.002, y, f"{v:.3f}  ({v - ref:+.3f})" if y != len(rows) - 1 else f"{v:.3f}", va="center",
                fontsize=10, color=INK_2)
    ax.set_yticks(range(len(rows)))
    for t in ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=10.5):
        t.set_color(INK)
    ax.set_xlim(0.86, 0.905)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Mean Dice (post-processed); all with CE + Dice + sampler, 25 epochs")
    ax.set_title("A transformer layer adds nothing once ENet is wide enough", loc="left", fontsize=11, color=INK_2)
    save(fig, "junis_16_transformer")


def j17_val_curves():
    H = WB["histories"]
    pairs = [("combo_core_ce_k16", "25 epochs, constant lr", MUTED), ("combo_core_ce_k16_cosine50", "50 epochs, cosine", BLUE)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    style_axes(ax)
    for e, lab, c in pairs:
        for s in (42, 43):
            h = H[f"{e}/holdout40-f0-s{s}"]
            ep = [r["epoch"] for r in h]
            v = [r["val_dice"] for r in h]
            ax.plot(ep, v, color=c, linewidth=2, alpha=0.95 if s == 42 else 0.55)
            b = int(np.argmax(v))
            dot(ax, [ep[b]], [v[b]], c, s=60)
    k = [r["val_dice"] for r in H["combo_core_ce_k16/holdout40-f0-s42"]]
    ax.annotate(f"best epoch {max(k):.3f}\nlast epoch {k[-1]:.3f}", xy=(24, k[-1]), xytext=(27.5, 0.835), fontsize=10,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_ylim(0.80, 0.90)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice (2D, per epoch)")
    legend(ax, [(l, c) for _, l, c in pairs], y=1.12)
    note(ax, "ENet k16, CE + Dice + sampler. Two seeds per schedule (lighter = seed 43); dot = best epoch.", y=-0.2)
    save(fig, "junis_17_val_curves_constant_vs_cosine")


def j18_lr_schedules():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    style_axes(ax)
    e25, e50 = np.arange(25), np.arange(50)
    ax.plot(e25, np.full(25, 5e-4), color=MUTED, linewidth=2, label="Constant, 25 epochs")
    ax.plot(e25, 2.5e-4 * (1 + np.cos(np.pi * e25 / 25)), color=RED, linewidth=2, label="Cosine, 25 epochs (−0.012)")
    ax.plot(e50, 2.5e-4 * (1 + np.cos(np.pi * e50 / 50)), color=BLUE, linewidth=2, label="Cosine, 50 epochs (final)")
    ax.set_ylim(0, 5.6e-4)
    ax.set_yticks([0, 2.5e-4, 5e-4])
    ax.set_yticklabels(["0", "2.5e-4", "5e-4"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    lg = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, borderaxespad=0)
    for t in lg.get_texts():
        t.set_color(INK_2)
    save(fig, "junis_18_lr_schedules")


# ================================================================ BRITT
def b21_hd95_raw_vs_post_models():
    rows = [("CE baseline (k8)", [repo("augmentation-native384", s) for s in (42, 43, 44)]),
            ("CE + Dice + sampler (k8)", [repo("combo_core_ce", s) for s in (42, 43)]),
            ("+ k16", [repo("combo_core_ce_k16", s) for s in (42, 43)]),
            ("+ k16, 50 ep. cosine = final", final_runs()),
            ("k32", [repo("combo_core_ce_k32", s) for s in (42, 43)]),
            ("k32, 50 ep. cosine", [wb(f"combo_core_ce_k32_cosine50/holdout40-f0-s{s}") for s in (42, 43)])]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    hgrid(ax)
    for y, (lab, rs) in enumerate(rows[::-1]):
        raw, post = mean_of(rs, metric="hd95", key="metrics_3d"), mean_of(rs, metric="hd95")
        ax.plot([post, raw], [y, y], color=AXIS, linewidth=2, zorder=1)
        dot(ax, [raw], [y], MUTED)
        dot(ax, [post], [y], BLUE)
        ax.text(max(raw, post) + 0.5, y, f"{raw:.1f} → {post:.1f}", va="center", fontsize=10, color=INK_2)
    ax.set_yticks(range(len(rows)))
    for t in ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=10.5):
        t.set_color(INK)
        if "final" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("Mean HD95, mm (lower is better)")
    legend(ax, [("Raw", MUTED), ("After post-processing", BLUE)], y=1.12)
    save(fig, "britt_21_hd95_raw_vs_post_per_model")


def b22_postprocessing_variants():
    def two(e):
        return float(np.mean([repo(e, s).get("hd95", key="metrics_3d") for s in (42, 43)])), \
               float(np.mean([repo(e, s).get("dice", key="metrics_3d") for s in (42, 43)]))
    weak = [("Keep ≥10% of largest (kept)", "post_native_lcc_min_fraction"), ("+ fill holes", "post_native_lcc_fill_holes"),
            ("Largest + within 20 mm", "post_native_lcc_distance"), ("Close along z + ≥10%", "post_native_close_lcc"),
            ("Close + 20 mm + fill", "post_native_close_distance_fill")]
    strong = [("Keep ≥10% of largest (kept)", [repo("combo_core_ce", s).get("hd95") for s in (42, 43)]),
              ("Close + 20 mm + fill", [repo("post_combo_core_ce_close_distance_fill", s).get("hd95", key="metrics_3d") for s in (42, 43)])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    hgrid(ax)
    ref = two(weak[0][1])[0]
    for y, (lab, e) in enumerate(weak[::-1]):
        h, _ = two(e)
        dot(ax, [h], [y], BLUE if h < ref - 0.3 else MUTED)
        ax.text(h + 0.15, y, f"{h:.1f}", va="center", fontsize=10, color=INK_2)
    ax.axvline(ref, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(weak)))
    for t in ax.set_yticklabels([w[0] for w in weak[::-1]], fontsize=10):
        t.set_color(INK)
    ax.set_xlim(9, 13.5)
    ax.set_title("Weak model (native, no augmentation): helps", loc="left", fontsize=11, color=INK_2)
    ax.set_xlabel("Mean HD95, mm")
    ax = axes[1]
    hgrid(ax)
    for y, (lab, vals) in enumerate(strong[::-1]):
        h = float(np.mean(vals))
        dot(ax, [h], [y], RED if y == 0 else MUTED)
        ax.text(h + 0.05, y, f"{h:.2f}", va="center", fontsize=10, color=INK_2)
    ax.set_yticks([0, 1])
    for t in ax.set_yticklabels([s[0] for s in strong[::-1]], fontsize=10):
        t.set_color(INK)
    ax.set_xlim(8, 9.3)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Stronger model (CE + Dice + sampler): hurts", loc="left", fontsize=11, color=INK_2)
    ax.set_xlabel("Mean HD95, mm")
    note(axes[0], "Mean of 2 seeds each. Same predictions, only the post-processing differs per row.", y=-0.25)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "britt_22_postprocessing_variants")


def b23_one_seed_lies():
    rows = [("Keep ≥10% of largest", "post_native_lcc_min_fraction"), ("Skip the esophagus", "post_native_lcc_skip_esophagus")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    style_axes(ax)
    bw = 0.32
    for i, s in enumerate((42, 43)):
        for j, (lab, e) in enumerate(rows):
            v = repo(e, s).get("hd95", "esophagus", key="metrics_3d")
            x = i + (j - 0.5) * (bw + 0.02)
            rounded_bar(ax, x, bw, v, BLUE if j == 0 else ORANGE)
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=10.5, color=INK_2)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 78)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Seed 42", "Seed 43"], fontsize=11)
    ax.set_ylabel("Esophagus HD95, mm")
    legend(ax, [(r[0], c) for r, c in zip(rows, (BLUE, ORANGE))], kind="s", y=1.14)
    note(ax, "Same native-resolution model, two seeds. Seed 42 alone would have picked the wrong rule.", y=-0.2)
    save(fig, "britt_23_one_seed_can_lie")


def b24_stray_blob_projection():
    """Coronal maximum-intensity projection of a predicted volume: raw vs post-processed (native model, seed 43)."""
    raw_dir = ROOT / "results/spacing_native_384/holdout40-f0-s43/best_epoch/val"
    post_dir = ROOT / "results/post_native_lcc_min_fraction/holdout40-f0-s43/best_epoch/val"
    rr = repo("spacing_native_384", 43)
    worst = max(((p, o, rr.get("hd95", o, p, key="metrics_3d")) for p in HOLDOUT for o in ORGANS), key=lambda t: t[2])
    p = worst[0]

    def vol(d):
        fs = sorted(glob.glob(str(d / f"{p}_*.png")))
        return np.stack([np.array(Image.open(f)) // 63 for f in fs])

    v_raw = vol(raw_dir)
    v_gt = np.stack([np.array(Image.open(f)) // 63 for f in sorted(glob.glob(str(CACHE_N / "gt" / f"{p}_*.png")))])
    from segpipe.postprocess import build_postprocess
    pp = build_postprocess([{"name": "largest_cc", "min_fraction": 0.1}])
    try:
        v_post = pp(v_raw.copy(), (2.0, 0.98, 0.98))
    except TypeError:
        v_post = pp(v_raw.copy())
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
    for ax, v, t in ((axes[0], v_gt, "Ground truth"), (axes[1], v_raw, f"Raw prediction ({worst[1]} HD95 {worst[2]:.0f} mm)"),
                     (axes[2], v_post, "After post-processing")):
        rgb = np.full(v.shape[::2][::-1] if False else (v.shape[0], v.shape[2], 3), 1.0)
        rgb = np.ones((v.shape[0], v.shape[2], 3))
        for o in ["heart", "aorta", "trachea", "esophagus"]:  # paint order: esophagus on top
            k = ORGANS.index(o) + 1
            m = (v == k).any(axis=2)  # (z, x): collapse the anterior-posterior axis -> coronal view
            col = np.array([int(ORGAN_C[o][i:i + 2], 16) / 255 for i in (1, 3, 5)])
            rgb[m] = col
        ax.imshow(rgb, aspect=2.0 / 0.98, origin="lower")
        ax.set_title(t, fontsize=10.5, color=INK, loc="left")
        ax.axis("off")
    legend(axes[0], [(o.capitalize(), ORGAN_C[o]) for o in ORGANS], kind="s", y=-0.02, ncol=4)
    fig.suptitle(f"{p}, native-resolution model (seed 43): coronal projection of the 3D prediction", x=0.01, ha="left",
                 fontsize=11, color=INK_2)
    fig.subplots_adjust(wspace=0.03, bottom=0.08)
    save(fig, "britt_24_stray_blob_projection")
    print("   stray blob case:", worst)


# ================================================================ PART 5
def f25_final_per_organ():
    mid = repo("current")
    ce = [repo("augmentation-native384", s) for s in (42, 43, 44)]
    fin = final_runs()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    hgrid(ax)
    for y, o in enumerate(ORGANS[::-1]):
        a = mid.get("dice", o, key="metrics_3d")
        b = mean_of(ce, metric="dice", cls=o)
        c = mean_of(fin, metric="dice", cls=o)
        ax.plot([min(a, b, c), max(a, b, c)], [y, y], color=AXIS, linewidth=2, zorder=1)
        dot(ax, [a], [y], AXIS)
        dot(ax, [b], [y], MUTED)
        dot(ax, [c], [y], ORANGE if o == "esophagus" else BLUE)
        ax.text(1.02, y, f"{a:.3f} → {c:.3f}  ({c - a:+.3f})", va="center", ha="left", fontsize=10,
                transform=ax.get_yaxis_transform(),
                color=INK if o == "esophagus" else INK_2, fontweight="bold" if o == "esophagus" else None)
    ax.set_yticks(range(4))
    for t in ax.set_yticklabels([o.capitalize() for o in ORGANS[::-1]], fontsize=11):
        t.set_color(INK)
    ax.set_xlim(0.5, 0.97)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("3D Dice")
    fig.subplots_adjust(right=0.72)
    legend(ax, [("Midterm pipeline", AXIS), ("CE baseline", MUTED), ("Final model", BLUE)], y=1.14)
    save(fig, "final_25_per_organ_midterm_baseline_final")


def f26_per_patient():
    mid = repo("current")
    fin = final_runs()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    style_axes(ax)
    ends = []
    for p in HOLDOUT:
        a = mid.pmean(p, key="metrics_3d")
        b = float(np.mean([r.pmean(p) for r in fin]))
        ax.plot([0, 1], [a, b], color=BLUE, linewidth=2, alpha=0.7)
        dot(ax, [0, 1], [a, b], BLUE, s=60)
        ends.append([b, b, f"{p.replace('Patient_', 'P')}  +{b - a:.2f}"])
    ends.sort(key=lambda t: t[0])
    for i in range(1, len(ends)):  # push labels apart: at least 0.0075 Dice between neighbours
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.0075)
    for b, yl, txt in ends:
        ax.plot([1.01, 1.05], [b, yl], color=AXIS, linewidth=0.8)
        ax.text(1.06, yl, txt, va="center", fontsize=9.5, color=INK_2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Midterm\npipeline", "Final\nmodel"], fontsize=11)
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylabel("Mean Dice per patient")
    ax.set_title("Better in all 8 holdout patients", loc="left", fontsize=11, color=INK_2)
    save(fig, "final_26_per_patient_midterm_vs_final")


def f27_cv_per_patient():
    runs = {0: wb("combo_core_ce_k16_cosine50/holdout40-f0-s42")}
    for f in (1, 2, 3, 4):
        runs[f] = wb(f"combo_core_ce_k16_cosine50/cv5_40-f{f}-s42")
    pts = []
    for f, r in runs.items():
        ps = sorted({k.split("/")[3] for k in r.d if k.startswith("metrics_3d_post/dice/per_patient/")})
        pts += [(f, p, r.pmean(p), r.get("dice", "esophagus", p)) for p in ps]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, i, title, c in ((axes[0], 2, "Mean Dice per patient", BLUE), (axes[1], 3, "Esophagus Dice per patient", ORANGE)):
        style_axes(ax)
        for f in range(5):
            ys = [t[i] for t in pts if t[0] == f]
            xs = f + rng.uniform(-0.15, 0.15, len(ys))
            dot(ax, xs, ys, MUTED if f == 0 else c, s=55)
            ax.plot([f - 0.25, f + 0.25], [np.mean(ys)] * 2, color=INK, linewidth=2)
        worst = sorted(pts, key=lambda t: t[i])[:3]
        for f, p, *_ in worst:
            y = [t for t in pts if t[1] == p][0][i]
            ax.text(f + 0.22, y, p.replace("Patient_", "P"), fontsize=9, color=INK_2, va="center")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["Fold 0\n(holdout)", "Fold 1", "Fold 2", "Fold 3", "Fold 4"])
        ax.set_title(title, loc="left", fontsize=11, color=INK_2)
    axes[0].set_ylim(0.70, 0.95)
    axes[1].set_ylim(0.45, 0.90)
    note(axes[0], "40 patients, final model, post-processed. Line = fold mean. Grey = the 8 patients we tuned on.", y=-0.24)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "final_27_cv_per_patient")


def f28_starter_vs_final_cv():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hgrid(ax)
    for y, f in enumerate((4, 3, 2, 1)):
        s = wb(f"full_data_baseline/cv5_40-f{f}-s42").get("dice", key="metrics_3d")
        e = wb(f"combo_core_ce_k16_cosine50/cv5_40-f{f}-s42").get("dice", key="metrics_3d")
        ax.plot([s, e], [y, y], color=AXIS, linewidth=2, zorder=1)
        dot(ax, [s], [y], MUTED)
        dot(ax, [e], [y], BLUE)
        ax.text(e + 0.006, y, f"{s:.3f} → {e:.3f}  (+{e - s:.3f})", va="center", fontsize=10, color=INK_2)
    ax.set_yticks(range(4))
    for t in ax.set_yticklabels(["Fold 4", "Fold 3", "Fold 2", "Fold 1"], fontsize=11):
        t.set_color(INK)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.6, 3.5)
    ax.set_xlabel("3D Dice (raw), 8 unseen patients per fold")
    legend(ax, [("Starter code", MUTED), ("Final model", BLUE)], y=1.14)
    save(fig, "final_28_starter_vs_final_cv")


def f29_cv_hd95_raw_vs_post():
    rs = [wb(f"combo_core_ce_k16_cosine50/cv5_40-f{f}-s42") for f in (1, 2, 3, 4)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hgrid(ax)
    for y, o in enumerate(ORGANS[::-1]):
        raw = mean_of(rs, metric="hd95", cls=o, key="metrics_3d")
        post = mean_of(rs, metric="hd95", cls=o)
        ax.annotate("", xy=(post, y), xytext=(raw, y), arrowprops=dict(arrowstyle="-|>", color=RED if post > raw else BLUE,
                                                                       linewidth=2, mutation_scale=14))
        dot(ax, [raw], [y], MUTED)
        ax.text(max(raw, post) + 0.4, y, f"{raw:.1f} → {post:.1f}", va="center", fontsize=10,
                color=RED if post > raw else INK_2, fontweight="bold" if post > raw else None)
    ax.set_yticks(range(4))
    for t in ax.set_yticklabels([o.capitalize() for o in ORGANS[::-1]], fontsize=11):
        t.set_color(INK)
    ax.set_xlim(5, 24)
    ax.set_ylim(-0.6, 3.5)
    ax.set_xlabel("HD95, mm, CV folds 1–4 (grey dot = raw, arrow = after post-processing)")
    ax.set_title("On unseen patients the filter removes real esophagus pieces", loc="left", fontsize=11, color=INK_2)
    save(fig, "final_29_cv_hd95_raw_vs_post")


def f30_pipeline():
    steps = ["HU window\n[40, 400]", "Resample\n0.98 mm\ncrop 384²", "ENet k16\n(1.1M)", "GPU\naugmentation",
             "CE + Dice\n+ weighted\nsampler", "50 epochs\ncosine lr", "Keep 3D\ncomponents\n≥10%"]
    owner = ["Elena", "Elena", "Junis", "Githa", "Puck · David", "Junis", "Britt"]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (s, o) in enumerate(zip(steps, owner)):
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.25), 0.78, 0.6, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="#eef3fb", edgecolor=BLUE, linewidth=1.5))
        ax.text(i + 0.47, 0.55, s, ha="center", va="center", fontsize=10.5, color=INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.08, 0.55), xytext=(i + 0.86, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=AXIS, linewidth=1.5))
    save(fig, "final_30_pipeline_diagram")


def main():
    for f in (e01_midterm_per_organ, e02_ct_slices, e03_esophagus_resolution, e04_preprocessing_ablation,
              e05_resolution_per_organ, e06_qualitative_midterm_vs_native, g07_augmentation_2x2, g08_seed_spread,
              g09_augmentation_examples, p10_pixel_share, p11_loss_comparison, p12_per_patient_esophagus,
              d13_sampler_share, d14_sampler_hd95, j15_width_curve, j16_transformer, j17_val_curves, j18_lr_schedules,
              b21_hd95_raw_vs_post_models, b22_postprocessing_variants, b23_one_seed_lies, b24_stray_blob_projection,
              f25_final_per_organ, f26_per_patient, f27_cv_per_patient, f28_starter_vs_final_cv,
              f29_cv_hd95_raw_vs_post, f30_pipeline):
        try:
            f()
        except Exception as e:  # keep going; report at the end
            import traceback
            print("FAILED", f.__name__, repr(e))
            traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()
