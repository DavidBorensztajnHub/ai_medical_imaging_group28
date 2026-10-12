"""Figures for the final presentation: the ladder (slides 4, 5, 7, 9, 12) and the cross-validation (slide 13).

    ai4mi/bin/python figures/make_presentation_figs.py

Writes PNG (for the slides) and SVG (scalable) to figures/presentation/. Numbers are the frozen values from
PRESENTATION_NOTES.md section 1 (3D Dice on the 8 holdout patients, post-processed, mean of seeds; the
first three ladder steps are raw). Colours: blue = mean Dice, orange = esophagus, on every figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).parent / "presentation"

# Reference palette (dataviz skill), light mode
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"    # series 1: mean Dice
ORANGE = "#eb6834"  # series 2: esophagus

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
})

# label, mean Dice, esophagus Dice, raw?
LADDER = [
    ("Starter\ncode", 0.703, 0.468, True),
    ("Midterm\npipeline", 0.787, 0.561, True),
    ("+ native\nresolution", 0.835, 0.659, True),
    ("+ augmentation\n= CE baseline", 0.866, 0.724, False),
    ("+ Dice term\n+ sampler", 0.880, 0.766, False),
    ("+ wider ENet,\n50 ep. cosine\n= final", 0.8945, 0.793, False),
]

# HD95 (mm, mean over organs) per ladder step, post-processed wherever a post-processed run exists:
# starter raw (no post-processed holdout run), midterm and native from the post_* rescoring runs.
LADDER_HD95 = [(23.3, True), (11.2, False), (12.5, False), (8.2, False), (8.4, False), (6.4, False)]

# Britt, slide 10: the final model's stray blobs, raw vs post-processed (case, Dice raw, post, HD95 raw, post)
BLOBS = [
    ("Patient_22 · trachea\n(seed 43)", 0.887, 0.899, 271.4, 5.0),
    ("Patient_13 · trachea\n(seed 43)", 0.911, 0.923, 182.6, 6.0),
    ("Patient_30 · aorta\n(seed 42)", 0.929, 0.940, 82.3, 2.7),
]
# Britt, slide 11: mean HD95 (mm) raw vs post-processed, CE baseline (3 seeds) vs + boundary loss weight 5 (2 seeds)
BOUNDARY = [("CE baseline", 15.1, 8.2), ("CE + boundary loss\n(weight 5)", 8.7, 9.6)]
VIOLET = "#4a3aa7"  # series 7: HD95 panel of the ladder (a different measure, so not blue/orange)

CV_FOLDS = [("Fold 1", 0.8581), ("Fold 2", 0.8635), ("Fold 3", 0.8846), ("Fold 4", 0.8579)]
CV_MEAN, CV_STD, HOLDOUT = 0.866, 0.011, 0.8945
# organ, holdout (2 seeds), CV folds 1-4
ORGANS = [("Esophagus", 0.793, 0.746), ("Trachea", 0.902, 0.870), ("Aorta", 0.938, 0.908), ("Heart", 0.945, 0.940)]


def rounded_bar(ax, x, width, height, color, radius_px=4, alpha=1.0):
    """Column from the baseline with a rounded top (radius in pixels) and a square foot."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = radius_px * (x1 - x0) / bbox.width
    ry = radius_px * (y1 - y0) / bbox.height
    rx, ry = min(rx, width / 2), min(ry, height)
    left, right = x - width / 2, x + width / 2
    k = 0.5523  # bezier circle constant
    verts = [
        (left, 0), (left, height - ry),
        (left, height - ry + k * ry), (left + rx - k * rx, height), (left + rx, height),
        (right - rx, height),
        (right - rx + k * rx, height), (right, height - ry + k * ry), (right, height - ry),
        (right, 0), (left, 0),
    ]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def style_axes(ax, ygrid=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def legend_key(ax, items, loc="upper left"):
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, markerfacecolor=c,
                          markeredgecolor="none", label=l) for l, c in items]
    leg = ax.legend(handles=handles, loc=loc, frameon=False, ncol=len(items), handletextpad=0.4,
                    columnspacing=1.4, bbox_to_anchor=(0, 1.08), borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)


def ladder(upto: int | None, name: str, size=(10, 5.6)):
    """upto = index of the current step (earlier steps faded, later steps empty); None = full ladder."""
    fig, ax = plt.subplots(figsize=size)
    n = len(LADDER)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.0)
    style_axes(ax)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("3D Dice (8 holdout patients)")
    bar_w, gap = 0.34, 0.02  # data units; gap ~2 px at this size
    last = n - 1 if upto is None else upto
    for i, (label, mean, eso, raw) in enumerate(LADDER):
        if i > last:
            continue
        current = upto is None or i == upto
        alpha = 1.0 if current else 0.35
        rounded_bar(ax, i - bar_w / 2 - gap / 2, bar_w, mean, BLUE, alpha=alpha)
        rounded_bar(ax, i + bar_w / 2 + gap / 2, bar_w, eso, ORANGE, alpha=alpha)
        if current:  # values only on the step(s) being presented
            for xo, v in ((-bar_w / 2 - gap / 2, mean), (bar_w / 2 + gap / 2, eso)):
                ax.text(i + xo, v + 0.012, f"{v:.4f}" if v == 0.8945 else f"{v:.3f}", ha="center",
                        va="bottom", fontsize=10 if upto is None else 11, color=INK_2)
    ax.set_xticks(range(n))
    labels = ax.set_xticklabels([l for l, *_ in LADDER], fontsize=10)
    for i, t in enumerate(labels):
        if i > last:
            t.set_color(GRID)
        elif upto is not None and i == upto:
            t.set_color(INK)
            t.set_fontweight("bold")
    legend_key(ax, [("Mean Dice (4 organs)", BLUE), ("Esophagus", ORANGE)])
    ax.text(0, -0.30, "First three steps raw (no post-processing; it adds ≤ 0.003 Dice). "
            "Others post-processed, mean of 2–3 seeds.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, name)


def ladder_with_hd95(name: str):
    """Slide 12: Dice ladder on top, HD95 per step below (same steps; separate axes, no dual axis)."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.7, 1], "hspace": 0.12})
    n = len(LADDER)
    for ax in (top, bot):
        ax.set_xlim(-0.6, n - 0.4)
        style_axes(ax)
    top.set_ylim(0, 1.0)
    top.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    top.set_ylabel("3D Dice\n(higher is better)")
    bar_w, gap = 0.34, 0.02
    for i, (_, mean, eso, _) in enumerate(LADDER):
        for xo, v, c in ((-bar_w / 2 - gap / 2, mean, BLUE), (bar_w / 2 + gap / 2, eso, ORANGE)):
            rounded_bar(top, i + xo, bar_w, v, c)
            top.text(i + xo, v + 0.012, f"{v:.4f}" if v == 0.8945 else f"{v:.3f}", ha="center", va="bottom",
                     fontsize=9.5, color=INK_2)
    legend_key(top, [("Mean Dice (4 organs)", BLUE), ("Esophagus", ORANGE)])

    bot.set_ylim(0, 27)
    bot.set_yticks([0, 5, 10, 15, 20, 25])
    bot.set_ylabel("HD95, mm\n(lower is better)")
    for i, (v, raw) in enumerate(LADDER_HD95):
        rounded_bar(bot, i, 0.45, v, VIOLET, alpha=0.45 if raw else 1.0)
        bot.text(i, v + 0.5, f"{v:.1f}" + (" (raw)" if raw else ""), ha="center", va="bottom", fontsize=9.5,
                 color=INK_2)
    bot.set_xticks(range(n))
    labels = bot.set_xticklabels([l for l, *_ in LADDER], fontsize=10)
    labels[-1].set_fontweight("bold")
    bot.text(0, -0.62, "Dice: first three steps raw (post-processing adds ≤ 0.003). HD95: post-processed for every "
             "step except the starter code (no post-processed run). Mean of 1–3 seeds.",
             transform=bot.transAxes, fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.17)
    save(fig, name)


def blobs(name: str):
    """Slide 10 (Britt): for the 3 stray-blob cases, Dice barely moves while HD95 collapses."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.08})
    rows = list(range(len(BLOBS)))[::-1]
    for ax, lo, hi, ticks, title in ((left, 0.80, 1.00, [0.80, 0.85, 0.90, 0.95, 1.00], "Dice (organ)"),
                                     (right, 0, 300, [0, 50, 100, 150, 200, 250, 300], "HD95, mm (organ)")):
        style_axes(ax, ygrid=False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.set_title(title, loc="left", fontsize=11, color=INK_2, pad=8)
    for y, (case, d_raw, d_post, h_raw, h_post) in zip(rows, BLOBS):
        left.plot([d_raw, d_post], [y, y], color=AXIS, linewidth=2, zorder=1)
        left.scatter([d_raw], [y], s=110, color=MUTED, edgecolors=SURFACE, linewidths=2, zorder=3)
        left.scatter([d_post], [y], s=110, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        left.text(0.805, y + 0.28, f"{d_raw:.3f} → {d_post:.3f}  (+{d_post - d_raw:.3f})", fontsize=9.5,
                  color=INK_2, va="center")
        right.plot([h_post, h_raw], [y, y], color=AXIS, linewidth=2, zorder=1)
        right.scatter([h_raw], [y], s=110, color=MUTED, edgecolors=SURFACE, linewidths=2, zorder=3)
        right.scatter([h_post], [y], s=110, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        right.text(h_raw + 8, y, f"{h_raw:.0f} → {h_post:.0f} mm", fontsize=10, color=INK, va="center",
                   fontweight="bold")
    left.set_yticks(rows)
    for t in left.set_yticklabels([c for c, *_ in BLOBS], fontsize=10):
        t.set_color(INK)
    left.set_ylim(-0.6, len(BLOBS) - 0.3)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=c,
                          markeredgecolor=SURFACE, markeredgewidth=2, label=l)
               for l, c in (("Raw prediction", MUTED), ("After post-processing (keep components ≥ 10%)", BLUE))]
    leg = left.legend(handles=handles, loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.3),
                      borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK_2)
    right.text(1, -0.16, "Final model, holdout. Whole model: Dice 0.893 → 0.8945, HD95 16.0 → 6.4 mm.",
               transform=right.transAxes, ha="right", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.14)
    save(fig, name)


def boundary(name: str):
    """Slide 11 (Britt): the boundary loss removes blobs during training, but post-processing does it better."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style_axes(ax)
    ax.set_ylim(0, 17)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylabel("Mean HD95, mm (lower is better)")
    ax.set_xlim(-0.6, len(BOUNDARY) - 0.4)
    bar_w, gap = 0.32, 0.02
    for i, (label, raw, post) in enumerate(BOUNDARY):
        for xo, v, c, a in ((-bar_w / 2 - gap / 2, raw, MUTED, 0.6), (bar_w / 2 + gap / 2, post, BLUE, 1.0)):
            rounded_bar(ax, i + xo, bar_w, v, c, alpha=a)
            ax.text(i + xo, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK_2)
    ax.set_xticks(range(len(BOUNDARY)))
    for t in ax.set_xticklabels([b for b, *_ in BOUNDARY], fontsize=10.5):
        t.set_color(INK)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, markerfacecolor=c, alpha=a,
                          markeredgecolor="none", label=l)
               for l, c, a in (("Raw prediction", MUTED, 0.6), ("After post-processing", BLUE, 1.0))]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.1),
                    borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.text(0, -0.22, "CE baseline: 3 seeds. Boundary loss (per organ, weight 5): 2 seeds.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, name)


def cv_folds(ax):
    style_axes(ax)
    ax.set_ylim(0.80, 0.92)
    ax.set_yticks([0.80, 0.83, 0.86, 0.89, 0.92])
    ax.set_xlim(-0.5, len(CV_FOLDS) - 0.5)
    # reference lines: holdout (muted) and CV mean (series colour)
    ax.axhline(HOLDOUT, color=MUTED, linewidth=2, zorder=1)
    ax.axhspan(CV_MEAN - CV_STD, CV_MEAN + CV_STD, color=BLUE, alpha=0.10, linewidth=0, zorder=0)
    ax.axhline(CV_MEAN, color=BLUE, linewidth=2, zorder=1)
    xs = range(len(CV_FOLDS))
    ys = [v for _, v in CV_FOLDS]
    ax.scatter(xs, ys, s=110, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y - 0.0075, f"{y:.3f}", ha="center", va="top", fontsize=10, color=INK_2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f for f, _ in CV_FOLDS])
    ax.text(len(CV_FOLDS) - 0.5, HOLDOUT + 0.002, f"Holdout (8 patients we tuned on): {HOLDOUT:.4f}",
            ha="right", va="bottom", fontsize=10, color=INK_2)
    ax.text(-0.45, CV_MEAN + CV_STD + 0.002, f"CV mean: {CV_MEAN:.3f} ± {CV_STD:.3f}",
            ha="left", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("3D Dice, final model")
    ax.set_title("8 unseen patients per fold", loc="left", fontsize=11, color=INK_2, pad=10)


def cv_organs(ax):
    style_axes(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0.70, 0.98)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    rows = list(range(len(ORGANS)))[::-1]
    for y, (organ, hold, cv) in zip(rows, ORGANS):
        ax.plot([cv, hold], [y, y], color=AXIS, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([hold], [y], s=110, color=MUTED, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.scatter([cv], [y], s=110, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(0.975, y, f"{cv - hold:+.3f}", ha="right", va="center", fontsize=10.5,
                color=INK if organ == "Esophagus" else INK_2, fontweight="bold" if organ == "Esophagus" else None)
    ax.set_yticks(rows)
    tl = ax.set_yticklabels([o for o, *_ in ORGANS], fontsize=11)
    tl[0].set_fontweight("bold")
    for t in tl:
        t.set_color(INK)
    ax.set_ylim(-0.6, len(ORGANS) - 0.4)
    ax.set_xlabel("3D Dice")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=c,
                          markeredgecolor=SURFACE, markeredgewidth=2, label=l)
               for l, c in (("Holdout (2 seeds)", MUTED), ("Cross-validation (folds 1–4)", BLUE))]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.13),
                    borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK_2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main():
    ladder(None, "ladder_full")
    for i, slide in [(2, "slide04_native"), (3, "slide05_augmentation"), (4, "slide07_loss_sampler"),
                     (5, "slide09_final")]:
        ladder(i, f"ladder_{slide}")
    ladder_with_hd95("slide12_ladder_dice_hd95")
    blobs("slide10_stray_blobs")
    boundary("slide11_boundary_loss")

    fig, ax = plt.subplots(figsize=(6, 4.6))
    cv_folds(ax)
    fig.tight_layout()
    save(fig, "cv_folds")

    fig, ax = plt.subplots(figsize=(6, 4.6))
    cv_organs(ax)
    fig.tight_layout()
    save(fig, "cv_organs")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    cv_folds(a)
    cv_organs(b)
    fig.tight_layout(w_pad=4)
    save(fig, "slide13_cv")


if __name__ == "__main__":
    main()
