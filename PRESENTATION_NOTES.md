# Final presentation — notes

Working notes for the final presentation (13 / 15 Oct, 10 min + 3 min questions). Taken from
RESULTS.md, W&B (`ai4migroup28/ai4mi-segthor`), DECISIONS.md, the configs and the git log as of
2026-10-09. Numbers are 3D on the native GT grid, `holdout40` split (32 train / 8 val patients),
**after post-processing** (`metrics_3d_post`, keep CCs ≥10% of the largest) and **averaged over 2 seeds
(42, 43)** unless noted.

**Rule from the brief:** the numbers on the slides must be the numbers in the submission. Freeze one
final config and one results table before making the result slides.

---

## 1. Where the midterm left off (≈ 21 Sep)

What we had shown then (on the 20-patient part-1 data, 15 train / 5 val):

- GT repair: aorta split out of the esophagus label (watershed, `watershed_refined`), affine restored.
- HU window [40, 400] instead of per-volume min-max.
- Resampling to 1.95 × 1.95 × 2.5 mm with crop/pad instead of per-slice resize.
- 3D metrics (Dice, HD95, ASSD, NSD) on the stitched volumes.
- First tries of augmentation, 2.5D and transformer-in-ENet, all on the old split.

**Check with the group:** is this what was presented? Adjust the start of the story if not.

## 2. What changed since then (the story)

The thread through everything: **the esophagus is the hard organ** (small, thin, low contrast, touches
aorta and trachea). Almost every kept change helps it most.

| Step | Mean 3D Dice | Esophagus Dice | HD95 (mm) | Source row |
|---|---|---|---|---|
| Starter code on full data (no window, no resampling), 1 seed, raw | 0.703 | 0.468 | 23.3 | `full_data_baseline` |
| Midterm pipeline on full data (window + 1.95 mm resampling), 1 seed, raw | 0.787 | 0.561 | 14.5 | `current` holdout40 |
| + native resolution 0.98 mm, 384×384, raw | 0.836 | 0.659 | 21.2 | `spacing_native_384` |
| + post-processing (keep CCs ≥10% of largest) | 0.840 | 0.666 | 12.5 | `post_native_lcc_min_fraction` |
| + GPU augmentation = **CE baseline** for everything below | 0.864 | 0.721 | 8.4 | `augmentation-native384` |
| + Dice term in the loss (CE + Dice) | 0.876 | 0.772 | 10.0 | `loss_ce_dice` |
| + weighted slice sampler | 0.880 | 0.766 | 8.5 | `combo_core_ce` |
| + wider network (ENet kernels 8 → 16) | **0.890** | **0.791** | 7.7 | `combo_core_ce_k16` |
| or + 50 epochs with cosine lr (instead of k16) | 0.885 | 0.766 | **6.8** | `combo_core_ce_cosine50` |
| k16 + 50 epochs cosine | pending | | | `combo_core_ce_k16_cosine50` |

Note: `configs/current.yaml` is now native resolution + augmentation + post-processing (=
`augmentation-native384`). The `current` row in RESULTS.md is still the **old** 1.95 mm pipeline, so
say "midterm pipeline" on the slides, not "current".

### 2a. Data: full 40-patient release (29 Sep)
- Clean GT (aorta present, affine correct); 01–20 agree with our repaired GT (Dice 0.99–1.0), which
  validates the midterm GT fix.
- New split `holdout40` (32/8) and `cv5_40`. Every old `holdout` run is no longer comparable.
- `cv5_40` fold 0 has exactly the same 8 val patients as `holdout40` (see limitations).

### 2b. Preprocessing ablation on the full data (Elena)
Kept:
- **HU window [40, 400]**: removing it costs −0.036 Dice (0/8 patients better, p = 0.008) and
  doubles HD95 (14.5 → 29.4 mm).
- **Resampling**: removing it costs −0.020, almost all from Patient_35 (the only 1.37 mm scan, −0.147).
  Window + resampling together are worth more than the sum (−0.084 vs −0.056).
- **Native in-plane resolution 0.98 mm, 384×384** — biggest single gain: +0.049 (both seeds, 8/8 and
  7/8 patients up), esophagus +0.098. Gain grows as organs get smaller. About ⅔ resolution, ⅓ the
  tighter crop (`spacing_1.95_crop192` = +0.016). Cost: 2.6× training time.
  **Why:** the esophagus is only a few pixels wide at 1.95 mm, and ENet downsamples 8× — at native
  resolution it survives the encoder.

Thrown away:
- `window_narrow` [40, 300]: −0.025, 1/8 patients better.
- `spacing_1.95_z2` (only slice thickness 2.5 → 2.0 mm): no effect (−0.007). Shows the gain is in-plane.
- `spacing_1.5_256`: +0.014, borderline (6/8, p = 0.08); superseded by native.
- `lung_window_channel` at 1.95 mm: +0.003, trade-off (trachea up, heart/aorta down).
- `lung_window_native_384`: same Dice as native seed 42 (0.847 vs 0.849) but HD95 15.7 → 10.2 mm.
  Not kept alone; it reappears in 2.5D + lung (2e).

### 2c. Post-processing (Elena, David)
- Largest connected component per organ removes far stray blobs that blow up HD95 (e.g. a heart
  fragment 126 mm away in Patient_30). Dice hardly moves; HD95 roughly halves.
- **Kept:** keep every 3D component ≥10% of the largest (`min_fraction: 0.1`). On every later run it
  gives −2 to −11 mm HD95 at +0.000 to +0.003 Dice (RESULTS.md post table).
- **Thrown away:** plain largest-CC (cuts the esophagus, which is predicted as several pieces along z)
  and skip-esophagus (best on seed 42, but seed 43 predicts flat esophagus blobs ~6 cm off in every
  patient → esophagus HD95 70 mm). Good slide on *why one seed lies*.

Second round (8 Oct), new operations on `spacing_native_384` (2 seeds, vs `min_fraction`
Dice 0.840 / HD95 12.5 / ASSD 2.44):

| Variant | Dice | HD95 (mm) | ASSD | Verdict |
|---|---|---|---|---|
| + fill holes per slice | 0.840 | 12.5 | 2.44 | no effect (no holes to fill) |
| largest CC + everything within 20 mm (instead of ≥10%) | 0.838 | 11.8 | 2.51 | worse Dice |
| closing of the esophagus along z (5 mm) + `min_fraction` | **0.843** | 10.9 | 2.29 | best Dice |
| closing + 20 mm distance + fill holes | **0.843** | **10.3** | **2.17** | best on this weak model |

- **Why closing helps:** the 2D model predicts the esophagus as fragments along z; closing bridges up to
  ~10 mm of missing slices so they are joined before the CC filter, instead of being removed by it.
- **But on our current best model it does not transfer** (`post_combo_core_ce_close_distance_fill`,
  2 seeds): Dice 0.880 → 0.880, HD95 8.45 → 8.89 mm. Seed 42 improves (esophagus HD95 9.7 → 8.0), seed 43
  gets worse (aorta HD95 6.1 → 9.4, heart 10.0 → 11.2).
  **Why:** the 20 mm distance rule keeps false-positive blobs that sit *next to* an organ, which the
  ≥10% rule removes. The old model had many far-away fragments (where the rule helps); the new model
  has few, so it mostly keeps the wrong ones. **Lesson: post-processing has to be re-validated on the
  final model.**
- Pending: closing + `min_fraction` only (no distance rule) on `combo_core_ce` and `combo_core_ce_k16`
  (`post_combo_core_ce_close_lcc`, `post_combo_core_ce_k16_close_lcc`).
- Post-processing only rescores saved predictions (`postprocess_run.py`), no retraining.

### 2d. Data augmentation (Githa)
- Affine, elastic, brightness/contrast (+ roll), moved to the GPU (CPU augmentation was the bottleneck).
- At 1.95 mm and 25 epochs: no gain (`augmentation-new-data` −0.003; 2.0: +0.025).
- **At native resolution: +0.026** (0.836 → 0.861 raw), esophagus 0.659 → 0.720, and the two seeds now
  agree within 0.001 (was 0.027 without augmentation). Kept.
  **Why only at native:** at 1.95 mm the model is limited by resolution, not by overfitting.
- Per-sample vs per-batch draw: 0.866 vs 0.864 post — within noise. Per-batch kept (it is `current`).
- Bug found and fixed (7 Oct): torchvision's elastic transform crashed at random on Mac GPUs (MPS) with
  older torch versions (boolean-mask assignment). Rewritten without it; output bit-for-bit identical.
- Known flaw: `roll` wraps the image around (anatomically impossible) — limitation / should be a
  translation.

### 2e. 2.5D input (David)
- 2.5D alone (slice ± 1 neighbour) at 1.95 mm: +0.002 → no effect. Old-split 3-seed run unstable.
- **2.5D + lung window** at native + augmentation (1 seed): 0.870 (+0.007), esophagus 0.729, HD95 8.3.
  **Dropped:** same gain as the sampler at 1.8× the training time (12 h on a laptop), and 2.5D and
  lung window were never separated.

### 2f. Class imbalance: loss functions (Puck)
All on the CE baseline (0.864, esophagus 0.721, HD95 8.4), 2 seeds each. Dice/Tversky terms are
averaged over the 4 organs only (background excluded, as in nnU-Net).

| Loss | Dice | Esophagus | HD95 (mm) | Patients better | Verdict |
|---|---|---|---|---|---|
| CE + Dice | 0.876 (+0.013) | **0.772** | 10.0 ✗ | 7.8 / 8 | kept (in the final combo) |
| Focal (γ=2) + Dice | **0.878** (+0.015) | 0.766 | 8.4 | 7.3 / 8 | tie with CE + Dice |
| CE + Tversky (α 0.3, β 0.7) | 0.873 (+0.009) | 0.756 | 8.5 | 5.8 / 8 | dropped |

- **Why a Dice term helps:** CE is an average over pixels, so the background and the large heart
  dominate it. A Dice term counts every organ equally regardless of size, so the small esophagus gets
  the same weight as the heart → esophagus +0.045–0.05 in both seeds. The clearest win after native
  resolution.
- **Focal vs CE inside it:** tie on Dice (within seed noise). CE + Dice has worse HD95 (10.0 mm in both
  seeds) — but that disappears once the sampler is added (2g), so CE + Dice was kept (simpler).
- **Why Tversky was dropped:** it weights false negatives more (β = 0.7), which favours recall: smaller
  Dice gain than the plain Dice term. At 1.95 mm it also produced stray blobs (HD95 19.9).

### 2g. Class imbalance: weighted slice sampler (David)
- Draw slices by weight instead of each once per epoch: empty slices ×0.7 (37% → 18% of draws),
  esophagus slices ×2 (53% → 75%); same epoch length.
- Alone: 0.869 (+0.005), esophagus 0.729. Seed 43 hardly gained (0.865), so weak on its own.
- **In combination it is useful:** CE + Dice + sampler (`combo_core_ce`) = 0.880 with HD95 back to 8.5
  (CE + Dice alone 10.0) and the best NSD at that point (0.620). With focal + Dice it added nothing
  (0.876 vs 0.878). We have no clear mechanism for the HD95 fix — observed in both seeds.
- Concern from the proposal (too few empty slices → stray esophagus pieces) did not happen: esophagus
  HD95 improved.

### 2h. Boundary regularizer (Britt)
Sobel-edge L1 between predicted and GT boundaries, added to the loss; per organ (boundaries between
touching organs count) or on the merged foreground.
- At native + augmentation, CE, 2 seeds (numbers from Britt's RESULTS.md branch; post-processed):
  - per organ, weight 0.1: 0.865 (+0.002), HD95 9.0 → **no effect**.
  - per organ, weight 5: 0.868 (+0.005); raw HD95 13.9 → 8.7 mm, but **after** post-processing
    9.6 vs 8.4 mm.
- In the combination (`combo_full` vs `combo_cosine`, weight 0.1): −0.003 Dice, esophagus −0.013,
  HD95 8.1 → 8.8. **Dropped** (see also weight 5 below).
- **Why it doesn't help us:** its main effect is removing stray blobs, which the CC post-processing
  already does. At weight 0.1 its gradient is too small next to CE to change anything.
- Merged-foreground variant collapses at weight 5 (0.703 on the old pipeline): it ignores the boundaries
  between touching organs, which is exactly where the esophagus errors are.
- Weight 5 on top of CE + Dice + sampler (`combo_core_ce_boundary_w5`, seed 42 only, from Britt's
  RESULTS.md): 0.873 vs 0.882, esophagus 0.753 vs 0.768, HD95 8.2 vs 8.0, NSD 0.601 vs 0.621 — worse on
  every metric, by more than the seed spread of `combo_core_ce` (0.882 / 0.879). Best epoch 13 of 25
  (vs 20–23): validation stopped improving halfway, so the term seems to work against the Dice term late
  in training. No second seed needed.

### 2i. Training schedule: cosine lr and longer training (Elena)
Constant lr 5e-4 for 25 epochs had two problems: every run peaks at epoch 20–24 (still learning), and
val Dice jumps ±0.005–0.03 between the last epochs, so the best epoch is partly luck.
- **Cosine over 25 epochs** (`schedule_cosine`): **−0.009**, esophagus −0.026, worse in both seeds.
  **Why:** it lowers the lr before the model has converged. It did make the curve flat (val Dice
  ±0.001 over the last epochs in `combo_cosine`).
- **Cosine over 50 epochs** (`combo_core_ce_cosine50`): 0.885 (+0.004 over `combo_core_ce`, 5–6/8
  patients), **best HD95 (6.8 mm) and ASSD (1.48) of all runs**, esophagus unchanged (0.766). Flat
  curve: best epoch − last epoch only +0.001 / +0.003 → the score does not depend on a lucky epoch.
  Cost: ~6.2 h per run on Snellius.
- Early stopping (David): implemented, superseded by the cosine schedule.

### 2j. Network width (Elena)
- ENet with `kernels: 8` has only 0.28M parameters. **`kernels: 16` (1.12M)** on CE + Dice + sampler:
  **0.890** in both seeds (0.8901 / 0.8899), esophagus **0.791** (+0.025), HD95 7.7, NSD 0.640 — best
  Dice of all runs, 7/8 patients better than `combo_core_ce` in every seed pairing, 8/8 vs CE baseline.
- **Why:** 0.28M parameters is very small for 384×384 inputs; the network was underfitting. Thin
  structures (esophagus, Patient_22: 0.62–0.65 → 0.73/0.65) gain most.
- No extra training time on Snellius (~3 h for 25 epochs): the A100 was underused by the small network.
  (On a MacBook it is ~2× slower per step.)
- **Caveat:** still constant lr → jumpy curve. Seed 42's best epoch had 2D val Dice 0.888, its last
  epoch 0.860. Part of k16's lead over cosine50 is epoch-selection luck → `k16_cosine50` settles it.
- Pending: `combo_core_ce_k16_cosine50`, `combo_core_ce_k32` (4.44M parameters).

### 2k. Architecture: transformer in ENet (Junis)
- One transformer layer inserted at bottleneck / stage1 / stage2 / decoder1, 1 vs 2 layers,
  positional embedding. 3 seeds each.
- Best: stage2 placement (3D 0.666 vs 0.569 for `current` on the same old split).
- **Thrown away / not carried forward:** only run on the old 20-patient split with the 1.95 mm
  pipeline; very seed-sensitive (e.g. stage2_2layer 0.371–0.713). Not comparable to the final numbers.
- **Check:** the `_nores` and with-resampling rows in RESULTS.md are identical number for number —
  looks like the same runs under two names. Sort out before showing.

## 3. Status overview

| Idea | Owner | Status | Verdict |
|---|---|---|---|
| Full 40-patient data | all | done | kept |
| HU window [40, 400] | Elena | done | **kept** |
| Narrow window [40, 300] | Elena | done | dropped |
| Resampling + crop/pad | Elena | done | **kept** |
| Native 0.98 mm, 384×384 | Elena | done, 2 seeds | **kept** |
| 1.5 mm / z = 2.0 mm / 1.95 mm crop192 | Elena | done | dropped (explain the native gain) |
| Lung window channel | Elena | done | dropped |
| Post-processing ≥10% of largest CC | Elena | done, 2 seeds | **kept** |
| Largest CC / skip esophagus | Elena | done | dropped |
| Closing / distance / fill holes | Elena, David | done on native; on combo_core_ce | closing + distance + fill dropped (doesn't transfer); closing + ≥10% pending |
| GPU augmentation | Githa | done, 2 seeds | **kept** |
| Augmentation per sample vs per batch | Githa / Elena | done, 2 seeds | tie — per batch kept |
| 2.5D (±1 slice) | David | done | dropped |
| 2.5D + lung window | David / Elena | done, 1 seed | dropped (cost) |
| CE + Dice | Puck | done, 2 seeds | **kept** |
| Focal + Dice | Puck | done, 2 seeds | tie with CE + Dice; not kept |
| CE + Tversky | Puck | done, 2 seeds | dropped |
| Weighted slice sampler | David | done, 2 seeds | **kept** (in combination) |
| Boundary regularizer | Britt | done, 2 seeds (w 0.1, 5) + in combo (w 0.1, w 5) | dropped |
| Cosine lr, 25 epochs | Elena | done, 2 seeds | dropped |
| Cosine lr, 50 epochs | Elena | done, 2 seeds | **kept candidate** (best HD95, stable) |
| ENet kernels 16 | Elena | done, 2 seeds | **kept candidate** (best Dice) |
| k16 + cosine 50 / kernels 32 | Elena | running | **decides the final config** |
| Transformer in ENet | Junis | done on old split | not carried forward |
| 5-fold cross-validation | — | not started | after final config |

## 4. Limitations (for the critical-analysis slide)

- **Seeds:** almost everything now has 2 seeds; they agree within 0.001–0.009 with augmentation.
  Differences of ~0.005 between the top candidates are within that range.
- **8 validation patients, no test set**, and the best epoch is chosen on those same 8 patients →
  optimistic numbers, worst with a constant lr (k16 seed 42: best epoch 0.888 vs last epoch 0.860).
  Cosine makes best ≈ last.
- **All design decisions were made on these 8 patients.** `cv5_40` fold 0 is the same 8 patients, so
  only folds 1–4 give an unbiased estimate of the final config.
- **Esophagus still worst** (0.79 vs 0.94 heart); predicted in fragments along z, which 2D slices
  can't fix and post-processing has to work around.
- **Distance metrics:** HD95 is NaN when an organ is missing from the prediction and dropped from
  the mean, which flatters runs that miss an organ.
- **Post-processing depends on the model:** a variant that helped the weak model hurt the strong one.
- `roll` augmentation is anatomically impossible.
- Mixed hardware (Snellius CUDA, MacBooks MPS) for runs that are compared with each other.
- Compute: native resolution 2.6× slower; 50 epochs ~6 h per run on Snellius.
- Possible solutions: 5-fold CV, more 2.5D/3D context for the esophagus, translation instead of roll.

## 5. Suggested 10-minute flow (~6 speakers)

1. Recap + the problem in one slide: esophagus is the bottleneck (midterm numbers, per-organ). ~1 min
2. Full data + preprocessing ablation → native resolution, with the "why" (pixels per esophagus,
   ENet's 8× downsampling). ~1.5 min
3. Augmentation: gain only at native resolution, and stabilises seeds. ~1 min
4. Class imbalance: Dice term (why it helps the esophagus), sampler, what was dropped (Tversky,
   boundary). ~2 min
5. Training and capacity: cosine needs enough epochs; wider ENet; 2.5D/transformer as tried-and-dropped.
   ~1.5 min
6. Post-processing: HD95 halves; the seed-43 esophagus story; re-validate on the final model. ~1 min
7. Final model vs baseline (Dice + HD95 + per organ + a qualitative 3D figure), CV, limitations. ~2 min

## 6. Open decisions before the slide deadline

- [x] Run the combo ladder → `combo_core_ce` (CE + Dice + sampler) is the base.
- [x] Second seed for the loss, sampler and cosine runs.
- [x] Per-sample vs per-batch augmentation → per batch.
- [x] Is 2.5D + lung in the final model? → no.
- [ ] `k16_cosine50` and `k32` results → pick the final config.
- [ ] Rescore the final model with closing + ≥10% CC; keep it only if it beats ≥10% alone.
- [ ] 5-fold CV of the final config (folds 1–4; fold 0 = holdout run).
- [ ] Push Britt's boundary runs and the remaining Snellius runs to master; regenerate RESULTS.md.
- [ ] Resolve the duplicated transformer rows in RESULTS.md.
- [ ] Confirm the midterm cut-off (section 1).
- [ ] Freeze RESULTS.md numbers = submission numbers.
