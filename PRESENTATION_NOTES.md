# Final presentation — notes

Working notes for the final presentation (13 / 15 Oct, 10 min + 3 min questions). Taken from
RESULTS.md, the run `summary.json` files, DECISIONS.md, the configs and the git log as of
2026-10-12. Numbers are 3D on the native GT grid, `holdout40` split (32 train / 8 val patients),
**after post-processing** (`metrics_3d_post`, keep CCs ≥10% of the largest) and **averaged over 2 seeds
(42, 43)** unless noted.

**Rule from the brief:** the numbers on the slides must be the numbers in the submission. Section 1
is the frozen table once the group confirms the final model (section 1c).

---

## 0. Speaker script (who says what)

### What the graders look for, and how this story answers it

| Criterion (from the brief) | How we cover it |
|---|---|
| Motivation of the final solution | Every part starts from a **property of SegTHOR / a failure we saw**, not from a technique |
| Results vs baseline | One recurring **ladder figure** that grows by one step per speaker; final model vs baseline + CV |
| Critical analysis, limitations, solutions | Each part says what was **dropped and why**; part 5 ends with CV + limitations → solutions |
| Balance between speakers | ~1.5 min each; part 5 still to be assigned (Elena, Githa or David) |
| No "catalog" | Each part ends with a **transition sentence** that hands the next problem to the next speaker |

**One thread through the whole talk: the esophagus.** It is the organ that fails, and almost every kept
change helps it most. Midterm pipeline: esophagus 0.561 vs heart 0.915.

**Conventions for every slide (so the numbers agree across speakers):**
- 3D Dice on the 8 holdout patients, after post-processing, mean of 2 seeds (3 for the CE baseline),
  unless the slide says "raw" or "1 seed".
- Say "midterm pipeline" for 0.787 (1.95 mm), "CE baseline" for 0.866 (native + augmentation), "final
  model" for 0.8945. Don't say "current" (it changed meaning).
- Round to 3 decimals; ±0.01 between single runs is noise (the CE baseline's own seeds: 0.863–0.870).
- Same organ colours on every chart (e.g. esophagus = one fixed colour, highlighted).

**Recurring figure — the ladder** (bar or step chart, mean Dice and esophagus Dice; each speaker adds
their bar):

| Step | Dice | Esophagus | Speaker |
|---|---|---|---|
| Starter code (raw) | 0.703 | 0.468 | intro |
| Midterm pipeline (raw) | 0.787 | 0.561 | intro |
| + native resolution (raw) | 0.835 | 0.659 | Elena |
| + augmentation = CE baseline | 0.866 | 0.724 | Githa |
| + Dice term + sampler | 0.880 | 0.766 | Puck & David |
| + wider ENet, 50 epochs cosine = final | 0.8945 | 0.793 | Junis |
| (post-processing is in every post-processed bar) | | | Britt |
| Final, cross-validation on 32 unseen patients | 0.866 ± 0.011 | 0.746 | part 5 |

**Timing (10:00):** intro 0:30 · part 0 1:30 · part 1 1:15 · part 2 1:45 · part 3 1:30 · part 4 1:30 ·
part 5 2:00.

---

### Intro + Part 0 — "Too small to see" → native resolution (Elena, ~2:00 incl. intro)

**Slide 1 — The problem (0:30).** Picture of one axial slice with the 4 organs; per-organ Dice of the
midterm pipeline.
- "SegTHOR: 4 organs at risk in thoracic CT. The heart is easy — the esophagus is the problem."
- "At the midterm we were at 0.787 Dice, but only 0.56 on the esophagus versus 0.92 on the heart."
- "The esophagus is 1–2 cm across, has almost the same intensity as the fat around it, takes up a tiny
  fraction of the voxels, and touches the aorta and trachea. Everything we did since is aimed at that."

**Slide 2 — Resolution (1:30).**
- Problem: "We resampled to 1.95 mm. The esophagus is then only 5–10 pixels wide, and ENet downsamples
  8× — so at the bottleneck it is about one pixel."
- What we did: resample to the scanners' native in-plane resolution, **0.98 × 0.98 × 2.0 mm**, crop 384×384.
- Result: **+0.049 Dice** (0.787 → 0.836), **esophagus +0.10** (0.561 → 0.659), in both seeds,
  8/8 and 7/8 patients better. Biggest single gain of the project. The gain grows as organs get smaller.
- Why it's really resolution: "a tight crop at the old resolution only gives +0.016, and changing only the
  slice thickness gives nothing (−0.007) — the gain is in-plane."
- Dropped, one line each: narrow HU window [40, 300] (−0.025, worse in 7/8 patients); slice thickness
  only (no effect); lung window as extra channel (trade-off at 1.95 mm, no gain at native: 0.847 vs
  0.849); 1.5 mm (superseded).
- Kept from before: HU window [40, 400] (removing it −0.036, HD95 doubles) and resampling (−0.020).
- Cost: 2.6× training time.
- **Transition:** "Higher resolution means more detail — but also more to learn from only 32 patients.
  That's where augmentation came in."

Likely questions:
- *Why not even higher resolution?* Native is the scanner's resolution; going finer only interpolates.
- *Why 2.0 mm in z?* Slice thickness alone made no difference; 2.0 mm is close to the native spacing.

---

### Part 1 — Augmentation as the enabler (Githa, ~1:15)

**Slide — Augmentation only helps once the resolution is right.** Small table or two bars:
augmentation at 1.95 mm vs at native.
- What: affine (rotation, scale), elastic deformation, brightness/contrast, roll; moved to the GPU
  because CPU augmentation was the training bottleneck.
- Result at 1.95 mm: **no reliable gain** (−0.003 in one run, +0.025 in the other — inconsistent).
- Result at native: **0.836 → 0.866** (+0.03), esophagus 0.659 → 0.724.
- **The second effect is the important one: it made the results reproducible.** Without augmentation
  two seeds differed by 0.027 (0.849 vs 0.822); with augmentation the seeds of a run differ by at most
  ~0.008 (CE baseline, 3 seeds: 0.863–0.870). Without that, none of the later comparisons could be
  trusted.
- Why only at native: "at 1.95 mm the model was limited by what it could see, not by overfitting.
  Once it could see the esophagus, it started to overfit the 32 training patients — that's when
  augmentation pays off."
- Detail if time: one draw per batch vs per slice: tie (0.866 / 0.866), kept per batch.
- Honest flaw: `roll` wraps the image around — anatomically impossible; a translation with padding
  would be better.
- **This bar = the CE baseline (0.866) that all later parts compare against.**
- **Transition:** "Now the model sees the esophagus, and results are stable. But it still learns
  mostly from background and heart — the esophagus is too rare."

Likely questions:
- *Why is per-batch augmentation enough?* Tested per slice too: same result.
- *Why GPU?* The CPU workers couldn't keep up at 384×384; the GPU sat idle.

---

### Part 2 — "Too rare to learn" → loss and sampling (Puck: loss, David: sampler, ~1:45)

**Slide — Make the esophagus count.** Table with the 2-seed results (green = kept):

| Change (on the CE baseline 0.866) | Dice | Esophagus | HD95 |
|---|---|---|---|
| CE + Dice (organs only) | 0.876 | **0.772** | 10.0 |
| Focal + Dice | 0.878 | 0.766 | 8.4 |
| CE + Tversky | 0.873 | 0.756 | 8.5 |
| Weighted sampler alone | 0.869 | 0.729 | 8.2 |
| **CE + Dice + sampler** | **0.880** | 0.766 | **8.5** |

**Puck — the loss:**
- Problem: "Cross-entropy is an average over pixels. In a slice, background and heart are almost all
  the pixels, so the esophagus hardly contributes to the loss."
- Fix: add a **Dice term**, computed per organ and averaged over the 4 organs (background excluded).
  Every organ then counts equally, whatever its size.
- Result: esophagus **+0.05** (0.724 → 0.772) in both seeds; mean Dice better in 7–8 of 8 patients.
- Focal + Dice: same Dice (0.878 vs 0.876, within noise) → kept CE + Dice as the simpler one.
- Tversky (weights missed pixels more, β = 0.7): smaller gain (0.873) → dropped.
- Side effect: CE + Dice has worse HD95 (10.0 mm, both seeds) — stray false positives.

**David — the sampler (and 2.5D):**
- Problem on the data side: 37% of the training slices contain no organ at all; the esophagus is in 53%.
- Fix: draw slices by weight — empty slices ×0.7, esophagus slices ×2 → 18% empty, 75% esophagus per
  epoch, same epoch length.
- Alone: small, +0.003 (0.869; one seed gained, the other hardly).
- **Together with CE + Dice: 0.880, and HD95 back to 8.5 mm** (from 10.0) in both seeds, best NSD at that
  point. We have no clean explanation for why the combination fixes the HD95 — say it's observed, not
  explained.
- 2.5D (neighbouring slices as extra channels): no effect at 1.95 mm (+0.002); with the lung window at
  native 0.870 vs 0.866 (+0.004, 1 seed), at 1.8× the training time → dropped for the cost.
- **Transition:** "Loss and data now point the model at the esophagus. Next question: is the network
  big enough, and does it see enough context?"

Likely questions:
- *Why exclude background from the Dice term?* Its Dice is always ~1, so it only dilutes the signal
  (same choice as nnU-Net).
- *Why not combine focal + Dice with the sampler?* Tried: 0.876, no better.

---

### Part 3 — "Model too small or lacking context" → transformer vs width (Junis, ~1:30)

**Slide — Two ways to give ENet more capacity.** Left: transformer; right: width curve
(0.28M → 1.1M → 4.4M parameters vs Dice).
- Hypothesis A, **context**: add a self-attention layer inside ENet so each position sees the whole slice.
  - On the old 20-patient split: best at the end of stage 2, +0.098 (0.569 → 0.666, 3 seeds); other
    placements and 2 layers were very seed-sensitive.
  - **On the final pipeline** (stage 2, in the wide ENet, with CE + Dice + sampler): **0.894 vs 0.894
    without it — no gain.** Without resampling it drops to 0.869, the same lesson as part 0.
- Hypothesis B, **capacity**: ENet with 8 kernels has only 0.28M parameters for 384×384 inputs.
  - kernels 8 → 16 → 32: **0.880 → 0.890 → 0.894**; esophagus 0.766 → 0.791 → 0.794.
  - Most of the gain is at 16; doubling again adds +0.004 at 4× the parameters → **width saturates
    at 16**. No extra training time on the A100 (the small network left the GPU underused).
- And train long enough: every run peaked at epoch 20–24 of 25 with a jumpy curve (k16: best epoch
  0.888, last epoch 0.860).
  - Cosine decay over 25 epochs: worse (−0.012) — the learning rate drops before the model has converged.
  - **50 epochs with cosine: 0.8945 → the final model**, flat at the end (best epoch ≈ last epoch), and the
    best HD95 (6.4 mm).
- Conclusion: "**Plain width was the cheaper way to give ENet capacity.** Once the network is wide enough,
  the attention layer adds nothing measurable."
- **Transition:** "The model is now good on average — but sometimes it predicts a piece of trachea or
  aorta 10–20 cm away from where it should be."

Likely questions:
- *Why not k32 + 50 epochs as final?* 0.896 vs 0.8945 — within noise, worse HD95 (7.4 vs 6.4), 4× the
  parameters, and only k16 has a cross-validation.
- *Why did the transformer help on the old split?* The old ENet was small and the split had 5 val
  patients; with a wide ENet the extra layer has nothing left to add.

---

### Part 4 — "Fragmented predictions and stray blobs" → post-processing and boundary loss (Britt, ~1:30)

**Slide — Dice barely notices, HD95 does.** One image of a stray blob + raw vs post HD95 bars.
- Problem: a 2D model sometimes predicts an organ far from where it is (e.g. a trachea blob 27 cm
  away in Patient_22). Dice hardly changes; **HD95 explodes** (271 mm for that organ).
- **Post-processing (kept):** per organ, keep every 3D component that is at least 10% of the largest.
  - Final model: HD95 **16.0 → 6.4 mm**, Dice +0.001. Same on every model (CE baseline 15.1 → 8.2).
  - Why ≥10% and not only the largest: the esophagus is predicted as several pieces along z; keeping
    only the largest cuts real esophagus away.
  - "Skip the esophagus" looked best on one seed, but the other seed put flat esophagus blobs 6 cm off
    in every patient (esophagus HD95 70 mm) → **one seed can lie**.
  - Tried closing along z, a distance rule, hole filling: helped the weak early model (HD95 12.5 → 10.3)
    but **made a stronger model worse** (aorta HD95 6.1 → 9.4) → kept the simple rule; post-processing
    must be re-checked on the final model.
- **Boundary loss (Britt):** penalise differences between predicted and true organ edges (Sobel L1),
  per organ.
  - Weight 0.1: no effect. Weight 5: **removes the stray blobs during training** (raw HD95 15.1 → 8.7 mm)
    — the same job the post-processing does.
  - After post-processing nothing is left to gain (9.6 vs 8.2), and on top of the final loss it made
    things worse (0.873 vs 0.882, 1 seed) → dropped.
  - Merged-foreground version collapsed (0.703): it ignores the boundary between touching organs —
    exactly where the esophagus errors are.
- Take-away: "**the stray-blob problem is learnable** — but a cheap post-processing rule does the same,
  for free."
- **Transition:** "So this is our final pipeline. How good is it really?"

Likely questions:
- *Why report HD95 at all?* Because Dice hides exactly these errors — a 27 cm mistake changes Dice by
  <0.01.
- *Why didn't the boundary loss help Dice?* It optimises edges, not overlap, and competes with the Dice
  term late in training (best epoch 13 of 25 instead of 20+).

---

### Part 5 — Final model, cross-validation, limitations (Elena / Githa / David — to decide, ~2:00)

**Slide A — Final model vs baseline (0:40).** Pipeline diagram + table:

| | Starter | Midterm | CE baseline | **Final** |
|---|---|---|---|---|
| Mean Dice | 0.703 | 0.787 | 0.866 | **0.8945** |
| Esophagus | 0.468 | 0.561 | 0.724 | **0.793** |
| Heart / trachea / aorta | .840 / .786 / .720 | .915 / .808 / .865 | .939 / .883 / .918 | **.945 / .902 / .938** |
| HD95 (mm) | 23.3 | 14.5 | 8.2 | **6.4** |

- Final pipeline: HU window → native 0.98 mm, 384×384 → ENet kernels 16 → GPU augmentation →
  CE + Dice + weighted sampler → 50 epochs cosine → keep components ≥10% of the largest.
- "+0.108 over the midterm pipeline, better in all 8 patients; the weakest midterm patients gain most
  (Patient_28: 0.70 → 0.87)."
- Qualitative figure: one patient, midterm vs final (e.g. Patient_28), esophagus highlighted.

**Slide B — Cross-validation: the honest number (0:40).**
- "All our choices were made on the same 8 patients. So at the end we trained the final model on 4 other
  folds — 32 patients that played no role in any decision."
- **0.866 ± 0.011** (folds 0.858 / 0.864 / 0.885 / 0.858) vs 0.8945 on the holdout.
- The drop is in the small organs: esophagus 0.746 vs 0.793, trachea −0.03, aorta −0.03; the heart
  generalises (0.940 vs 0.945).
- Why: about half is fold difficulty (the untuned starter code is also ~0.013 better on our holdout),
  half is **selection bias** from tuning on 8 patients.
- The gain is real: on the same 32 unseen patients, starter code → final model = **0.690 → 0.865**
  (raw, +0.175).

**Slide C — Limitations → solutions (0:40).**

| Limitation | Evidence | Solution |
|---|---|---|
| Holdout is optimistic | CV 0.866 vs 0.8945 | Choose on CV folds, keep an untouched test set |
| Esophagus still the weakest | 0.75 in CV vs 0.94 heart; Patient_24 0.50 | 3D / 2.5D context, shape prior |
| Post-processing can't fix large wrong blobs | Patient_14 trachea HD95 208 mm in CV | Anatomical position rule; boundary loss |
| Post-processing removes real esophagus pieces on unseen patients | esophagus HD95 raw 9.6 → post 10.6 in CV | Close along z before filtering |
| Few seeds / 8 val patients | ±0.01 is noise | More seeds, CV for every decision |
| `roll` augmentation | wraps anatomy around | Translation with padding |

- Close with one sentence: "From 0.56 to 0.79 on the esophagus — mostly by letting the model see it,
  making it count, and giving the network enough capacity."

Likely questions:
- *Why not use CV for every decision?* Compute: one 50-epoch run is ~6 h on Snellius; CV × every
  experiment wasn't feasible. That's the first thing we'd change.
- *Is 0.866 better than the baseline?* The baseline was never cross-validated except the starter
  code (0.690 raw on the same folds). On the holdout: 0.8945 vs 0.866, +0.029, in both seeds.
- *What about the test set?* No labels for it; CV is our estimate of unseen-patient performance.

---

## 1. Final results

### 1a. The ladder (one change per step, post-processed unless "raw")

| Step | Runs | Dice | Esophagus | Heart | Trachea | Aorta | HD95 (mm) | ASSD | NSD@1mm |
|---|---|---|---|---|---|---|---|---|---|
| Starter code, full data (raw) | 1 | 0.703 | 0.468 | 0.840 | 0.786 | 0.720 | 23.3 | 5.43 | 0.360 |
| Midterm pipeline: HU window + 1.95 mm resampling (raw) | 1 | 0.787 | 0.561 | 0.915 | 0.808 | 0.865 | 14.5 | 2.93 | 0.440 |
| + native resolution 0.98 mm, 384×384 (raw) | 2 | 0.835 | 0.659 | 0.923 | 0.861 | 0.899 | 21.2 | 3.26 | 0.534 |
| + GPU augmentation (+ post-processing) | 3 | 0.866 | 0.724 | 0.939 | 0.883 | 0.918 | 8.2 | 1.71 | 0.587 |
| + CE + Dice loss | 2 | 0.876 | 0.772 | 0.929 | 0.886 | 0.918 | 10.0 | 1.83 | 0.605 |
| + weighted slice sampler (`combo_core_ce`) | 2 | 0.880 | 0.766 | 0.936 | 0.890 | 0.928 | 8.5 | 1.67 | 0.620 |
| + wider ENet, kernels 8 → 16 | 2 | 0.890 | 0.791 | 0.938 | 0.900 | 0.931 | 7.7 | 1.55 | 0.640 |
| **+ 50 epochs, cosine lr = FINAL (`combo_core_ce_k16_cosine50`)** | 2 | **0.8945** | **0.793** | **0.945** | **0.902** | **0.938** | **6.4** | **1.38** | **0.653** |
| Final, 4-fold CV (folds 1–4, 32 other patients) | 4 | 0.866 ± 0.011 | 0.746 | 0.940 | 0.870 | 0.908 | 11.1 | 1.91 | 0.637 |
| (side result: kernels 32, 25 epochs, `combo_core_ce_k32`) | 2 | 0.894 | 0.794 | 0.945 | 0.901 | 0.935 | 7.1 | 1.44 | 0.654 |
| (side result: kernels 32, 50 epochs cosine, `combo_core_ce_k32_cosine50`) | 2 | 0.896 | 0.800 | 0.943 | 0.904 | 0.937 | 7.4 | 1.46 | 0.661 |
| (side result: + stage-2 transformer on k32, 25 epochs, `tf_st2_combo_core_ce_k32_res_pos`) | 1 | 0.894 | 0.798 | 0.949 | 0.901 | 0.928 | 7.5 | 1.44 | 0.655 |

Final model per seed: 0.8953 (s42, Snellius) / 0.8937 (s43). **Seed 42 is in W&B but not yet in the
repo** — push `results/combo_core_ce_k16_cosine50/holdout40-f0-s42/` before freezing.

Headlines:
- **Starter code → final: Dice 0.703 → 0.8945, esophagus 0.468 → 0.793 (+0.33), HD95 23.3 → 6.4 mm.**
  (Starter is raw, final post-processed; final raw is 0.893, so it hardly matters. Like-for-like on the
  CV folds 1–4, both raw: 0.690 → 0.865.)
- **Midterm pipeline → final: +0.108 Dice, esophagus +0.232; better in 8/8 validation patients**
  (per patient, mean Dice: 01 .82→.92, 13 .82→.89, 16 .83→.90, 22 .78→.87, 28 .70→.87, 30 .73→.88,
  35 .82→.91, 39 .80→.91). The weakest midterm patients (28, 30) gain most.
- Seeds agree within 0.000–0.008 for every step after augmentation (0.027 before it); the CE
  baseline's 3 seeds span 0.863–0.870.
- The final holdout number is **2 seeds** (per-patient values above are the 2-seed mean).

### 1b. Post-processing: still needed for the final model

HD95 raw → post-processed (≥10% CC):

| Model | Raw HD95 | Post HD95 | Δ |
|---|---|---|---|
| augmentation-native384 (k8, CE), 3 seeds | 15.1 | 8.2 | −7.0 |
| combo_core_ce (k8), 2 seeds | 14.7 | 8.5 | −6.3 |
| combo_core_ce_k16, 2 seeds | 12.9 | 7.7 | −5.3 |
| **final k16_cosine50, holdout, 2 seeds** | **16.0** (s42 8.8 / s43 23.1) | **6.4** | **−9.6** |
| final k16_cosine50, CV folds 1–4 | 13.0 | 11.1 | −2.0 |
| combo_core_ce_k32, 2 seeds | 6.9 | 7.1 | +0.2 |
| combo_core_ce_k32_cosine50, 2 seeds | 7.0 | 7.4 | +0.3 |

- The final model still makes **stray blobs far from the organ**, different ones per seed: seed 43 a
  trachea blob in Patient_13 (trachea HD95 183 → 6 mm) and Patient_22 (271 → 5 mm), seed 42 an aorta blob
  in Patient_30 (82 → 3 mm). Post-processing removes all of them. Dice barely notices (+0.001), HD95 does:
  a good slide on why we report a distance metric next to Dice.
- k32 makes no such blobs (side note: more capacity also removes them; the boundary loss at weight 5
  removed most of them too, see 2e). For k32 the filter even costs a little (+0.2–0.3 mm).

### 1c. Final model: `combo_core_ce_k16_cosine50` (group decision, 12 Oct)

Why this one over k32 (0.894 at 25 epochs) and k32 + cosine50 (0.896, the highest holdout Dice):
- **The Dice differences are within noise:** 0.8945 vs 0.894 vs 0.896 (the CE baseline's own seeds
  span 0.863–0.870).
- **It is the only configuration with a cross-validation** (folds 1–4) → the final model has an
  estimate on 32 patients that played no part in choosing it.
- **4× fewer parameters** (1.1M vs 4.4M): smaller and faster at inference.
- Best HD95 / ASSD after post-processing (6.4 / 1.38 mm vs 7.1 / 1.44 for k32 and 7.4 / 1.46 for
  k32 + cosine50).
- Cosine decay flattens the end of the curve, so the score depends less on picking a lucky epoch than
  with a constant lr (k16 at constant lr, seed 42: best epoch 0.888 vs last 0.860).

**Not a reason:** training time. On the same Runpod A40, k16_cosine50 took 561 min (seed 43) vs 259 / 325
min for k32 (similar time per epoch, but twice the epochs). Seed 42 took 348 min on Snellius;
k32 + cosine50 447 / 469 min.

Final pipeline: HU window [40, 400] → resample to 0.98 × 0.98 × 2.0 mm, crop 384×384 → ENet,
kernels 16 (1.1M parameters) → GPU augmentation (affine, elastic, brightness/contrast, roll; one draw
per batch) → CE + Dice (organs only) → weighted slice sampler (empty ×0.7, esophagus ×2) → Adam 5e-4
with cosine decay to 0 over 50 epochs → best epoch on 2D val Dice → stitch to 3D → keep CCs ≥10% of the
largest.

### 1d. Cross-validation (final model, folds 1–4, seed 42) — closing section of the talk

- **0.866 ± 0.011** vs 0.8945 on the holdout (2 seeds; seed 42 is fold 0 of the CV split: 0.895).
  Fold scores 0.858 / 0.864 / 0.885 / 0.858 (post-processed; raw 0.856 / 0.861 / 0.882 / 0.859).
  All 5 folds (fold 0 = seed 42 holdout run): 0.872 ± 0.015.
- Per organ, CV vs holdout: esophagus 0.746 vs 0.793 (−0.047), trachea 0.870 vs 0.902 (−0.032), aorta
  0.908 vs 0.938 (−0.030), heart 0.940 vs 0.945 (−0.005). **The drop is in the small organs; the heart
  generalises.**
- Why it drops:
  1. **Selection bias:** every design choice and every best epoch was picked on the 8 holdout patients.
     The CV folds are the first numbers nobody tuned on.
  2. **Harder patients in other folds:** 14 (0.72), 24 (0.74), 20 (0.78), 03 (0.78), 05 (0.79). The
     8 holdout patients range 0.867–0.924 (2-seed mean). The starter code, never tuned on anything,
     is only slightly better on fold 0 than on folds 1–4 (raw 0.703 vs 0.690): fold 0 is ~0.013
     easier, so roughly half of the 0.029 drop is fold difficulty and half is selection bias.
  3. **Failures post-processing can't fix:** Patient_14 has a trachea blob that is large enough
     (≥10% of the trachea) to survive the filter → trachea HD95 208 mm even after post-processing.
     Patient_12 and 20 the same at ~21–24 mm.
- The gain is still real: CV 0.866 vs midterm pipeline 0.787 (holdout). Strictly that compares different
  patients; it is the honest lower bound we have.
- Solutions: tune on CV folds and keep an untouched test set; anatomical-position check in
  post-processing (trachea must be above the carina, near the midline); 2.5D/3D context for the
  esophagus; inspect the hard patients' scans and GT.
- Optional qualitative figure: Patient_14 (worst) next to Patient_01 (best).

---

## 2. What we tried, grouped by problem (details per experiment)

The thread through everything: **the esophagus is the hard organ**. In SegTHOR it is small (1–2 cm
across), low contrast against surrounding fat, a tiny fraction of the voxels compared to the heart,
and it touches the aorta and trachea. The midterm model got 0.56 on it vs 0.92 on the heart.

### 2a. "It is too small to see" → resolution (Elena) — KEPT
- Full 40-patient release (29 Sep): clean GT; 01–20 agree with our repaired GT (Dice 0.99–1.0), which
  validates the midterm GT fix. New split `holdout40`; old `holdout` runs no longer comparable.
- HU window [40, 400]: removing it costs −0.036 (0/8 patients better, p = 0.008), HD95 14.5 → 29.4.
- Resampling: removing it costs −0.020, almost all from Patient_35 (only 1.37 mm scan, −0.147).
- **Native in-plane resolution 0.98 mm, 384×384: +0.049 (both seeds), esophagus +0.098** — biggest
  single gain. The esophagus is 5–10 px wide at 1.95 mm, and ENet downsamples 8× → ~1 px at the
  bottleneck. At 0.98 mm it survives. ⅔ resolution, ⅓ tighter crop (`spacing_1.95_crop192` +0.016).
- Dropped: narrow window [40, 300] (−0.025); slice thickness only (−0.007, gain is in-plane);
  1.5 mm (+0.014, superseded); lung window channel (+0.003 at 1.95 mm, trade-off trachea vs
  heart/aorta; no Dice gain at native).

### 2b. Augmentation (Githa) — KEPT, the enabler
- At 1.95 mm: no gain (−0.003 / +0.025). **At native: +0.026–0.031**, esophagus 0.659 → 0.724, and seed
  spread 0.027 → 0.002–0.008. Why only at native: at 1.95 mm the model was limited by resolution,
  not by overfitting.
- Moved to the GPU (CPU was the bottleneck); per batch vs per sample: tie (0.866 / 0.866).
- Flaw: `roll` wraps the image around (anatomically impossible) → should be a translation.

### 2c. "It is too rare to learn" → class imbalance (Puck, David) — KEPT: CE + Dice + sampler
On the augmented native baseline (0.866), 2 seeds each:

| Change | Dice | Esophagus | HD95 | Verdict |
|---|---|---|---|---|
| CE + Dice | 0.876 | **0.772** | 10.0 | **kept** |
| Focal (γ=2) + Dice | 0.878 | 0.766 | 8.4 | tie; CE + Dice simpler |
| CE + Tversky (α 0.3, β 0.7) | 0.873 | 0.756 | 8.5 | dropped |
| Weighted sampler alone | 0.869 | 0.729 | 8.2 | weak alone |
| CE + Dice + sampler | **0.880** | 0.766 | **8.5** | **kept** |
| Focal + Dice + sampler | 0.876 | 0.761 | 8.3 | dropped |

- Why the Dice term helps: CE averages over pixels, so background and heart dominate; Dice weighs each
  organ equally → esophagus +0.05 in both seeds.
- Tversky favours recall (β = 0.7) → smaller Dice gain; at 1.95 mm it also produced stray blobs.
- Sampler: empty slices 37% → 18% of draws, esophagus slices 53% → 75%. Alone weak; with CE + Dice it
  fixed CE + Dice's HD95 (10.0 → 8.5). No clear mechanism — observed in both seeds.

### 2d. "The model is too small / lacks context" → architecture (Junis, David, Elena)
- **Transformer in ENet (Junis):** one self-attention layer at 4 places, 1 vs 2 layers, positional
  embedding, 3 seeds each, on the old 20-patient split. Best placement: end of stage 2 (H/8), +0.098
  3D Dice over the then-baseline (0.569 → 0.666) with low seed spread (±0.013); bottleneck and 2-layer
  variants very seed-sensitive (stage2_2layer 0.371–0.713).
  On the full data (`enet_tf_stage2_nores_full`, k8, CE, with augmentation but **without resampling**):
  0.816, HD95 12.9.
  **On the final pipeline (11 Oct):** stage-2 transformer in the k32 ENet with CE + Dice + sampler,
  resampled, 25 epochs, 1 seed each:

  | Variant | Dice | Esophagus | HD95 | vs plain k32 (2 seeds) |
  |---|---|---|---|---|
  | stage 2, positional embedding | 0.894 | 0.798 | 7.5 | 0.894 / 0.794 / 7.1 → no gain |
  | stage 2, no positional embedding | 0.893 | 0.793 | 8.0 | no gain |
  | stage 2 + pos. embedding, **no resampling** | 0.869 | 0.768 | 8.3 | −0.025 |
  | stage 2, no resampling | 0.869 | 0.762 | 8.9 | −0.025 |

  → **Tested on the final pipeline: no gain over plain k32.** Once ENet is wide enough, the extra
  attention layer adds nothing measurable. Without resampling it loses 0.025 — the same lesson as 2a.
  Dropped. ⚠️ The 2-layer run (`tf_st2_2l_combo_core_ce_k32_res_nopos`) reports exactly the same scores
  as the 1-layer run in every metric, although `num_layers: 2` is used by the model → its results
  folder almost certainly holds the 1-layer results. Don't show it until Junis has checked.
- **2.5D (David):** ±1 slice alone: +0.002 at 1.95 mm. With lung window at native: +0.007 at 1.8× the
  training time (12 h laptop), 1 seed. Dropped.
- **Wider ENet (Elena): KEPT, kernels 16.** Kernels 8 (0.28M params) → 16 (1.1M) → 32 (4.4M):
  0.880 → 0.890 → 0.894, esophagus 0.766 → 0.791 → 0.794. The network was underfitting; most of the
  gain is at k16, and k16 + 50 epochs cosine (0.8945) matches k32 (0.894) with 4× fewer parameters.
  k32 + 50 epochs cosine (David): 0.896, esophagus 0.800, HD95 7.4 — +0.0015 over the final model,
  within noise, with worse HD95. Width saturates at 16.
- The link for the story: the transformer was the "more context / capacity" hypothesis; plain width
  turned out to be the cheaper way to give ENet that capacity.

### 2e. "Predictions are fragmented and leave stray blobs" → shape (Elena, David, Britt)
- **Post-processing — KEPT:** keep every 3D component ≥10% of the largest. HD95 roughly halves on the
  k8 models (e.g. a heart fragment 126 mm away, Patient_30). Plain largest-CC cuts the fragmented
  esophagus; skip-esophagus looked best on seed 42 but seed 43 has flat esophagus blobs ~6 cm off
  (esophagus HD95 70 mm) → **one seed can lie**.
- Second round: closing along z + ≥10% (best on the weak native model, 0.843 / 10.9 mm), distance rule,
  hole filling. **Did not transfer** to better models: close + distance + fill on `combo_core_ce` HD95
  8.45 → 8.89; close + ≥10% on k16 0.890 → 0.891, HD95 7.65 → 7.60 (negligible). Kept the simple rule.
- **Boundary regularizer (Britt):** Sobel-edge L1 between predicted and GT boundaries, per organ or
  merged. Per organ, 2 seeds:
  - weight 0.1: 0.865, no effect (gradient too small next to CE).
  - **weight 5: raw HD95 15.1 → 8.7 mm** — the loss itself removes the stray blobs, matching what
    post-processing does (8.2). After post-processing nothing is left to gain (9.6).
  - In the combination: weight 0.1 (`combo_full`) −0.003; weight 5 on CE + Dice + sampler, 1 seed:
    0.873 vs 0.882, best epoch 13 of 25 → it competes with the Dice term late in training.
  - Merged-foreground variant collapses at weight 5 (0.703, old pipeline): ignores boundaries between
    touching organs, exactly where the esophagus errors are.
  - Dropped: post-processing does the same job for free. But it shows the stray-blob problem is
    learnable — and the final model still makes blobs (1b), so it is a natural next step.

### 2f. Training schedule (Elena, David) — KEPT: 50 epochs cosine
- Cosine over 25 epochs: 0.854, −0.012 vs the 3-seed baseline (lr drops before convergence) — dropped.
- Cosine over 50 epochs: 0.885 vs 0.880 at k8, best HD95 at k8 (6.8), flat curve (best ≈ last epoch).
  **With k16: 0.8945 → final model.** With k32: 0.896.
- Early stopping (David): implemented, superseded by cosine.

## 3. Status overview

| Idea | Owner | Verdict |
|---|---|---|
| HU window, resampling, native 0.98 mm | Elena | **kept** |
| Narrow window, z only, 1.5 mm, lung window | Elena | dropped |
| GPU augmentation | Githa | **kept** |
| CE + Dice | Puck | **kept** |
| Focal + Dice, CE + Tversky | Puck | dropped |
| Weighted slice sampler | David | **kept** |
| 2.5D, 2.5D + lung | David | dropped |
| Transformer in ENet | Junis | dropped (on the final k32 pipeline: no gain, 0.894 vs 0.894) |
| Wider ENet, kernels 16 | Elena | **kept** (k32 and k32 + cosine50: equal Dice within noise, 4× the parameters) |
| Boundary regularizer | Britt | dropped (redundant with post-processing) |
| Post-processing ≥10% CC | Elena | **kept** (HD95 16.0 → 6.4 on the final model) |
| Closing / distance / fill holes | Elena, David | dropped |
| Cosine 50 epochs | Elena | **kept** |
| Cosine 25 epochs, early stopping | Elena, David | dropped |
| 4-fold CV of the final model | Elena | done (0.866 ± 0.011) |

## 4. Limitations and possible solutions

| Limitation | Evidence | Possible solution |
|---|---|---|
| Holdout is optimistic | CV 0.866 vs holdout 0.8945; all choices and best epochs on the same 8 patients | Select on CV folds, keep a final untouched test set |
| CV is one seed per fold | folds 1–4 seed 42 only | more seeds per fold |
| Post-processing misses large wrong blobs | Patient_14 trachea HD95 208 mm after filter | anatomical-position rule, boundary loss, more capacity (k32 has none) |
| Esophagus still the weakest | 0.79 holdout / 0.75 CV vs heart 0.94; hard patients 14, 24 | 3D or longer-context 2.5D model, esophagus-specific shape prior |
| 2D model, z-fragmentation | esophagus predicted as pieces along z | 3D / 2.5D context, closing along z |
| Transformer tested at one place only | stage 2 on k32, 1 seed, 25 epochs constant lr; 2-layer result unreliable | other placements / 50 epochs cosine on the final pipeline |
| Only 2 seeds, 8 val patients | differences < ~0.01 within noise (CE baseline seeds 0.863–0.870) | more seeds, CV |
| HD95 NaN dropped when an organ is missing | flatters runs that miss organs | penalise missing organs (e.g. max distance) |
| `roll` augmentation | wraps anatomy around | translation with padding |
| Mixed hardware (MPS / CUDA) | runs compared across machines | rerun key baselines on one machine |

## 5. Open decisions before the slide deadline

- [x] Final model: `combo_core_ce_k16_cosine50` (12 Oct), with its 4-fold CV at the end of the talk.
- [x] `k16_cosine50` seed 42 on holdout: done (0.8953, W&B). [ ] Push its results folder to the repo.
- [ ] Ask Junis about the 2-layer transformer run (identical scores to the 1-layer run).
- [ ] Push `combo_core_ce_k32_cosine50` (David, W&B only) to the repo if it stays in the table.
- [ ] Set `configs/current.yaml` to the final model and log it in DECISIONS.md.
- [ ] Qualitative figure: final vs midterm on one easy and one hard patient (e.g. 28 and 22).
- [ ] Freeze RESULTS.md numbers = submission numbers.
