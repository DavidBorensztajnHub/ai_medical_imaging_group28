# Slide outline — final presentation (13 / 15 Oct)

10 minutes, 15 content slides + title + questions, plus backup slides for Q&A. What to *say* per slide is
in [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md), section 0 (speaker script); numbers come from its
section 1 (frozen table). This file is about what is *on* each slide.

**Design rules for every slide**
- **Headline = the message**, written as a full sentence ("Native resolution was the biggest single
  gain"), not a topic ("Resolution").
- At most ~3 short bullets or one table/figure per slide; the speaker says the rest.
- Numbers: 3 decimals, post-processed, mean of seeds; label "raw" or "1 seed" when that applies.
- Fixed organ colours on every chart; the **esophagus is always the highlighted colour**.
- The **ladder** (mean Dice + esophagus Dice per step) appears as a small strip in the corner of
  slides 4, 5, 7, 9 and 12, with the current step highlighted, so the audience always sees where we are.
- Each speaker's last slide ends with the transition line from the script (spoken, not written).

| # | Speaker | Time | Headline |
|---|---|---|---|
| 1 | — | 0:00 | Title |
| 2 | Elena | 0:30 | The esophagus is the organ that fails |
| 3 | Elena | 0:45 | At 1.95 mm the esophagus disappears inside ENet |
| 4 | Elena | 0:45 | Native resolution was the biggest single gain |
| 5 | Githa | 1:15 | Augmentation only paid off at native resolution — and made results reproducible |
| 6 | Puck | 0:50 | Cross-entropy ignores the esophagus; a Dice term makes every organ count |
| 7 | David | 0:55 | Sampling more esophagus slices fixes the Dice term's stray errors |
| 8 | Junis | 0:50 | More context or more capacity? Width won |
| 9 | Junis | 0:40 | Trained long enough, with cosine decay: the final model |
| 10 | Britt | 0:50 | Dice hides stray blobs; HD95 shows them — post-processing removes them |
| 11 | Britt | 0:40 | A boundary loss learns the same thing, but post-processing does it for free |
| 12 | Part 5 | 0:40 | Final model: +0.108 Dice over the midterm, better in all 8 patients |
| 13 | Part 5 | 0:40 | On 32 unseen patients: 0.866 — the honest number |
| 14 | Part 5 | 0:40 | Limitations and what we'd do next |
| 15 | — | — | Questions |
| B1–B6 | — | — | Backup slides (only when asked) |

Total: 10:00. Part 5 speaker still to decide (Elena, Githa or David).

---

## Slide 1 — Title (no time)
- Title: *Segmenting the esophagus: from 0.56 to 0.79 Dice on SegTHOR* (or the group's own title).
- Group 28, six names, date.
- Visual: a 3D render of the 4 organs from the final model (esophagus highlighted).

## Elena — intro and resolution (2:00)

### Slide 2 — The esophagus is the organ that fails (0:30)
- **Visual:** left, one axial CT slice with the 4 organs outlined (esophagus small, next to aorta and
  trachea); right, bar chart of midterm per-organ Dice: esophagus 0.561 · heart 0.915 · trachea 0.808 ·
  aorta 0.865.
- **On slide:** "1–2 cm across · same intensity as the fat around it · few voxels · touches aorta and
  trachea".
- Script: section 0, Slide 1.

### Slide 3 — At 1.95 mm the esophagus disappears inside ENet (0:45)
- **Visual:** the same zoomed crop around the esophagus at 1.95 mm and at 0.98 mm (pixel grid
  visible), plus a small arrow "ENet downsamples 8× → ~1 pixel at the bottleneck".
- **On slide:** "1.95 mm: 5–10 px wide → ~1 px at the bottleneck" vs "0.98 mm (native): 10–20 px".
- Script: section 0, Slide 2, first two bullets.

### Slide 4 — Native resolution was the biggest single gain (0:45)
- **Visual:** ladder strip with the first three bars (starter 0.703 → midterm 0.787 → native 0.835),
  esophagus 0.468 → 0.561 → 0.659.
- **On slide:** small table "kept / dropped":
  - Kept: HU window [40, 400], native 0.98 mm resampling (+0.049, esophagus +0.10, both seeds).
  - Dropped: narrow window (−0.025) · slice thickness only (no effect) · lung window channel (no gain
    at native).
- Footnote: "gain is in-plane: tight crop at 1.95 mm only +0.016; slice thickness only −0.007".
- Script: section 0, Slide 2, results and dropped.

## Githa — augmentation (1:15)

### Slide 5 — Augmentation only paid off at native resolution — and made results reproducible (1:15)
- **Visual, two panels:**
  - Left: 2×2 bars — no augmentation vs augmentation, at 1.95 mm and at native. At 1.95 mm no reliable
    change (−0.003 / +0.025); at native 0.836 → 0.866, esophagus 0.659 → 0.724.
  - Right: dot plot of seeds — without augmentation 0.849 and 0.822 (spread 0.027), with augmentation
    0.863 / 0.863 / 0.870 (spread ~0.008).
- **On slide:** row of 4 small augmented example slices (affine, elastic, brightness/contrast, roll),
  "on the GPU".
- Bottom line: "**CE baseline: 0.866** — every later step is compared to this".
- Script: section 0, Part 1.

## Puck and David — class imbalance (1:45)

### Slide 6 — Cross-entropy ignores the esophagus; a Dice term makes every organ count (Puck, 0:50)
- **Visual:** left, pixel-share bar of a typical slice (background ≫ heart ≫ … ≫ esophagus); right,
  the loss table (2 seeds each):

  | Loss | Dice | Esophagus | HD95 |
  |---|---|---|---|
  | CE (baseline) | 0.866 | 0.724 | 8.2 |
  | **CE + Dice** | **0.876** | **0.772** | 10.0 |
  | Focal + Dice | 0.878 | 0.766 | 8.4 |
  | CE + Tversky | 0.873 | 0.756 | 8.5 |

- **On slide:** "Dice term per organ, background excluded → esophagus +0.05 in both seeds".
- Mark: CE + Dice kept (simplest, ties focal); Tversky dropped; the 10.0 mm HD95 in red (handed to David).
- Script: section 0, Part 2, Puck.

### Slide 7 — Sampling more esophagus slices fixes the Dice term's stray errors (David, 0:55)
- **Visual:** two stacked bars "share of slices per epoch": before 37% empty / 53% esophagus → after
  18% / 75%; next to it before/after for CE + Dice: Dice 0.876 → **0.880**, HD95 10.0 → **8.5 mm**.
- **On slide:** "empty ×0.7 · esophagus ×2 · same epoch length"; one line "2.5D (+ lung window):
  +0.004 at 1.8× training time → dropped".
- Ladder strip: + Dice term + sampler = 0.880 / esophagus 0.766.
- Script: section 0, Part 2, David.

## Junis — capacity and context (1:30)

### Slide 8 — More context or more capacity? Width won (0:50)
- **Visual, two panels:**
  - Left "context": ENet sketch with a self-attention block at the end of stage 2; result on the final
    pipeline: 0.894 with vs 0.894 without → no gain (and 0.869 without resampling).
  - Right "capacity": line chart parameters (0.28M → 1.1M → 4.4M, kernels 8 → 16 → 32) vs Dice
    (0.880 → 0.890 → 0.894) and esophagus (0.766 → 0.791 → 0.794); annotate "saturates at 16".
- **On slide:** "Plain width is the cheaper way to give ENet capacity".
- Script: section 0, Part 3, first two hypotheses.

### Slide 9 — Trained long enough, with cosine decay: the final model (0:40)
- **Visual:** validation-Dice curves per epoch (from W&B): k16 at constant lr for 25 epochs (jumpy, best
  0.888 vs last 0.860) vs k16 with cosine for 50 epochs (rises, then flat).
- **On slide:** "cosine over 25 epochs: −0.012 (lr drops too early) · over 50 epochs: **0.8945**, best
  epoch ≈ last epoch, HD95 6.4 mm".
- Ladder strip: final = 0.8945 / esophagus 0.793.
- Script: section 0, Part 3, "train long enough".

## Britt — stray blobs (1:30)

### Slide 10 — Dice hides stray blobs; HD95 shows them — post-processing removes them (0:50)
- **Visual:** 3D render (or coronal view) of one patient with a stray trachea blob far from the trachea
  (Patient_22 or Patient_13, final model seed 43), before and after post-processing; next to it bars of
  HD95 raw vs post: final model 16.0 → 6.4 mm, CE baseline 15.1 → 8.2 mm.
- **On slide:** "Keep every 3D component ≥10% of the largest · Dice +0.001 · HD95 −60%";
  small note "only-largest cuts the esophagus (several pieces along z)".
- Optional one-liner: "one seed can lie: skip-esophagus looked best on seed 42, failed on seed 43".
- Script: section 0, Part 4, post-processing.

### Slide 11 — A boundary loss learns the same thing, but post-processing does it for free (0:40)
- **Visual:** bars of HD95 for CE vs CE + boundary loss (weight 5): raw 15.1 → 8.7 mm, after
  post-processing 8.2 vs 9.6 mm; small inset Sobel edge map.
- **On slide:** "weight 0.1: no effect · weight 5: removes blobs during training · on the final loss:
  0.873 vs 0.882 → dropped"; "merged-foreground version collapses (0.703): ignores boundaries between
  touching organs".
- Script: section 0, Part 4, boundary loss.

## Part 5 — final model, cross-validation, limitations (2:00, speaker TBD)

### Slide 12 — Final model: +0.108 Dice over the midterm, better in all 8 patients (0:40)
- **Visual:** pipeline diagram in one row: HU window → native 0.98 mm, 384×384 → ENet k16 → GPU
  augmentation → CE + Dice + sampler → 50 epochs cosine → keep components ≥10%. Below it the comparison
  table:

  | | Starter | Midterm | CE baseline | **Final** |
  |---|---|---|---|---|
  | Mean Dice | 0.703 | 0.787 | 0.866 | **0.8945** |
  | Esophagus | 0.468 | 0.561 | 0.724 | **0.793** |
  | HD95 (mm) | 23.3 | 14.5 | 8.2 | **6.4** |

- Right side: qualitative image, Patient_28 midterm vs final (esophagus highlighted; 0.70 → 0.87).
- Script: section 0, Part 5, Slide A.

### Slide 13 — On 32 unseen patients: 0.866 — the honest number (0:40)
- **Visual:** left, bars per fold (folds 1–4: 0.858 / 0.864 / 0.885 / 0.858, mean 0.866 ± 0.011) with
  the holdout 0.8945 as a dashed line; right, per-organ holdout vs CV (esophagus 0.793 vs 0.746, trachea
  0.902 vs 0.870, aorta 0.938 vs 0.908, heart 0.945 vs 0.940).
- **On slide:** "drop is in the small organs · ~half fold difficulty, ~half selection bias ·
  starter → final on the same 32 patients: 0.690 → 0.865 (raw)".
- Script: section 0, Part 5, Slide B.

### Slide 14 — Limitations and what we'd do next (0:40)
- **Visual:** the limitations → solutions table (4–5 rows max):

  | Limitation | Evidence | Next step |
  |---|---|---|
  | Holdout is optimistic | CV 0.866 vs 0.8945 | Decide on CV folds, untouched test set |
  | Esophagus still weakest | 0.75 in CV; Patient_24 0.50 | 3D / 2.5D context, shape prior |
  | Large wrong blobs survive | Patient_14 trachea HD95 208 mm | Anatomical position rule |
  | Filter removes real esophagus pieces | CV esophagus HD95 9.6 → 10.6 | Close along z before filtering |
  | Few seeds, 8 patients | ±0.01 is noise | More seeds, CV per decision |

- Closing line (spoken): "From 0.56 to 0.79 on the esophagus — by letting the model see it, making it
  count, and giving the network enough capacity."
- Script: section 0, Part 5, Slide C.

## Slide 15 — Questions
- Final ladder strip as the background, "Thank you — questions?".

---

## Backup slides (only shown when asked)

| # | Content | Likely question it answers |
|---|---|---|
| B1 | Full ladder table, all organs + HD95/ASSD/NSD (notes section 1a) | "What are the exact numbers per organ?" |
| B2 | All loss/sampler runs with 2 seeds (notes section 2c) | "Did you try focal + Dice + sampler?" (0.876) |
| B3 | k32, k32 + cosine50 (0.896), transformer variants (notes 1a side rows, 2d) | "Why not the biggest model?" |
| B4 | Post-processing variants: closing, distance rule, hole filling (notes 2e) | "Why such a simple post-processing?" |
| B5 | Worst CV patients: Patient_14 (trachea blob), Patient_24 (esophagus 0.50), images | "Where does it still fail?" |
| B6 | Training cost: native 2.6×, k16 ~3 h / 25 epochs, final ~6–9 h on one GPU | "How expensive is it?" |

---

## Figures to produce (assign before the slide deadline)

| Slide | Figure | Data source | Suggested owner |
|---|---|---|---|
| 2 | CT slice with 4 organs + midterm per-organ bars | any patient + `results/current/holdout40-f0-s42` | Elena |
| 3 | Esophagus crop at 1.95 vs 0.98 mm | preprocessing cache, one patient | Elena |
| 4, 5, 7, 9, 12 | Ladder strip (mean + esophagus Dice per step) | notes section 0 ladder table | one person for all, same style |
| 5 | Aug vs no-aug bars + seed dot plot; 4 augmented example slices | notes 2b; `segpipe/augment.py` on one slice | Githa |
| 6 | Pixel share per class in a slice + loss table | GT slices; notes 2c | Puck |
| 7 | Slice-share bars before/after sampler | sampler printout (37/53 → 18/75 %) | David |
| 8 | Params vs Dice curve; ENet sketch with attention block | notes 2d | Junis |
| 9 | Val-Dice curves k16 constant vs k16 cosine50 | W&B (`combo_core_ce_k16`, `combo_core_ce_k16_cosine50`) | Junis |
| 10 | Stray trachea blob before/after (3D or coronal) + HD95 bars | final model seed 43 predictions, Patient_22 | Britt |
| 11 | HD95 bars CE vs boundary w5 (raw and post) | notes 2e | Britt |
| 12 | Pipeline diagram + qualitative Patient_28 midterm vs final | predictions of both models | part 5 speaker |
| 13 | Per-fold bars + per-organ holdout vs CV | notes 1d | part 5 speaker |
