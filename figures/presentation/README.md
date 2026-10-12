# Presentation figures

Candidates for the slides; pick favourites. PNG for the slides, SVG to scale without blur. Regenerate with:

```
ai4mi/bin/python figures/make_presentation_figs.py          # ladder, CV, slides 10–12
ai4mi/bin/python figures/make_presentation_figs_extra.py    # everything in extra/
```

Numbers: 3D, 8 holdout patients, post-processed, mean of seeds, unless the figure says raw / 1 seed.
Colours: blue = mean Dice, orange = esophagus (charts); organ outlines use one fixed colour per organ.
CT images are shown in the radiological view (front up, the patient's left on the image's right).

## Main figures (`figures/presentation/`)

| File | Slide | Shows |
|---|---|---|
| `ladder_full` | 12 / backup | All 6 steps, mean + esophagus Dice, all values |
| `ladder_slide04_native` … `_slide09_final` | 4, 5, 7, 9 | Same ladder growing per speaker, current step highlighted |
| `slide12_ladder_dice_hd95` | 12 | Dice ladder + HD95 per step underneath |
| `slide13_cv` (`cv_folds`, `cv_organs`) | 13 | CV per fold vs holdout; per-organ drop holdout → CV |
| `slide10_stray_blobs` | 10 | 3 stray-blob cases: Dice +0.01, HD95 271/183/82 → 5/6/3 mm |
| `slide11_boundary_loss` | 11 | HD95 raw vs post: CE vs boundary loss |

## Extra candidates (`figures/presentation/extra/`)

| File | Speaker / slide | Shows |
|---|---|---|
| `elena_01_midterm_per_organ` | Elena, 2 | Midterm per-organ Dice, esophagus highlighted |
| `elena_02_ct_slices_organs` | Elena, 2 / title | Two CT slices with the 4 organ outlines |
| `elena_03_esophagus_at_two_resolutions` | Elena, 3 | Same esophagus at 1.95 vs 0.98 mm: 53 vs 237 pixels |
| `elena_04_preprocessing_ablation` | Elena, 4 / backup | Δ Dice of every preprocessing change vs midterm |
| `elena_05_resolution_per_organ` | Elena, 4 | Dice per organ at 1.95 → 1.5 → 0.98 mm; esophagus +0.098 |
| `elena_06_qualitative_midterm_vs_native` | Elena, 4 | GT vs midterm vs native prediction, same slice |
| `githa_07_augmentation_by_resolution` | Githa, 5 | Augmentation helps at native, not at 1.95 mm |
| `githa_08_seed_spread` | Githa, 5 | Seed spread 0.026 → 0.007 with augmentation |
| `githa_09_augmentation_examples` | Githa, 5 | One slice under each augmentation; roll wraps around |
| `puck_10_voxel_share` | Puck, 6 | Esophagus = 0.09% of voxels (1 in ~1,170) |
| `puck_11_loss_comparison` | Puck, 6 | CE / Tversky / focal + Dice / CE + Dice, seeds shown |
| `puck_12_per_patient_esophagus` | Puck, 6 | Esophagus per patient, CE → CE + Dice: up in 8/8 |
| `david_13_sampler_slice_share` | David, 7 | Slice share before/after the sampler |
| `david_14_sampler_fixes_hd95` | David, 7 | CE + Dice vs + sampler: Dice and HD95 per seed |
| `junis_15_width_curve` | Junis, 8 | k8 → k16 → k32, constant vs cosine 50 |
| `junis_16_transformer` | Junis, 8 | Transformer variants vs plain k32 |
| `junis_17_val_curves_constant_vs_cosine` | Junis, 9 | Validation curves: jumpy constant lr vs flat cosine |
| `junis_18_lr_schedules` | Junis, 9 / backup | The three learning-rate schedules |
| `britt_21_hd95_raw_vs_post_per_model` | Britt, 10 / backup | Raw → post HD95 for 6 models |
| `britt_22_postprocessing_variants` | Britt, 10 / backup | Fancier post-processing: helps weak model, hurts strong |
| `britt_23_one_seed_can_lie` | Britt, 10 | Skip-esophagus: best on seed 42, 70 mm on seed 43 |
| `britt_24_stray_blob_projection` | Britt, 10 | Coronal projection: GT / raw (stray pieces) / post-processed |
| `final_25_per_organ_midterm_baseline_final` | Part 5, 12 | Per organ: midterm → CE baseline → final |
| `final_26_per_patient_midterm_vs_final` | Part 5, 12 | All 8 patients better, midterm → final |
| `final_27_cv_per_patient` | Part 5, 13 / B5 | All 40 patients by fold; worst labelled |
| `final_28_starter_vs_final_cv` | Part 5, 13 | Starter → final on unseen folds (raw), +0.155 to +0.211 |
| `final_29_cv_hd95_raw_vs_post` | Part 5, 14 | In CV the filter makes esophagus HD95 worse |
| `final_30_pipeline_diagram` | Part 5, 12 | Final pipeline as 7 boxes |

Not possible from the local data: a qualitative image of the **final model** (its predictions are on
Snellius: `results/combo_core_ce_k16_cosine50/holdout40-f0-s4*/best_epoch/val/`). Copy that folder to a
laptop and `elena_06` / `britt_24` can be redone for the final model.
