# Pose-derived microstructure of the social versus food preference test (SFPT)

Analysis code accompanying the manuscript:

> Surana AA\*, Biju N\*, Pal S, Neumann ID, Rappeneau V. **Pose-derived microstructure resolves how hunger and social isolation shape social–food choice in male mice.** *(journal / DOI to be added)*
> \*Equal contribution.

Adult male CD-1 mice, group-housed (GH) or single-housed (SH), were tested in the SFPT either fed ad libitum (AL) or after 16 h of food deprivation (FD). Behaviour was tracked with DeepLabCut, converted into per-frame and per-animal features, and analysed for (i) microstructure of approach, orientation and engagement, (ii) time courses across the 10-min session, (iii) cross-validated decoding of feeding state and housing condition, and (iv) Zif268 immunofluorescence across the SFPT, social-only (ST) and food-only (FT) tests.

The four experimental groups are labelled `GH AL`, `GH FD`, `SH AL` and `SH FD` throughout the code.

---

## Contents

| File | Type | What it does |
|---|---|---|
| `total_distance_travelled.py` | script | Total distance travelled per video (px and cm) from the centroid trace |
| `transition_probabilities.ipynb` | notebook | Chamber-state sequences (SC / FC / Neutral), first- and second-order transition probabilities, switch probabilities, group statistics and plots |
| `02_plot_polar_angles_and_absolute_angle_boxplots.ipynb` | notebook | Head-orientation analysis: polar histograms of signed angle to each stimulus, orientation index, per-video summaries and boxplots |
| `paired_approach_vs_baseline_velocity.ipynb` | notebook | Paired comparison of approach vs baseline speed toward the social and food stimulus per treatment group |
| `time_course_speed_v2.ipynb` | notebook | 60-s binned approach-speed time course, per-video fits, and treatment × time GEE / mixed-model tests |
| `time_course_for_oi_v3.ipynb` | notebook | 60-s binned orientation-index and facing time course, same statistical framework as above |
| `roi_bout_analysis.py` | script | Converts per-frame tracking CSVs into bout tables (SS / FS / SC / FC / Neutral) and ethogram plots, one set per video |
| `bout_and_preference_analysis_time.py` | script | Per-animal bout and preference metrics from the bout tables, 2 × 2 (housing × feeding state) statistics, and time-binned preference and bout-duration analysis (mixed models and GEE) |
| `sfpt_decoder_final.py` | script | Random-forest decoders on pose-derived features (feeding state, housing, 4-class) and main/supplementary figures |
| `sfpt_boris_decoder.py` | script | Same decoding pipeline on manually scored (BORIS) measures only, with optional overlay on the pose-based results |
| `zif268_atlas_heatmaps_v2.ipynb` | notebook | One-way ANOVA + Tukey across SFPT/ST/FT per region and group, with compact letters, drawn on Allen Mouse Brain Atlas coronal sections |

---

## Analysis pipeline

```
SFPT videos ──► DeepLabCut (8 key points) ──► per-frame "augmented" CSVs
                                              (centroid, distances, angles, zone flags)
                                                        │
        ┌───────────────────────┬───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼                       ▼
 total_distance_         transition_             polar / orientation      approach speed
 travelled.py            probabilities           notebook                 notebooks
        │                       │                       │                       │
        └───────────────────────┴───────────┬───────────┴───────────────────────┘
                                            ▼
                         per-animal feature sheets (Excel, one tab per feature family)
                                            ▼
                      sfpt_decoder_final.py / sfpt_boris_decoder.py


augmented CSVs ──► roi_bout_analysis.py ──► bout tables (SS / FS / SC / FC / Neutral)
                                                        │
                                                        ▼
                                      bout_and_preference_analysis_time.py


Zif268 counts (Excel) ──► zif268_atlas_heatmaps_v2.ipynb
```

> **Not included in this repository:** the DeepLabCut project and trained network, the step that converts DLC output into the "augmented" per-frame CSVs (`all_augmented_csv/results_with_vertices/`, including ROI vertices, zone flags and angle columns), and the parametric sweep that produced `approach_speed_summary_all_radii.csv`. *(Edit this note if these are added to the repository.)*

---

## Installation

Python ≥ 3.10 is recommended.

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pandas numpy scipy statsmodels patsy scikit-learn joblib \
            matplotlib seaborn openpyxl jupyter
```

Only for `zif268_atlas_heatmaps_v2.ipynb`:

```bash
pip install "zarr>=3" brainglobe-heatmap brainglobe-atlasapi
```

The first run downloads the Allen Mouse Brain Atlas (a few hundred MB, cached in `~/.brainglobe`). Run this notebook locally with a normal internet connection. If you see `GroupNotFoundError`, make sure `zarr>=3` is installed and restart the kernel.

Pose estimation itself used DeepLabCut v3.0.0rc9 and is not needed to run the code here.

---

## Input data

Most scripts expect the following (paths are set in the configuration cell at the top of each notebook or script):

The two bout scripts (`roi_bout_analysis.py`, `bout_and_preference_analysis_time.py`) are interactive and ask for their paths when run.

| Input | Used by | Description |
|---|---|---|
| `all_augmented_csv/results_with_vertices/*.csv` | distance, transitions, polar, time-course notebooks, roi_bout_analysis.py | Per-frame tracking output, 25 fps. Columns used include `centroid_x_final` / `center_x_final`, `centroid_y_final` / `center_y_final`, speed columns (`speed_px_per_frame`, `speed_cm_per_frame`, or the per-second equivalents), `in_SC`, `in_FC`, `abs_angle_to_SS`, `abs_angle_to_FS`, `signed_angle_to_SS`, `signed_angle_to_FS` . For `roi_bout_analysis.py`: ROI polygons in `all_roi_vertices` and/or `roi_<NAME>_vertices`, coordinates as `<bodypart>_x_final` / `<bodypart>_y_final`, and optionally `nose_quality` / `center_quality`|
| `housing_food_condition.xlsx` | polar, time-course notebooks | Master sheet. First sheet with columns `Video`, `Treatment` (`GH AL`, `GH FD`, `SH AL`, `SH FD`), `Housing`, `Food` |
| `Figure1_master.csv` | `transition_probabilities.ipynb` | Master sheet with video names and treatment labels (optional but recommended) |
| `approach_speed_summary_all_radii.csv` | `paired_approach_vs_baseline_velocity.ipynb` | One row per video, stimulus and interaction radius, with `approach_speed_SS/FS` and `baseline_speed_SS/FS` |
| `All_data_for_the_MS.xlsx` | both decoder scripts | One animal per row, one tab per feature family (see below) |
| `IHC_effect_size_analysis.xlsx`, sheet `Raw IHC extracted` | `zif268_atlas_heatmaps_v2.ipynb` | One row per animal: group, region, context (SFPT / ST / FT) and % Zif268+ cells |
| Start-frame file (`.xlsx` / `.csv`, optional) | `roi_bout_analysis.py` | One row per video: a video column (`Video file`, `Movie`, `Video`, `File` or `Filename`) and a start-frame column (`Start frame` or `Start`) |
| Group / master file (`.xlsx` / `.csv`, optional) | `roi_bout_analysis.py` | Video key plus `housing` and `food_state` (or a ready-made `group` column); used only to sort outputs into one folder per group |
| `<key>_bouts_detailed.csv` | `bout_and_preference_analysis_time.py` | Output of `roi_bout_analysis.py`. One row per bout with columns `roi` (`SS`, `FS`, `SC`, `FC`, `Neutral`), `start_time_s`, `end_time_s` |
| `<key>_bouts_collapsed_SC_FC_Neutral.csv` | `bout_and_preference_analysis_time.py` | Output of `roi_bout_analysis.py`. Same format, states `SC`, `FC`, `Neutral` only. Every key needs both files; keys with only one are skipped with a warning |
| Group file (`.csv` / `.xlsx`) | `bout_and_preference_analysis_time.py` | One row per key with columns `key`, `housing` (`GH` / `SH`), `food_state` (`AL` / `FD`). Keys must be unique |

**File naming.** Per-frame CSVs are matched to the master sheet by video name: `aug_<video>…csv` is matched to the longest `Video` entry that the name starts with.

For the bout scripts, a *key* (e.g. `MVI_0006_roi1`) is one animal and one 600-s window. It is the file-name prefix before `DLC` in the tracking CSV and before `_bouts_detailed.csv` in the bout tables, and it must match the `key` column of the group file. Nothing is averaged or merged across keys.

**Decoder workbook layout** (`All_data_for_the_MS.xlsx`, tab positions are 1-based):

| Tab | Content | Feature family |
|---|---|---|
| 1 | Investigation time, preference index (RDI), bouts, entries | Investigation time / preference; Investigation bouts / entries |
| 4 | Chamber transitions and switch probabilities | Chamber transitions |
| 5 | Head orientation (mean / SD of absolute angle, facing) | Head orientation |
| 6 | Total distance (output of `total_distance_travelled.py`) | Locomotion |
| 7 | Approach, interaction and baseline speed | Approach / interaction speed |
| 8 | Approach completion | Approach completion |

All tabs must carry the same `Treatment` labels per animal; `sfpt_decoder_final.py` stops with an error if they disagree.


---

## Usage

### 1. Total distance travelled

```bash
python total_distance_travelled.py
```

Edit the folder passed to `process_centroid_distances()` at the bottom of the file (default `all_augmented_csv/results_with_vertices`). The video ID is parsed from the file name between `aug_` and `DLC`. Distance is the summed frame-to-frame centroid displacement, converted to cm with the ratio of the cm and px speed columns.

Output: `centroid_total_distances.csv` with `id`, `total_distance_px`, `cm_per_px`, `total_distance_cm`.

### 2. Chamber transitions (`transition_probabilities.ipynb`)

Set `INPUT_PATH`, `MASTER_SHEET_PATH`, `OUTPUT_DIR`, `GROUP_BY` and `N_FRAMES_WINDOW` (15,000 frames = 10 min at 25 fps) in the configuration cell, then run all cells.

Each frame is assigned to one of three states: `SC` (`in_SC == 1`), `FC` (`in_FC == 1` and not `SC`) or `Neutral` (everything else). Consecutive duplicates are collapsed, and first-order transition matrices, two-step transitions through the neutral zone, and SC ↔ FC switch probabilities are computed per video and merged per group. Output includes per-video and merged transition CSVs, `per_video_transition_summary.csv`, matrix PDFs, and statistical comparisons of housing and feeding effects (effect size: Cohen's *h*).

### 3. Head orientation (`02_plot_polar_angles_and_absolute_angle_boxplots.ipynb`)

Set `MASTER_SHEET_PATH`, `AUGMENTED_CSV_FOLDER` and `OUTPUT_PLOTS_FOLDER`. Produces 2 × 2 polar histograms (1° bins) of signed head angle to the social (SS) and food (FS) stimulus per group, per-video orientation summaries and boxplots. Orientation index is the cosine of the absolute angle; facing is ≤ 45°.

The first cell of this notebook carries a note that it still needs editing for the five-group layout. Check the plotted groups before using the output.

### 4. Approach vs baseline speed (`paired_approach_vs_baseline_velocity.ipynb`)

Reads `approach_speed_summary_all_radii.csv`, keeps `RADIUS = 6` cm and plots paired baseline and approach speeds per animal for each treatment and stimulus, with paired tests (Shapiro–Wilk for normality, then paired *t*-test or Wilcoxon).

### 5. Time courses (`time_course_speed_v2.ipynb`, `time_course_for_oi_v3.ipynb`)

Both notebooks follow the same three stages and write to `advanced_analysis_plots/`:

1. **Binning.** Per-frame masks are averaged in 60-s bins (`TIME_BIN_SECONDS = 60`, `FPS = 25`) → `time_course_speed.csv` / `time_course_orientation.csv` (one row per video per bin).
2. **Per-video fits.** Slope, R² and *p* of the within-video time trend (≥ 3 valid bins required) → `*_fit_quality.csv`.
3. **Treatment × time tests.** GEE (primary, robust SEs) with a MixedLM cross-check, `outcome ~ time_bin * C(Treatment)` clustered on video. `GH AL` is the reference group. Interaction contrasts, a global Wald test, simple slopes per group and Holm correction are written to `*_treatment_trends.csv`. Ribbon, spaghetti and forest plots follow.

Settings used by the speed notebook: approach requires radial closing velocity ≥ 0.5 cm/s, body speed ≥ 1 cm/s, distance within `INTERACTION_RADIUS_CM = 6`, and `abs_angle ≤ 45°`; distance smoothing window = 5 frames.

### 6. Decoding

```bash
# pose-derived features (10 CV repeats, 200 permutations by default)
python sfpt_decoder_final.py --xlsx All_data_for_the_MS.xlsx --outdir decoder_figures

# manually scored (BORIS) measures, optionally overlaid on the pose-based results
python sfpt_boris_decoder.py --xlsx All_data_for_the_MS.xlsx --outdir decoder_boris_only \
       --compare decoder_figures/task_metrics.csv

# quick smoke test
python sfpt_decoder_final.py --xlsx All_data_for_the_MS.xlsx --repeats 1 --n-perm 5 --imp-perm 1
```

**Method.** Random forest (500 trees, `max_features="sqrt"`, balanced subsample class weights), repeated stratified 5-fold cross-validation with one observation per mouse, median imputation inside each training fold, and a label-permutation test against chance. Metrics are AUC and balanced accuracy. Feature-family importance is the drop in balanced accuracy after jointly shuffling a family in held-out folds.

**Tasks.** Feeding (all / within GH / within SH), housing (all / within AL / within FD), and a 4-class decoder.

**Options (both scripts).**

| Flag | Default | Meaning |
|---|---|---|
| `--xlsx` | `All_data_for_the_MS.xlsx` | Input workbook |
| `--outdir` | `decoder_figures` / `decoder_boris_only` | Output folder |
| `--n-perm` | 200 | Label permutations per task |
| `--repeats` | 10 | Cross-validation repeats |
| `--imp-perm` | 5 | Shuffles per feature per fold for importance |
| `--exclude` | none | Extra feature columns to drop (exact names) |
| `--n-jobs` | -1 | Parallel workers |

Additional flags: `sfpt_decoder_final.py` has `--supp-start` (number of first supplementary figure) and `--no-pca`; `sfpt_boris_decoder.py` has `--sheet1`, `--id-col`, `--treat-col`, `--compare` and `--supp-start`.

**Outputs.** Panel PDFs (`Figure_main_A_accuracy.pdf`, `…_B_family.pdf`, `…_C_heatmap.pdf`, or `Figure_boris_*` for the BORIS decoder), `Supplementary_*decoders.pdf`, PNG previews, and CSVs with all numbers behind the figures (task metrics, feature and family importances, per-mouse probabilities, features used). PDF fonts are embedded as editable text (Type 42).

Columns that Excel has converted to dates are detected and set to missing; columns with > 5% corrupted cells are dropped and reported at run time. Total-count and duplicate columns (e.g. `Total_Transitions`, `median_abs_angle_*`) are excluded by default, see `EXCLUDE_COLUMNS` in `sfpt_decoder_final.py`.

### 7. Zif268 atlas heatmaps (`zif268_atlas_heatmaps_v2.ipynb`)

Edit the **Settings** cell (`RAW_XLSX`, `RAW_SHEET`, `GROUPS`, `ACRONYM`, `SECTIONS`, `VMIN/VMAX`), then run all cells. For each region × group, a one-way ANOVA across SFPT, ST and FT is followed by Tukey HSD and a compact letter display. No condition is treated as reference, and letters are drawn only when the ANOVA is significant. Three figures (one per condition) show coronal sections colored by mean % Zif268+ cells. The bregma levels in `SECTIONS` should be checked against your own histology.

The manuscript additionally uses Welch's ANOVA with Games–Howell where variances are unequal; that step is not implemented in this notebook.

### 8. Bout extraction (`roi_bout_analysis.py`)

```bash
python roi_bout_analysis.py
```

The script is interactive and prompts for:

1. the folder with the tracking CSVs and an output folder (default `<input folder>/roi_bout_output`),
2. the frame rate (default 25 fps),
3. optionally, new values for the grace period, gap-fraction threshold and analysis window,
4. which body part(s) define "inside an ROI" for each region, and the rule: `cascade` (first body part with a valid coordinate), `any`, `all` or `single`,
5. a start-frame file, or whether to enter start frames manually or analyse the whole CSV,
6. optionally, a group file to sort outputs into one folder per group.

ROI polygons are read from each CSV, so no separate ROI files are needed.

**Method.** Each frame is assigned to one state: `SS` / `FS` (small stimulus zones), `SC` / `FC` (chambers) or `Neutral`. If a frame is inside several regions, the small stimulus zone wins over the chamber. Exits of ≤ 10 frames (`GRACE_FRAMES`) are absorbed into the surrounding bout. The analysed window is 15,000 frames (10 min at 25 fps) from the start frame. A bout is flagged `low_tracking_quality` if more than 50% of its frames are gap frames in the `nose` or `center` tracking.

**Outputs** (per video, named by the key, e.g. `MVI_0006_roi1`)

| File | Content |
|---|---|
| `<key>_bouts_detailed.csv` | Bouts in the states `SS`, `FS`, `SC`, `FC`, `Neutral`, with start and end frame and time, duration, previous and next state, % gap frames, tracking-quality flag and truncation flags |
| `<key>_bouts_collapsed_SC_FC_Neutral.csv` | Same, with `SS` merged into `SC` and `FS` into `FC` |
| `<key>_ethogram.png` | Timeline strip with the time spent in each region (titled with the group name if a group file was given) |
| `run_report.csv` | One row per video: group, status (processed / skipped / failed), start frame, number of bouts and any flags |

If you give a group file, outputs are sorted into one subfolder per group (`GHAL/`, `GHFD/`, …; keys not found go to `UNASSIGNED/`). `bout_and_preference_analysis_time.py` reads bout CSVs from a single folder, so either skip the group file or copy the bout CSVs into one folder before running it.

### 9. Bout and preference analysis (`bout_and_preference_analysis_time.py`)

```bash
python bout_and_preference_analysis_time.py
```

The script is interactive and prompts for:

1. the folder with the bout CSVs from `roi_bout_analysis.py` (all in one folder),
2. an output folder (default `<bout folder>/bout_and_preference_output`),
3. the group file (press Enter to skip all group statistics and only write per-animal tables),
4. the time-bin length in seconds (default 60).

The script can also be called from Python with `run_pipeline(bout_dir, out_dir, group_file, bin_seconds=...)`.

**Group labels.** Unlike the other scripts, this one does not read a `Treatment` column. It builds the four groups from two columns of the group file, `housing` (`GH` / `SH`) and `food_state` (`AL` / `FD`), and refers to them as `GHAL`, `GHFD`, `SHAL` and `SHFD` (no space) in its output tables. Figure legends use the `GH AL` style.

**Per-animal metrics** (`animal_metrics.csv`)

| Metric | Meaning |
|---|---|
| `p_choose_fc`, `n_choice_points` | Share of exits from the Neutral zone that go to the food chamber, and the number of such exits |
| `history_phi` (+ `chi2`, `p`) | Whether the previous chamber predicts the next one |
| `ss_*` / `fs_*` bout metrics | Number of bouts, mean duration, first- and second-half mean duration, within-session slope |
| `pref_bin_slope`, `ss_/fs_logdur_bin_slope` | Change in preference and in log bout duration across time bins |

**Time bins.** Bins start at the beginning of each animal's window. A window up to 1 s shorter than a whole number of bins still gets that last bin (`BIN_END_TOLERANCE_S`). A longer window has its remainder added to the last bin, so no data are dropped. Per bin:

- **Preference index** = (SS time − FS time) / (SS time + FS time), using stimulus-zone time only (+1 = social, −1 = food). It is `NaN` if the animal spent no time in either zone.
- **Mean bout duration** of SS and FS bouts that *start* in the bin (`NaN` if none started).

**Statistics**

- *Group effects:* 2 × 2 ANOVA (Type II, housing × food state) per metric, with assumption checks, HC3-robust and permutation p-values (5,000 permutations), Benjamini–Hochberg q-values, an outlier sensitivity test (|studentized residual| > 3), and four planned comparisons (Welch *t*-test, Holm correction). Durations are analysed on the log scale and counts as log(1 + x). A metric is tested only if each of the four groups has ≥ 5 animals; the others are listed in `metrics_not_tested.csv`. `history_phi` also gets a weighted (WLS) sensitivity test weighted by `n_choice_points`.
- *Time courses:* mixed models and GEE (`housing × food state × time`, animals as clusters). Preference uses a Gaussian family, bout durations a Gamma family with log link. The working correlation is exchangeable (a QIC comparison with AR(1) is saved). Covariance is the bias-reduced (Mancl–DeRouen) sandwich estimator, falling back to the robust one. Sensitivity analyses: preference restricted to bins with ≥ 5 s in SS + FS, and durations at the individual-bout level.

**Outputs** (in the output folder)

| File | Content |
|---|---|
| `animal_metrics.csv`, `binned_long.csv` | Per-animal metrics and one row per animal per bin (merged with group labels) |
| `analysis_settings.json` | All upper-case parameters used for the run |
| `group_2x2_anova.csv`, `assumption_checks.csv`, `planned_comparisons.csv`, `outlier_sensitivity.csv`, `metrics_not_tested.csv` | Group statistics |
| `timecourse_mixed_models.csv`, `timecourse_bin_comparisons.csv`, `timecourse_crosscheck.csv` | Mixed-model time-course results and a cross-check against the per-animal slope ANOVA |
| `timecourse_gee_model.csv`, `…_slopes.csv`, `…_slope_contrasts.csv`, `…_time_shape_qic.csv` | GEE coefficients, per-group slopes, slope contrasts and time-shape comparison |
| `timecourse_<response>_points.{png,pdf}`, `timecourse_<response>_GEE.{png,pdf}` | Group means ± SEM per bin, without and with GEE fit and 95% band, for the preference index and SS / FS bout duration |
| `group_plots/` | One boxplot per tested metric (`<metric>.png`) with individual animals, n per group and the housing / food / interaction p-values |
| `diagnostics/` | Residual QQ and residuals-vs-fitted plots for each ANOVA (`<metric>_diagnostics.png`) and for each time-course mixed model (`timecourse_<response>_diagnostics.png`) |
| `per_key/` | Per-animal summary PNG and choice-point CSV |

Figure and model options (fixed preference y-axis of −1 to +1, shared duration y-axis, GEE line span, error bars, minimum animals per point) are set in the `TUNABLE PARAMETERS` block at the top of the script.

---

## Relationship to the manuscript

| Manuscript item | Code |
|---|---|
| Fig. 1G (chamber transition probabilities) | `transition_probabilities.ipynb` |
| Fig. 1I (total distance) | `total_distance_travelled.py` |
| Fig. 2B,C (approach speed vs baseline) | `paired_approach_vs_baseline_velocity.ipynb` |
| Fig. 2E–J (orientation index, polar plots, facing) | `02_plot_polar_angles_and_absolute_angle_boxplots.ipynb` |
| Fig. 2K,L (preference and bout-duration time courses) | `bout_and_preference_analysis_time.py`, after `roi_bout_analysis.py` |
| Fig. 2M (time courses of bouts and orientation) | `time_course_for_oi_v3.ipynb` (orientation); approach-speed analogue in `time_course_speed_v2.ipynb` |
| Fig. 3 (decoding) | `sfpt_decoder_final.py` (A–D), `sfpt_boris_decoder.py` (E–H) |
| Fig. 4 / Zif268 | `zif268_atlas_heatmaps_v2.ipynb` (atlas visualisation); box plots were made in GraphPad Prism |

Stimulus abbreviations in the code: **SS** = social stimulus , **FS** = food stimulus , **SC** = social chamber, **FC** = food chamber.

Statistics reported in the paper that were run in GraphPad Prism (ANOVAs, Tukey tests, planned *t*-tests) are not reproduced in this repository.

---

## Reproducibility notes

- Random forests use `random_state=0` and CV splits use `SEED = 0`.
- Sample sizes: GH-AL n = 10, GH-FD n = 10, SH-AL n = 11, SH-FD n = 18 (49 mice in total; one observation per mouse in the decoders).
- Frame rate is 25 fps; the test window is 10 min (15,000 frames).
- Several notebooks contain hard-coded relative paths in their configuration cells. Adjust these to your folder layout.
- The permutation tests in `bout_and_preference_analysis_time.py` use `PERM_SEED = 12345` and 5,000 permutations. All settings of a run are saved to `analysis_settings.json`.

---

## Data availability

*Repository name, DOI and reference number to be added.*

## License

*To be added.*

## Contact

Virginie Rappeneau, Department of Behavioural and Molecular Neurobiology, University of Regensburg — virginie.rappeneau@biologie.uni-regensburg.de

## Citation

If you use this code, please cite the manuscript above *(update with the final reference once published)*.
