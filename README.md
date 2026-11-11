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

| Input | Used by | Description |
|---|---|---|
| `all_augmented_csv/results_with_vertices/*.csv` | distance, transitions, polar, time-course notebooks | Per-frame tracking output, 25 fps. Columns used include `centroid_x_final` / `center_x_final`, `centroid_y_final` / `center_y_final`, speed columns (`speed_px_per_frame`, `speed_cm_per_frame`, or the per-second equivalents), `in_SC`, `in_FC`, `abs_angle_to_SS`, `abs_angle_to_FS`, `signed_angle_to_SS`, `signed_angle_to_FS` |
| `housing_food_condition.xlsx` | polar, time-course notebooks | Master sheet. First sheet with columns `Video`, `Treatment` (`GH AL`, `GH FD`, `SH AL`, `SH FD`), `Housing`, `Food` |
| `Figure1_master.csv` | `transition_probabilities.ipynb` | Master sheet with video names and treatment labels (optional but recommended) |
| `approach_speed_summary_all_radii.csv` | `paired_approach_vs_baseline_velocity.ipynb` | One row per video, stimulus and interaction radius, with `approach_speed_SS/FS` and `baseline_speed_SS/FS` |
| `All_data_for_the_MS.xlsx` | both decoder scripts | One animal per row, one tab per feature family (see below) |
| `IHC_effect_size_analysis.xlsx`, sheet `Raw IHC extracted` | `zif268_atlas_heatmaps_v2.ipynb` | One row per animal: group, region, context (SFPT / ST / FT) and % Zif268+ cells |

**File naming.** Per-frame CSVs are matched to the master sheet by video name: `aug_<video>…csv` is matched to the longest `Video` entry that the name starts with.

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

---

## Relationship to the manuscript

| Manuscript item | Code |
|---|---|
| Fig. 1G (chamber transition probabilities) | `transition_probabilities.ipynb` |
| Fig. 1I (total distance) | `total_distance_travelled.py` |
| Fig. 2B,C (approach speed vs baseline) | `paired_approach_vs_baseline_velocity.ipynb` |
| Fig. 2E–J (orientation index, polar plots, facing) | `02_plot_polar_angles_and_absolute_angle_boxplots.ipynb` |
| Fig. 2L,M (time courses of bouts and orientation) | `time_course_for_oi_v3.ipynb` (orientation); approach-speed analogue in `time_course_speed_v2.ipynb` |
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

---

## Data availability

*Repository name, DOI and reference number to be added.*

## License

*To be added.*

## Contact

Virginie Rappeneau, Department of Behavioural and Molecular Neurobiology, University of Regensburg — virginie.rappeneau@biologie.uni-regensburg.de

## Citation

If you use this code, please cite the manuscript above *(update with the final reference once published)*.
