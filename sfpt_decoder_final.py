#!/usr/bin/env python3
"""
SFPT random-forest decoders -> publication figures (main PDF + supplementary PDF)
==================================================================================
Self-contained: needs only the Excel file (no other script).

Analyses (random forest, repeated stratified 5-fold CV, imputation inside each fold,
label-permutation test against chance):
  * FEEDING decoder  : AL vs FD         (all mice, and within GH / within SH)
  * HOUSING decoder  : GH vs SH         (all mice, and within AL / within FD)
  * 4-class decoder  : GH AL / GH FD / SH AL / SH FD   (supplement)
  * Held-out permutation importance (single features and feature families) for the
    two pooled binary decoders; unsupervised PCA (supplement, exploratory)

Outputs (in --outdir)
  Figure_main_A_accuracy.pdf        Panel A alone
  Figure_main_B_family.pdf          Panel B alone
  Figure_main_C_heatmap.pdf         Panel C alone
  Supplementary_decoders.pdf        S-fig 1: confusion matrices + top-12 features
                                    S-fig 2a: 4-class decoder confusion-matrix heatmap (own page)
                                    S-fig 2:  4-class decoder permutation test + confidence
                                    S-fig 3:  PCA (exploratory)
  previews/*.png, *.csv             quick-look images and all numbers behind the figures

Usage
  python sfpt_decoder_figures.py --xlsx All_data_for_the_MS.xlsx
  python sfpt_decoder_figures.py --xlsx file.xlsx --n-perm 500 --repeats 10 --supp-start 3
  python sfpt_decoder_figures.py --xlsx file.xlsx --repeats 1 --n-perm 5 --imp-perm 1   # quick test

Requires: pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
"""
from __future__ import annotations

import argparse
import datetime as dt
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,           # editable text in Illustrator
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.8})

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CLASSES = ["GH AL", "GH FD", "SH AL", "SH FD"]
GROUP_COLORS = {"GH AL": "#c8581a", "GH FD": "#e8a317", "SH AL": "#1f5a85", "SH FD": "#3cb8b2"}
FEED_COLOR, HOUS_COLOR, MULTI_COLOR = "#d95f02", "#1b9e77", "#555555"
FAM_COLORS = {
    "Investigation time / preference": "#4c78a8", "Investigation bouts / entries": "#72b7b2",
    "Chamber transitions": "#f58518", "Head orientation": "#54a24b",
    "Approach / interaction speed": "#b279a2", "Approach completion": "#e45756",
    "Locomotion": "#9d755d",
}

SHEETS = {1: "behavior", 4: "transition", 5: "head", 6: "distance",
          7: "approach_speed", 8: "approach_completion"}      # Excel tab positions (1-based); sheet 2 (BORIS) not used

# Extra columns to leave out of every analysis (exact column names). Can also be given as
# --exclude "Col A" "Col B" on the command line.
EXCLUDE_COLUMNS: set[str] = {
    # Transition totals (sums / duplicates of other transition columns)
    "Total_Transitions",        # == Total_Visits - 1
    "Neutral_Visits",
    "Total_to_Neutral",
    "Total_from_Neutral",
    "Total_Neutral_mediated",
    "Neutral_to_SC",
    "Neutral_to_FC",
    # Pose-tracker sampling artefacts
    "n_angles_SS",
    "n_angles_FS",
    # Central-tendency duplicates: keep mean_abs_angle_*, drop median_abs_angle_*
    "median_abs_angle_SS",
    "median_abs_angle_FS",
}
CORRUPT_DROP_FRAC = 0.05       # drop a column if >5% of its cells were mangled into dates by Excel
RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=1,
                 class_weight="balanced_subsample", random_state=0, n_jobs=1)
N_SPLITS, SEED = 5, 0

NEVER_FEATURES = {"Video", "video", "Animal ID", "Treatment", "Housing", "Food", "Start Frame",
                  "PI", "radius", "id", "mouse", "aid", "Unnamed: 4",
                  "ROIArea_SC (cm2)", "ROIArea_FC (cm2)", "ROIArea_SS (cm2)", "ROIArea_FS (cm2)",
                  "total_distance_px", "cm_per_px", "total_distance_m"}   # labels / IDs / camera geometry

# How many top features to show per pooled binary decoder in the Panel C heatmap
TOP_N_HEATMAP = 3
ALSO_WRITE_COMBINED_MAIN = False        # True -> also emit old 3-panel-wide Figure_main_decoders.pdf
# --------------------------------------------------------------------------- #
# Pretty names for the raw column names. Anything not listed keeps its raw name
# (a one-line warning is printed so you can extend this).
# SC = social chamber, FC = food chamber, SS = social-side, FS = food-side.
# --------------------------------------------------------------------------- #
FEATURE_LABELS: dict[str, str] = {
    # -------- the six that are currently in Panel C ----------------------------
    "mean_abs_angle_FS":  "Mean absolute angle to FS",   # <- change if you prefer
    "frac_approach_of_FC": "Fraction of time approaching FS",
    "interacting_speed_FS": "Interaction speed with FS",
    "approach_speed_FS":   "Approach speed to FS",
    "baseline_speed_FS":   "Baseline speed in FC",
    "std_abs_angle_SS":    "SD of absolute angle to SS",
    # -------- other commonly used metrics (extend as needed) -------------------
    "mean_abs_angle_SS":   "Mean head deviation (social side)",
    "std_abs_angle_FS":    "SD of head deviation (food side)",
    "frac_approach_of_SC": "Fraction of visits with approach to social",
    "interacting_speed_SS": "Interaction speed (social side)",
    "approach_speed_SS":   "Approach speed (social side)",
    "baseline_speed_SS":   "Baseline speed (social side)",
    "Time_SC":             "Time in social chamber (s)",
    "Time_FC":             "Time in food chamber (s)",
    "Time_Neutral":        "Time in neutral zone (s)",
    "RDI":                 "Social-preference index (RDI)",
    "TOTAL INVEST":        "Total investigation time (s)",
    "Total_Visits":        "Total chamber visits",
    "Total_Entries":       "Total chamber entries",
    "Switch_Probability_FC": "P(switch to social | at food)",
    "Switch_Probability_SC": "P(switch to food | at social)",
    "total_distance_m":    "Total distance walked (m)",
}


_UNMAPPED_WARNED: set[str] = set()


def _pretty(c: str) -> str:
    if c in FEATURE_LABELS:
        return FEATURE_LABELS[c]
    if c not in _UNMAPPED_WARNED:
        _UNMAPPED_WARNED.add(c)
        print(f"[label] no pretty name for '{c}' — add it to FEATURE_LABELS if it appears in a figure")
    return c


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def _clean_sheet(df):
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def _to_numeric_flag_dates(df, skip):
    out, n_bad = df.copy(), {}
    for c in df.columns:
        if c in skip:
            continue
        is_date = df[c].map(lambda v: isinstance(v, (dt.datetime, dt.date, pd.Timestamp)))
        if is_date.any():
            n_bad[c] = int(is_date.sum())
        out[c] = pd.to_numeric(df[c].where(~is_date), errors="coerce")
    return out, n_bad


def load_dataset(xlsx):
    xl = pd.ExcelFile(xlsx)
    raw = {i: _clean_sheet(xl.parse(xl.sheet_names[i - 1])) for i in SHEETS}
    report = {"dropped_corrupt": {}, "nan_corrupt": {}}

    beh = raw[1]
    beh["Video"] = beh["Video"].astype(str).str.strip()
    labels = beh.set_index("Video")["Treatment"].astype(str).str.strip()
    for i, key, tcol in [(4, "Video", "Treatment"), (5, "video", "Treatment"), (6, "id", "Unnamed: 4"),
                         (7, "video", "Treatment"), (8, "video", "Treatment")]:
        t = raw[i].assign(**{key: raw[i][key].astype(str).str.strip()}).set_index(key)[tcol]
        bad = (labels.reindex(t.index) != t.astype(str).str.strip()).sum()
        if bad:
            raise ValueError(f"Treatment labels in sheet {i} disagree with sheet 1 for {bad} mice")

    feats, family = [], {}

    def add(df, key, fam_of):
        df = df.copy()
        df[key] = df[key].astype(str).str.strip()
        df = df.set_index(key)
        df, n_bad = _to_numeric_flag_dates(df, skip=NEVER_FEATURES)
        keep = [c for c in df.columns if c not in NEVER_FEATURES and not df[c].isna().all()]
        df = df[keep]
        for c in keep:
            if n_bad.get(c, 0) / len(df) > CORRUPT_DROP_FRAC:
                report["dropped_corrupt"][c] = n_bad[c]
                df = df.drop(columns=c)
            elif n_bad.get(c, 0):
                report["nan_corrupt"][c] = n_bad[c]
        rename = {c: _pretty(c) for c in df.columns}
        for raw_c, pretty_c in rename.items():
            family[pretty_c] = fam_of(raw_c)
        df = df.rename(columns=rename)
        feats.append(df)

    def fam_beh(c):
        return ("Investigation time / preference"
                if c.startswith(("Time_", "RDI", "TOTAL INVEST")) else "Investigation bouts / entries")

    add(raw[1], "Video", fam_beh)
    add(raw[4], "Video", lambda c: "Chamber transitions")
    add(raw[5], "video", lambda c: "Head orientation")
    add(raw[6], "id", lambda c: "Locomotion")
    add(raw[7], "video", lambda c: "Approach / interaction speed")
    add(raw[8], "video", lambda c: "Approach completion")

    X = pd.concat(feats, axis=1, join="outer")
    assert not X.columns.duplicated().any(), "Duplicate feature names across sheets"
    X = X.reindex(labels.index)
    y = labels.loc[X.index]
    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X, y, {c: family[c] for c in X.columns}, report


def fix_features(X, family):
    X = X.copy()
    lbl_sc, lbl_fc = _pretty("FC_Neutral_SC"), _pretty("FC_Neutral_FC")
    lbl_sw = _pretty("Switch_Probability_FC")
    if {lbl_sc, lbl_fc, lbl_sw} <= set(X.columns):
        den = (X[lbl_sc] + X[lbl_fc]).replace(0, np.nan)
        X[lbl_sw] = X[lbl_sc] / den
        print(f"Recomputed {lbl_sw} = {lbl_sc} / ({lbl_sc} + {lbl_fc})")
    return X, family


def apply_exclusions(X, family, cols):
    pretty_cols = {_pretty(c) for c in cols}
    missing = sorted((set(cols) | pretty_cols) - set(X.columns))
    if missing:
        print("WARNING: --exclude names not found among the features:", missing)
    gone = [c for c in X.columns if c in pretty_cols]
    if gone:
        print("Excluded by request:", gone)
    X = X.drop(columns=gone)
    return X, {c: f for c, f in family.items() if c in X.columns}


# --------------------------------------------------------------------------- #
# Tasks, cross-validation, permutation test
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    name: str
    short: str
    mask: np.ndarray
    y: np.ndarray
    classes: list
    importance: bool
    color: str


def build_tasks(treat):
    housing, feeding = treat.str[:2].values, treat.str[-2:].values
    allm = np.ones(len(treat), bool)
    f, h = ["AL", "FD"], ["GH", "SH"]
    return [
        Task("Feeding: AL vs FD (all mice)", "Feeding (all)", allm, feeding, f, True, FEED_COLOR),
        Task("Housing: GH vs SH (all mice)", "Housing (all)", allm, housing, h, True, HOUS_COLOR),
        Task("Feeding within GH", "Feeding | GH", housing == "GH", feeding[housing == "GH"], f, False, FEED_COLOR),
        Task("Feeding within SH", "Feeding | SH", housing == "SH", feeding[housing == "SH"], f, False, FEED_COLOR),
        Task("Housing within AL", "Housing | AL", feeding == "AL", housing[feeding == "AL"], h, False, HOUS_COLOR),
        Task("Housing within FD", "Housing | FD", feeding == "FD", housing[feeding == "FD"], h, False, HOUS_COLOR),
        Task("4-class treatment", "4-class", allm, treat.values, CLASSES, False, MULTI_COLOR),
    ]


def _fold_job(Xv, yi, tr, te, groups, seed, n_perm, importance):
    imp = SimpleImputer(strategy="median").fit(Xv[tr])
    Xtr, Xte = imp.transform(Xv[tr]), imp.transform(Xv[te])
    rf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed}).fit(Xtr, yi[tr])
    proba = rf.predict_proba(Xte)
    if not importance:
        return te, proba, None, None, None
    base = balanced_accuracy_score(yi[te], proba.argmax(1))
    rng = np.random.default_rng(seed)

    def drop_for(cols):
        d = []
        for _ in range(n_perm):
            Xp = Xte.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            d.append(base - balanced_accuracy_score(yi[te], rf.predict(Xp)))
        return np.mean(d)

    return (te, proba, np.array([drop_for([j]) for j in range(Xv.shape[1])]),
            np.array([drop_for(g) for g in groups]), rf.feature_importances_)


def run_cv(X, family, task, n_repeats, imp_perm, n_jobs):
    Xsub = X.loc[task.mask]
    Xv = Xsub.values.astype(float)
    K = len(task.classes)
    yi = np.array([task.classes.index(v) for v in task.y])
    fam_names = list(dict.fromkeys(family[c] for c in X.columns))
    groups = [[j for j, c in enumerate(X.columns) if family[c] == f] for f in fam_names]
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                          random_state=SEED).split(Xv, yi))
    res = Parallel(n_jobs=n_jobs)(delayed(_fold_job)(Xv, yi, tr, te, groups, SEED + k, imp_perm,
                                                     task.importance) for k, (tr, te) in enumerate(splits))
    P = np.zeros((n_repeats, len(yi), K))
    for k, r in enumerate(res):
        P[k // N_SPLITS, r[0]] = r[1]
    pred = P.argmax(2)
    auc = [roc_auc_score(yi, P[r][:, 1]) if K == 2 else
           roc_auc_score(yi, P[r], multi_class="ovr", average="macro") for r in range(n_repeats)]
    out = dict(
        bacc=np.array([balanced_accuracy_score(yi, pred[r]) for r in range(n_repeats)]),
        auc=np.array(auc),
        cm=np.mean([confusion_matrix(yi, pred[r], labels=range(K)) for r in range(n_repeats)], axis=0),
        proba=pd.DataFrame(P.mean(0), index=Xsub.index, columns=task.classes), yi=yi)
    if task.importance:
        out["feat_imp"] = pd.DataFrame([r[2] for r in res], columns=X.columns)
        out["grp_imp"] = pd.DataFrame([r[3] for r in res], columns=fam_names)
        out["mdi"] = pd.DataFrame([r[4] for r in res], columns=X.columns)
    return out


def permutation_test(X, task, n_perm, observed, n_jobs):
    Xv = X.loc[task.mask].values.astype(float)
    yi = np.array([task.classes.index(v) for v in task.y])

    def one(p):
        ys = np.random.default_rng(1000 + p).permutation(yi)
        pred = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=p).split(Xv, ys):
            imp = SimpleImputer(strategy="median").fit(Xv[tr])
            rf = RandomForestClassifier(**{**RF_PARAMS, "n_estimators": 200, "random_state": p})
            pred[te] = rf.fit(imp.transform(Xv[tr]), ys[tr]).predict(imp.transform(Xv[te]))
        return balanced_accuracy_score(ys, pred)

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in range(n_perm)))
    return null, (1 + np.sum(null >= observed)) / (1 + n_perm)


def effect_direction(X, task, feats):
    Xs, (neg, pos) = X.loc[task.mask], task.classes
    return {c: (f"{pos}>{neg}" if Xs.loc[task.y == pos, c].median() > Xs.loc[task.y == neg, c].median()
                else f"{neg}>{pos}") for c in feats}


def top_features_both_factors(res_f, res_h, top_n=TOP_N_HEATMAP):
    fi = pd.DataFrame({"feeding": res_f["feat_imp"].mean(),
                       "housing": res_h["feat_imp"].mean()})
    keep = list(dict.fromkeys(list(fi["feeding"].nlargest(top_n).index)
                              + list(fi["housing"].nlargest(top_n).index)))
    out = fi.loc[keep].copy()
    out["best"] = out[["feeding", "housing"]].max(axis=1)
    out["driver"] = np.where(out["feeding"] >= out["housing"], "feeding", "housing")
    out = (out.assign(o=out["driver"].map({"feeding": 0, "housing": 1}))
              .sort_values(["o", "best"], ascending=[True, False])
              .drop(columns="o"))
    return out


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fmt_p(p, n_perm):
    return f"p ≤ {p:.3f}" if p <= 1 / (n_perm + 1) + 1e-12 else f"p = {p:.3f}"


def _title(ax, letter, text, fs=12):
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=fs, fontweight="bold")


def _add_legend(fig, text, fontsize=8.5, y=0.005, width=120):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    fig.supxlabel(wrapped, fontsize=fontsize, ha="left", x=0.01, y=y, va="bottom", color="#333333")


# Supplementary figures: enlarge and bold every text element (titles, tick labels, legends,
# annotations, colourbar labels) so they stay legible when the figure is reduced to page width.
SUPP_FONT_SCALE = 1.3          # multiply every font size by this factor
SUPP_BOLD = True


def _legible(fig, scale=None, bold=None):
    from matplotlib.text import Text
    scale = SUPP_FONT_SCALE if scale is None else scale
    bold = SUPP_BOLD if bold is None else bold
    fig.canvas.draw()                      # materialise automatic tick labels before restyling them
    for t in fig.findobj(Text):
        if not t.get_text():
            continue
        t.set_fontsize(t.get_fontsize() * scale)
        if bold:
            t.set_fontweight("bold")
    return fig


def panel_accuracy(ax, tasks, res, nulls, pvals, n_perm):
    for i, t in enumerate(tasks):
        y = len(tasks) - 1 - i
        lo, hi = np.percentile(nulls[t.name], [2.5, 97.5])
        ax.barh(y, hi - lo, left=lo, height=.55, color="#dddddd", zorder=1)
        b = res[t.name]["bacc"]
        ax.errorbar(b.mean(), y, xerr=b.std(), fmt="o", color=t.color, ms=9, mec="k", mew=.6,
                    capsize=3, zorder=3)
        ax.text(1.03, y, f"{_fmt_p(pvals[t.name], n_perm)}\nAUC {res[t.name]['auc'].mean():.2f}  "
                         f"(n={len(t.y)})", va="center", fontsize=8)
    ax.axvline(.5, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.short for t in tasks][::-1], fontsize=10)
    ax.set_xlim(.3, 1.3)
    ax.set_xticks(np.arange(.3, 1.01, .1))
    ax.spines["bottom"].set_bounds(.3, 1.0)
    ax.set_xlabel("Balanced accuracy")
    _clean(ax)


def panel_family(ax, res_f, res_h):
    gF, gH = res_f["grp_imp"], res_h["grp_imp"]
    order = gF.mean().add(gH.mean()).sort_values().index
    yy = np.arange(len(order))
    ax.barh(yy + .2, gF.mean()[order], .38, xerr=gF.sem()[order], color=FEED_COLOR,
            label="Feeding (AL vs FD)", error_kw=dict(lw=.8, capsize=2))
    ax.barh(yy - .2, gH.mean()[order], .38, xerr=gH.sem()[order], color=HOUS_COLOR,
            label="Housing (GH vs SH)", error_kw=dict(lw=.8, capsize=2))
    ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=9)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Drop in balanced accuracy (mean ± SEM)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _clean(ax)


def panel_heatmap(ax, X, treat, rep):
    cols = rep.index
    Z = (X[cols] - X[cols].mean()) / X[cols].std()
    H = Z.groupby(treat).mean().loc[CLASSES].T
    sns.heatmap(H, cmap="vlag", center=0, vmin=-1.5, vmax=1.5, ax=ax, linewidths=.5, linecolor="white",
                annot=True, fmt=".1f", annot_kws={"fontsize": 10},
                cbar_kws={"label": "Group mean (z-score)", "shrink": .8})
    ax.set_yticklabels(rep.index, fontsize=9.5, rotation=0)
    for lab, drv in zip(ax.get_yticklabels(), rep["driver"]):
        lab.set_color(FEED_COLOR if drv == "feeding" else HOUS_COLOR)
        lab.set_fontweight("bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)


def panel_confusion(ax, cm, classes, bacc, chance, letter, label):
    cmn = cm / cm.sum(1, keepdims=True)
    sns.heatmap(cmn, cmap="Blues", vmin=0, vmax=1, ax=ax, linewidths=1, linecolor="white",
                xticklabels=classes, yticklabels=classes, cbar=False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j + .5, i + .5, f"{cmn[i, j]:.2f}\n({cm[i, j]:.1f})", ha="center", va="center",
                    color="white" if cmn[i, j] > .55 else "black", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    _title(ax, letter, f"{label} — bal. acc {bacc:.2f} (chance {chance:.2f})")


def panel_top_features(ax, X, family, task, res, top_n=12):
    fi = res["feat_imp"]
    top = fi.mean().sort_values(ascending=False).head(top_n)
    sem = fi.sem()[top.index]
    dirs = effect_direction(X, task, top.index)
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, top.values, xerr=sem.values, color=[FAM_COLORS[family[c]] for c in top.index],
            edgecolor="k", linewidth=.3, error_kw=dict(lw=.8, capsize=2))
    xmax = (top.values + sem.values).max()
    for y_, c, v, e in zip(yy, top.index, top.values, sem.values):
        if v > 0:
            ax.text(v + e + xmax * .03, y_, dirs[c], va="center", fontsize=7.5, color=task.color,
                    fontweight="bold")
    ax.set_xlim(min(0, (top.values - sem.values).min() * 1.2), xmax * 1.3)
    ax.set_yticks(yy); ax.set_yticklabels(top.index, fontsize=8.5)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Drop in balanced accuracy when shuffled (mean ± SEM)")
    fams = [f for f in FAM_COLORS if f in {family[c] for c in top.index}]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=FAM_COLORS[f]) for f in fams], fams,
              fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False)
    _clean(ax)


# --------------------------------------------------------------------------- #
# Figures — main (three separate single-panel figures)
# --------------------------------------------------------------------------- #
def figure_main_A(bin_tasks, res, nulls, pvals, n_perm):
    """Panel A alone, sized for a single-column or half-page figure."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    panel_accuracy(ax, bin_tasks, res, nulls, pvals, n_perm)
    _title(ax, "A", "Decoder accuracy")
    _add_legend(fig,
        "Cross-validated balanced accuracy of the two pooled binary decoders (dot = mean, "
        "error bar = SD over CV repeats); grey band = 95% of shuffled-label runs; "
        "chance = 0.5. AUC and per-task n are shown on the right.",
        width=95, y=0.005)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def figure_main_B(res_f, res_h):
    """Panel B alone."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    panel_family(ax, res_f, res_h)
    _title(ax, "B", "Family importance")
    _add_legend(fig,
        "Drop in balanced accuracy when an entire metric family is shuffled, separately for "
        "the feeding (orange) and housing (green) decoders (mean ± SEM).",
        width=85, y=0.005)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def figure_main_C(X, treat, rep):
    """Panel C alone — sized to the number of rows so nothing overlaps."""
    n_rows = len(rep)
    fig, ax = plt.subplots(figsize=(5.0, 0.62 * n_rows + 1.8))
    panel_heatmap(ax, X, treat, rep)
    _title(ax, "C", "Direction of effect")
    _add_legend(fig,
        "Z-scored group means for the top three features of the feeding decoder and the top "
        "three of the housing decoder (union; up to six rows). The label colour indicates "
        "which decoder ranks the feature higher (orange = feeding, green = housing).",
        width=70, y=0.005)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def figure_main_combined(X, treat, family, bin_tasks, res, nulls, pvals, n_perm, rep):
    """Legacy single-page, three-panel-wide version (only produced if requested)."""
    fig = plt.figure(figsize=(19, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    panel_accuracy(ax, bin_tasks, res, nulls, pvals, n_perm)
    _title(ax, "A", "Decoder accuracy")

    ax = fig.add_subplot(gs[0, 1])
    panel_family(ax, res[bin_tasks[0].name], res[bin_tasks[1].name])
    _title(ax, "B", "Family importance")

    ax = fig.add_subplot(gs[0, 2])
    panel_heatmap(ax, X, treat, rep)
    _title(ax, "C", "Direction of effect")

    return fig


# --------------------------------------------------------------------------- #
# Supplementary figures
# --------------------------------------------------------------------------- #
def figure_supp1(X, family, bin_tasks, res, num):
    tF, tH = bin_tasks[0], bin_tasks[1]
    fig = plt.figure(figsize=(17, 13.5))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.95, height_ratios=[0.85, 1.15])
    for col, t, letter in ((0, tF, "A"), (1, tH, "B")):
        ax = fig.add_subplot(gs[0, col])
        panel_confusion(ax, res[t.name]["cm"], t.classes, res[t.name]["bacc"].mean(), .5, letter,
                        f"{t.short.split()[0]} decoder (all mice)")
    for col, t, letter in ((0, tF, "C"), (1, tH, "D")):
        ax = fig.add_subplot(gs[1, col])
        panel_top_features(ax, X, family, t, res[t.name])
        _title(ax, letter, f"Top 12 features — {t.short.split()[0].lower()}")
    fig.suptitle(f"Supplementary Figure {num}  |  Feeding and housing decoders: "
                 "error structure and single-feature importance",
                 fontsize=12, fontweight="bold", y=0.995)
    return _legible(fig)


def figure_supp2a_heatmap(task4, r4, num):
    """The 4-class decoder confusion-matrix heatmap on its own page."""
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    panel_confusion(ax, r4["cm"], CLASSES, r4["bacc"].mean(), .25, "A", "4-class decoder")
    fig.suptitle(f"Supplementary Figure {num}  |  Four-class decoder of the 2×2 treatment "
                 "(confusion matrix)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _legible(fig)


def figure_supp2(task4, r4, null4, p4, n_perm, num):
    """4-class decoder: permutation test (B) and per-mouse confidence (C).

    Panel A (the confusion-matrix heatmap) has been moved to its own page
    (see figure_supp2a_heatmap)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw=dict(wspace=0.55))

    ax = axes[0]
    ax.hist(null4, bins=25, color="#bbbbbb", edgecolor="white")
    ax.axvline(r4["bacc"].mean(), color="crimson", lw=2.5)
    ax.axvline(.25, color="k", ls=":", lw=1)
    ax.text(r4["bacc"].mean(), ax.get_ylim()[1] * .92, " observed", color="crimson", fontsize=9)
    ax.set_xlabel("Balanced accuracy"); ax.set_ylabel("Shuffled-label runs")
    _title(ax, "B", f"Permutation test — {_fmt_p(p4, n_perm)}, AUC {r4['auc'].mean():.2f}")
    _clean(ax)

    ax = axes[1]
    pt = np.array([r4["proba"].loc[m, c] for m, c in zip(r4["proba"].index, task4.y)])
    rng = np.random.default_rng(1)
    for k, c in enumerate(CLASSES):
        v = pt[task4.y == c]
        ax.scatter(np.full(len(v), k) + rng.uniform(-.18, .18, len(v)), v, color=GROUP_COLORS[c], s=42,
                   edgecolor="k", linewidth=.4, zorder=3)
        ax.hlines(np.median(v), k - .3, k + .3, color="k", lw=2, zorder=4)
    ax.axhline(.25, color="k", ls=":", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Out-of-fold P(true treatment)"); ax.set_ylim(-.02, 1.02)
    _title(ax, "C", "Per-mouse decoding confidence")
    _clean(ax)

    fig.suptitle(f"Supplementary Figure {num}  |  Four-class decoder of the 2×2 treatment "
                 "(permutation test and confidence)",
                 fontsize=12, fontweight="bold", y=1.03)
    return _legible(fig)


def figure_supp3(X, treat, family, num):
    Xi = SimpleImputer(strategy="median").fit_transform(X.values.astype(float))
    Z = StandardScaler().fit_transform(Xi)
    pca = PCA(n_components=min(10, Z.shape[1], Z.shape[0] - 1)).fit(Z)
    S = pca.transform(Z)
    ev = pca.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                             gridspec_kw=dict(width_ratios=[1.2, .7, 1.1], wspace=0.6))
    ax = axes[0]
    for c in CLASSES:
        m = (treat == c).values
        ax.scatter(S[m, 0], S[m, 1], color=GROUP_COLORS[c], s=48, edgecolor="k", linewidth=.4,
                   alpha=.85, label=f"{c} (n={m.sum()})")
        ax.scatter(S[m, 0].mean(), S[m, 1].mean(), color=GROUP_COLORS[c], s=260, marker="X",
                   edgecolor="k", linewidth=1.2, zorder=5)
    ax.axhline(0, color="#999999", lw=.6); ax.axvline(0, color="#999999", lw=.6)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% of variance)")
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "A", "PCA of all behavioural metrics")
    _clean(ax)
    ax = axes[1]
    ax.bar(range(1, len(ev) + 1), ev, color="#888888")
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), "k.-", lw=1)
    ax.set_xticks(range(1, len(ev) + 1))
    ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
    _title(ax, "B", "Variance explained")
    _clean(ax)
    ax = axes[2]
    L = pd.DataFrame(pca.components_[:2].T ** 2, index=X.columns, columns=["PC1", "PC2"]) * 100
    fam = L.groupby(pd.Series(family)).sum().sort_values("PC1")
    yy = np.arange(len(fam))
    ax.barh(yy + .2, fam["PC1"], .38, color="#4c4c4c", label="PC1")
    ax.barh(yy - .2, fam["PC2"], .38, color="#b0b0b0", label="PC2")
    ax.set_yticks(yy); ax.set_yticklabels(fam.index, fontsize=9)
    ax.set_xlabel("Share of component loading (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, "C", "Family contributions to PC1/PC2")
    _clean(ax)
    fig.suptitle(f"Supplementary Figure {num}  |  Unsupervised structure of the behavioural "
                 "metrics (exploratory)", fontsize=12, fontweight="bold", y=1.03)
    _legible(fig)
    return fig, pd.DataFrame(S[:, :2], index=X.index, columns=["PC1", "PC2"]).assign(treatment=treat)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="All_data_for_the_MS.xlsx")
    ap.add_argument("--outdir", default="decoder_figures")
    ap.add_argument("--n-perm", type=int, default=200, help="label permutations per task")
    ap.add_argument("--repeats", type=int, default=10, help="CV repeats")
    ap.add_argument("--imp-perm", type=int, default=5, help="shuffles per feature per fold (importance)")
    ap.add_argument("--supp-start", type=int, default=3, help="number of the first supplementary figure")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra feature columns to leave out (exact names)")
    ap.add_argument("--no-pca", action="store_true", help="skip the exploratory PCA page")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = ap.parse_args()
    out = Path(args.outdir); (out / "previews").mkdir(parents=True, exist_ok=True)

    X, treat, family, rep = load_dataset(args.xlsx)
    X, family = fix_features(X, family)
    X, family = apply_exclusions(X, family, EXCLUDE_COLUMNS | set(args.exclude))
    print(f"Loaded {X.shape[0]} mice x {X.shape[1]} features")
    print(treat.value_counts().reindex(CLASSES).to_string())
    if rep["dropped_corrupt"]:
        print("Dropped (cells mangled into dates by Excel):", list(rep["dropped_corrupt"]))
    pd.DataFrame({"family": pd.Series(family), "n_missing": X.isna().sum()}).to_csv(out / "features_used.csv")

    tasks = build_tasks(treat)
    res, nulls, pvals, rows = {}, {}, {}, []
    for t in tasks:
        print(f"\n== {t.name}  (n={len(t.y)}: " + ", ".join(f"{c}={int((t.y == c).sum())}" for c in t.classes) + ")")
        r = run_cv(X, family, t, args.repeats, args.imp_perm, args.n_jobs)
        null, p = permutation_test(X, t, args.n_perm, r["bacc"].mean(), args.n_jobs)
        res[t.name], nulls[t.name], pvals[t.name] = r, null, p
        print(f"   balanced acc {r['bacc'].mean():.3f} ± {r['bacc'].std():.3f} | AUC {r['auc'].mean():.3f} | "
              f"null mean {null.mean():.3f} | {_fmt_p(p, args.n_perm)}")
        rows.append(dict(task=t.name, n=len(t.y), balanced_acc=r["bacc"].mean(), balanced_acc_sd=r["bacc"].std(),
                         auc=r["auc"].mean(), null_mean=null.mean(), null_95th=np.percentile(null, 95),
                         p_perm=p, n_perm=args.n_perm))
    pd.DataFrame(rows).to_csv(out / "task_metrics.csv", index=False)

    bin_tasks, task4 = tasks[:6], tasks[6]
    tF, tH = bin_tasks[0], bin_tasks[1]
    for t, tag in ((tF, "feeding"), (tH, "housing")):
        r = res[t.name]
        pd.DataFrame({"family": pd.Series(family), "perm_importance_mean": r["feat_imp"].mean(),
                      "perm_importance_sem": r["feat_imp"].sem(), "mdi_mean": r["mdi"].mean(),
                      "higher_in": pd.Series(effect_direction(X, t, X.columns))}
                     ).sort_values("perm_importance_mean", ascending=False
                                   ).to_csv(out / f"feature_importance_{tag}.csv")
        r["grp_imp"].agg(["mean", "sem"]).T.to_csv(out / f"family_importance_{tag}.csv")
    pd.concat([res[t.name]["proba"].iloc[:, 1].rename(f"P({t.classes[1]})|{t.short}") for t in bin_tasks],
              axis=1).assign(treatment=treat).to_csv(out / "per_mouse_probabilities.csv")

    heat_feats = top_features_both_factors(res[tF.name], res[tH.name], top_n=TOP_N_HEATMAP)
    heat_feats.to_csv(out / "heatmap_features.csv")

    # ---- MAIN FIGURES: three separate single-panel pages ------------------------
    main_specs = [
        ("Figure_main_A_accuracy", figure_main_A(bin_tasks, res, nulls, pvals, args.n_perm)),
        ("Figure_main_B_family",   figure_main_B(res[tF.name], res[tH.name])),
        ("Figure_main_C_heatmap",  figure_main_C(X, treat, heat_feats)),
    ]
    for name, fig in main_specs:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / "previews" / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if ALSO_WRITE_COMBINED_MAIN:
        fig = figure_main_combined(X, treat, family, bin_tasks, res, nulls, pvals, args.n_perm, heat_feats)
        fig.savefig(out / "Figure_main_decoders.pdf", bbox_inches="tight")
        fig.savefig(out / "previews" / "Figure_main_decoders.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Supplementary PDF (multi-page) ----------------------------------------
    n0, pages = args.supp_start, []
    pages.append(("Supp1", figure_supp1(X, family, bin_tasks, res, n0)))
    # 4-class heatmap on its own page (panel A of the 4-class decoder)
    pages.append(("Supp2a", figure_supp2a_heatmap(task4, res[task4.name], n0 + 1)))
    # 4-class decoder: permutation test (B) + confidence (C)
    pages.append(("Supp2", figure_supp2(task4, res[task4.name], nulls[task4.name], pvals[task4.name],
                                        args.n_perm, n0 + 1)))
    if not args.no_pca:
        # PCA is the next page; account for the extra heatmap page so the number matches page position
        fig3, scores = figure_supp3(X, treat, family, n0 + 3)
        scores.to_csv(out / "pca_scores.csv")
        pages.append(("Supp3", fig3))
    with PdfPages(out / "Supplementary_decoders.pdf") as pdf:
        for name, f in pages:
            pdf.savefig(f, bbox_inches="tight")
            f.savefig(out / "previews" / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(f)

    print(f"\nSaved to {out.resolve()}:")
    print("  Figure_main_A_accuracy.pdf")
    print("  Figure_main_B_family.pdf")
    print("  Figure_main_C_heatmap.pdf")
    if ALSO_WRITE_COMBINED_MAIN:
        print("  Figure_main_decoders.pdf   (combined, optional)")
    print(f"  Supplementary_decoders.pdf ({len(pages)} pages)")
    print("  previews/, CSV tables")


if __name__ == "__main__":
    main()
