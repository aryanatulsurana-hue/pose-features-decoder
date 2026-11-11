#!/usr/bin/env python3
"""
SFPT random-forest decoders using ONLY sheet 1 of the Excel file (manually scored / BORIS)
==========================================================================================
Control analysis: how well do the classical, manually scored metrics (investigation times,
preference index, bouts ...) decode feeding state and housing, compared with the
pose-derived features?

Identical pipeline to the pose-based script so results are directly comparable:
  random forest, repeated stratified 5-fold CV, median imputation inside each fold,
  label-permutation test, same tasks:
    Feeding (all) | Housing (all) | Feeding|GH | Feeding|SH | Housing|AL | Housing|FD | 4-class

Labels (Treatment = "GH AL", "GH FD", "SH AL", "SH FD") are taken from sheet 1 if it has a
Treatment column, otherwise from sheet 2 (matched on the animal/video ID).

Outputs (in --outdir)
  Figure_boris_A_accuracy.pdf     A  decoder accuracy per task (+ pose-based overlay if --compare)
  Figure_boris_B_family.pdf       B  family importance (feeding vs housing decoder)
  Figure_boris_C_heatmap.pdf      C  direction of effect (z-scored group means, top features of
                                     the feeding and the housing decoder)
  Supplementary_boris_decoders.pdf  page 1: single-feature importance — feeding decoder
                                    page 2: single-feature importance — housing decoder
                                    page 3: 4-class decoder confusion-matrix heatmap
  boris_task_metrics.csv, boris_feature_importance_*.csv, boris_family_importance_*.csv,
  boris_heatmap_features.csv, boris_per_mouse_probabilities.csv, boris_features_used.csv
  previews/*.png

Usage
  python sfpt_boris_decoder.py --xlsx All_data_for_the_MS.xlsx
  python sfpt_boris_decoder.py --xlsx file.xlsx --compare decoder_figures/task_metrics.csv
  python sfpt_boris_decoder.py --xlsx file.xlsx --id-col Video --exclude "Some column"
  python sfpt_boris_decoder.py --xlsx file.xlsx --repeats 1 --n-perm 5 --imp-perm 1   # quick test

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

warnings.filterwarnings("ignore")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.linewidth": 0.8})

# --------------------------------------------------------------------------- #
# Configuration (same as the pose-based script)
# --------------------------------------------------------------------------- #
CLASSES = ["GH AL", "GH FD", "SH AL", "SH FD"]
FEED_COLOR, HOUS_COLOR, MULTI_COLOR = "#d95f02", "#1b9e77", "#555555"
FAM_COLORS = {
    "Investigation time / preference": "#4c78a8", "Investigation bouts / entries": "#72b7b2",
}
RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=1,
                 class_weight="balanced_subsample", random_state=0, n_jobs=1)
N_SPLITS, SEED = 5, 0
CORRUPT_DROP_FRAC = 0.05          # drop a column if >5% of cells were mangled into dates by Excel
TOP_N_HEATMAP = 3                 # top features per decoder shown in Panel C

ID_CANDIDATES = ["video", "id", "animal id", "animal", "mouse", "aid", "subject", "observation id"]
LABEL_COLS = {"video", "id", "animal id", "animal", "mouse", "aid", "subject", "observation id",
              "treatment", "housing", "food", "feeding", "group", "start frame", "radius",
              "pi_ignore","roiarea_sc (cm2)", "roiarea_fc (cm2)", "roiarea_ss (cm2)", "roiarea_fs (cm2), meanbout"}       # never used as features (labels / IDs)

# Same pretty names as the pose-based script (only the sheet-1 columns matter here).
# Anything not listed keeps its raw name.
FEATURE_LABELS: dict[str, str] = {
    "Time_SC":        "Time in social chamber (s)",
    "Time_FC":        "Time in food chamber (s)",
    "Time_Neutral":   "Time in neutral zone (s)",
    "RDI":            "Social-preference index (RDI)",
    "TOTAL INVEST":   "Total investigation time (s)",
    "Total_Visits":   "Total chamber visits",
    "Total_Entries":  "Total chamber entries",
    "frac_approach_of_SC": "Fraction of visits with approach to social",
}


def _pretty(c: str) -> str:
    return FEATURE_LABELS.get(c, c)


def family_of(c: str) -> str:
    """Same rule as the pose-based script for sheet-1 columns."""
    return ("Investigation time / preference"
            if str(c).startswith(("Time_", "RDI", "TOTAL INVEST")) else "Investigation bouts / entries")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _clean_sheet(df):
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def _find_col(df, candidates, override=None):
    if override:
        if override not in df.columns:
            raise ValueError(f"Column '{override}' not found. Available: {list(df.columns)}")
        return override
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


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


def load_boris(xlsx, sheet1=None, id_col=None, treat_col=None, exclude=()):
    xl = pd.ExcelFile(xlsx)
    name1 = sheet1 or xl.sheet_names[0]                       # sheet 1 (1-based) = index 0
    df = _clean_sheet(xl.parse(name1))
    print(f"Sheet 1 = '{name1}': {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    idc = _find_col(df, ID_CANDIDATES, id_col)
    if idc is None:
        raise ValueError("Could not identify the animal-ID column in sheet 1. "
                         f"Use --id-col. Columns: {list(df.columns)}")
    df[idc] = df[idc].astype(str).str.strip()

    # ---- labels -------------------------------------------------------------
    tc = _find_col(df, ["treatment"], treat_col)
    if tc is not None:
        labels = df.set_index(idc)[tc].astype(str).str.strip()
    else:
        print("No Treatment column in sheet 1 -> taking labels from sheet 2")
        s2 = _clean_sheet(xl.parse(xl.sheet_names[1]))
        id2 = _find_col(s2, ID_CANDIDATES)
        t2 = _find_col(s2, ["treatment"])
        if id2 is None or t2 is None:
            raise ValueError("Sheet 2 needs an ID and a Treatment column to supply labels.")
        s2[id2] = s2[id2].astype(str).str.strip()
        lab2 = s2.drop_duplicates(id2).set_index(id2)[t2].astype(str).str.strip()
        labels = lab2.reindex(df[idc].values)
        labels.index = df[idc].values
        if labels.isna().any():
            miss = labels[labels.isna()].index.tolist()
            raise ValueError(f"{len(miss)} IDs in sheet 1 not found in sheet 2, e.g. {miss[:5]}. "
                             f"Sheet 2 IDs look like {lab2.index[:5].tolist()}")

    # ---- features -----------------------------------------------------------
    skip = {c for c in df.columns if str(c).strip().lower() in LABEL_COLS or c == idc or c == tc}
    skip |= {c for c in df.columns if str(c).startswith("Unnamed")}
    skip |= set(exclude)
    num, n_bad = _to_numeric_flag_dates(df.set_index(idc), skip=skip | {idc})
    feats = [c for c in num.columns if c not in skip]
    X = num[feats]

    dropped = [c for c in feats if n_bad.get(c, 0) / len(X) > CORRUPT_DROP_FRAC]
    if dropped:
        print("Dropped (cells mangled into dates by Excel):", dropped)
    X = X.drop(columns=dropped)

    if X.index.duplicated().any():
        print("WARNING: duplicate IDs in sheet 1 (long format?) -> averaging rows per ID")
        X = X.groupby(level=0).mean()
        labels = labels[~labels.index.duplicated()]

    labels = labels.reindex(X.index)
    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    ok = labels.isin(CLASSES)
    if not ok.all():
        print("WARNING: dropping animals with unexpected labels:", labels[~ok].unique().tolist())
    X, labels = X[ok], labels[ok]
    print(f"Using {X.shape[0]} mice x {X.shape[1]} features: {list(X.columns)}")
    print(labels.value_counts().reindex(CLASSES).to_string())
    return X, labels


# --------------------------------------------------------------------------- #
# Tasks, CV, permutation test
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
        return te, proba, None, None
    base = balanced_accuracy_score(yi[te], proba.argmax(1))
    rng = np.random.default_rng(seed)

    def drop_for(cols):
        d = []
        for _ in range(n_perm):
            Xp = Xte.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]      # whole family shuffled jointly
            d.append(base - balanced_accuracy_score(yi[te], rf.predict(Xp)))
        return np.mean(d)

    return (te, proba, np.array([drop_for([j]) for j in range(Xv.shape[1])]),
            np.array([drop_for(g) for g in groups]))


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
        proba=pd.DataFrame(P.mean(0), index=Xsub.index, columns=task.classes))
    if task.importance:
        out["feat_imp"] = pd.DataFrame([r[2] for r in res], columns=X.columns)
        out["grp_imp"] = pd.DataFrame([r[3] for r in res], columns=fam_names)
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
# Identical to the pose-based script so the two supplements match visually.
SUPP_FONT_SCALE = 1.3
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


def panel_accuracy(ax, tasks, res, nulls, pvals, n_perm, pose_df=None):
    off = 0.14 if pose_df is not None else 0
    for i, t in enumerate(tasks):
        y = len(tasks) - 1 - i
        lo, hi = np.percentile(nulls[t.name], [2.5, 97.5])
        ax.barh(y, hi - lo, left=lo, height=.55, color="#dddddd", zorder=1)
        b = res[t.name]["bacc"]
        ax.errorbar(b.mean(), y + off, xerr=b.std(), fmt="o", color=t.color, ms=9, mec="k", mew=.6,
                    capsize=3, zorder=3, label="Manual (BORIS)" if i == 0 else None)
        if pose_df is not None and t.name in pose_df.index:
            pb = pose_df.loc[t.name]
            ax.errorbar(pb["balanced_acc"], y - off, xerr=pb["balanced_acc_sd"], fmt="s", color="white",
                        mec=t.color, mew=1.5, ms=8, capsize=3, zorder=3,
                        label="Pose-derived" if i == 0 else None)
        ax.text(1.03, y, f"{_fmt_p(pvals[t.name], n_perm)}\nAUC {res[t.name]['auc'].mean():.2f}  "
                         f"(n={len(t.y)})", va="center", fontsize=8)
    ax.axvline(.5, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.short for t in tasks][::-1], fontsize=10)
    ax.set_xlim(.3, 1.3)
    ax.set_xticks(np.arange(.3, 1.01, .1))
    ax.spines["bottom"].set_bounds(.3, 1.0)
    ax.set_xlabel("Balanced accuracy")
    if pose_df is not None:
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
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


# --------------------------------------------------------------------------- #
# Main figures — three separate single-panel figures (same layout as pose script)
# --------------------------------------------------------------------------- #
def figure_main_A(bin_tasks, res, nulls, pvals, n_perm, pose_df=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    panel_accuracy(ax, bin_tasks, res, nulls, pvals, n_perm, pose_df)
    _title(ax, "A", "Decoder accuracy — manually scored (BORIS) metrics")
    extra = " Squares = pose-derived decoder." if pose_df is not None else ""
    _add_legend(fig,
        "Cross-validated balanced accuracy of the two pooled binary decoders and the four "
        "subgroup decoders (dot = mean, error bar = SD over CV repeats); grey band = 95% of "
        "shuffled-label runs; chance = 0.5. AUC and per-task n are shown on the right." + extra,
        width=95, y=0.005)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def figure_main_B(res_f, res_h):
    n_fam = res_f["grp_imp"].shape[1]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.6 * n_fam + 2.4)))
    panel_family(ax, res_f, res_h)
    _title(ax, "B", "Family importance")
    _add_legend(fig,
        "Drop in balanced accuracy when an entire metric family is shuffled, separately for "
        "the feeding (orange) and housing (green) decoders (mean ± SEM).",
        width=85, y=0.005)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def figure_main_C(X, treat, rep):
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


# --------------------------------------------------------------------------- #
# Supplementary figures
# --------------------------------------------------------------------------- #
def figure_supp_importance(X, family, task, res, letter, num, driver_label):
    """Single-feature importance for one pooled binary decoder, on its own page.
    Sized like the pose script's supplementary panels; run through _legible() for
    consistent enlarged/bold text."""
    fi = res[task.name]["feat_imp"]
    m = fi.mean().sort_values(ascending=True)
    s = fi.sem()[m.index]
    dirs = effect_direction(X, task, m.index)
    yy = np.arange(len(m))

    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.4 * X.shape[1] + 1.8)))
    ax.barh(yy, m.values, xerr=s.values, color=[FAM_COLORS[family[c]] for c in m.index],
            edgecolor="k", linewidth=.3, error_kw=dict(lw=.8, capsize=2))
    xr = max((m + s).max(), 1e-3)
    for y_, c, v, e in zip(yy, m.index, m.values, s.values):
        if v > 0:
            ax.text(v + e + xr * .03, y_, dirs[c], va="center", fontsize=7.5, color=task.color,
                    fontweight="bold")
    ax.set_xlim(min(0, (m - s).min() * 1.2), xr * 1.35)
    ax.set_yticks(yy); ax.set_yticklabels(m.index, fontsize=8.5)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Drop in balanced accuracy when shuffled (mean ± SEM)")
    _title(ax, letter, f"{driver_label} decoder — single-feature importance")
    fams = [f for f in FAM_COLORS if f in {family[c] for c in m.index}]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=FAM_COLORS[f]) for f in fams], fams,
              fontsize=7, loc="lower right", frameon=False)
    _clean(ax)
    fig.suptitle(f"Supplementary Figure {num}  |  Manually scored (BORIS) metrics — "
                 f"{driver_label.lower()} decoder feature importance",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _legible(fig)


def figure_supp_fourclass(r4, num):
    """4-class decoder confusion-matrix heatmap, on its own page.
    Matches the 4-class panel of the pose-based supplement (single heatmap with
    a figure-level title), and goes through _legible() so all text is enlarged/bold."""
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    panel_confusion(ax, r4["cm"], CLASSES, r4["bacc"].mean(), .25, "A", "4-class decoder")
    fig.suptitle(f"Supplementary Figure {num}  |  Four-class decoder of the 2×2 treatment "
                 "(manual / BORIS metrics — confusion matrix)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _legible(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="All_data_for_the_MS.xlsx")
    ap.add_argument("--outdir", default="decoder_boris_only")
    ap.add_argument("--sheet1", default=None, help="name of sheet 1 (default: 1st tab)")
    ap.add_argument("--id-col", default=None, help="animal/video ID column in sheet 1")
    ap.add_argument("--treat-col", default=None, help="Treatment column in sheet 1 (optional)")
    ap.add_argument("--exclude", nargs="*", default=[], help="feature columns to leave out (exact names)")
    ap.add_argument("--n-perm", type=int, default=200, help="label permutations per task")
    ap.add_argument("--repeats", type=int, default=10, help="CV repeats")
    ap.add_argument("--imp-perm", type=int, default=5, help="shuffles per feature per fold")
    ap.add_argument("--compare", default=None,
                    help="task_metrics.csv from the pose-based script, to overlay its accuracies in Panel A")
    ap.add_argument("--supp-start", type=int, default=1,
                    help="number of the first supplementary figure in the BORIS supplement")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    out = Path(args.outdir); (out / "previews").mkdir(parents=True, exist_ok=True)

    X, treat = load_boris(args.xlsx, args.sheet1, args.id_col, args.treat_col, args.exclude)

    # families from the raw names, then rename columns to the pretty labels used in the figures
    family_raw = {c: family_of(c) for c in X.columns}
    X = X.rename(columns={c: _pretty(c) for c in X.columns})
    family = {_pretty(c): f for c, f in family_raw.items()}
    pd.DataFrame({"label": X.columns, "raw_name": list(family_raw), "family": [family[c] for c in X.columns],
                  "n_missing": X.isna().sum().values}).to_csv(out / "boris_features_used.csv", index=False)

    tasks = build_tasks(treat)
    res, nulls, pvals, rows = {}, {}, {}, []
    for t in tasks:
        print(f"\n== {t.name} (n={len(t.y)}: " + ", ".join(f"{c}={int((t.y == c).sum())}" for c in t.classes) + ")")
        r = run_cv(X, family, t, args.repeats, args.imp_perm, args.n_jobs)
        null, p = permutation_test(X, t, args.n_perm, r["bacc"].mean(), args.n_jobs)
        res[t.name], nulls[t.name], pvals[t.name] = r, null, p
        print(f"   balanced acc {r['bacc'].mean():.3f} ± {r['bacc'].std():.3f} | AUC {r['auc'].mean():.3f} | "
              f"null mean {null.mean():.3f} | {_fmt_p(p, args.n_perm)}")
        rows.append(dict(task=t.name, n=len(t.y), balanced_acc=r["bacc"].mean(), balanced_acc_sd=r["bacc"].std(),
                         auc=r["auc"].mean(), null_mean=null.mean(), null_95th=np.percentile(null, 95),
                         p_perm=p, n_perm=args.n_perm))
    pd.DataFrame(rows).to_csv(out / "boris_task_metrics.csv", index=False)

    bin_tasks, task4 = tasks[:6], tasks[6]
    tF, tH = bin_tasks[0], bin_tasks[1]
    for t, tag in ((tF, "feeding"), (tH, "housing")):
        r = res[t.name]
        pd.DataFrame({"family": pd.Series(family), "perm_importance_mean": r["feat_imp"].mean(),
                      "perm_importance_sem": r["feat_imp"].sem(),
                      "higher_in": pd.Series(effect_direction(X, t, X.columns))}
                     ).sort_values("perm_importance_mean", ascending=False
                                   ).to_csv(out / f"boris_feature_importance_{tag}.csv")
        r["grp_imp"].agg(["mean", "sem"]).T.to_csv(out / f"boris_family_importance_{tag}.csv")
    pd.concat([res[t.name]["proba"].iloc[:, 1].rename(f"P({t.classes[1]})|{t.short}") for t in bin_tasks],
              axis=1).assign(treatment=treat).to_csv(out / "boris_per_mouse_probabilities.csv")

    pose_df = None
    if args.compare:
        pose_df = pd.read_csv(args.compare).set_index("task")
        print("\nPose-based vs manual balanced accuracy:")
        cmp = pd.DataFrame({"manual": pd.DataFrame(rows).set_index("task")["balanced_acc"],
                            "pose": pose_df["balanced_acc"]})
        print(cmp.round(3).to_string())
        cmp.to_csv(out / "boris_vs_pose_balanced_accuracy.csv")

    heat_feats = top_features_both_factors(res[tF.name], res[tH.name], top_n=TOP_N_HEATMAP)
    heat_feats.to_csv(out / "boris_heatmap_features.csv")

    # ---- MAIN FIGURES: three separate single-panel pages (same as the pose decoder) ----
    main_specs = [
        ("Figure_boris_A_accuracy", figure_main_A(bin_tasks, res, nulls, pvals, args.n_perm, pose_df)),
        ("Figure_boris_B_family", figure_main_B(res[tF.name], res[tH.name])),
        ("Figure_boris_C_heatmap", figure_main_C(X, treat, heat_feats)),
    ]
    for name, fig in main_specs:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / "previews" / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- Supplementary PDF: single-feature importance (feeding + housing) + 4-class heatmap ----
    n0, pages = args.supp_start, []
    pages.append(("Supp_feed",
                  figure_supp_importance(X, family, tF, res, "A", n0, "Feeding")))
    pages.append(("Supp_hous",
                  figure_supp_importance(X, family, tH, res, "A", n0 + 1, "Housing")))
    pages.append(("Supp_fourclass",
                  figure_supp_fourclass(res[task4.name], n0 + 2)))
    with PdfPages(out / "Supplementary_boris_decoders.pdf") as pdf:
        for name, f in pages:
            pdf.savefig(f, bbox_inches="tight")
            f.savefig(out / "previews" / f"boris_{name}.png", dpi=170, bbox_inches="tight")
            plt.close(f)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
