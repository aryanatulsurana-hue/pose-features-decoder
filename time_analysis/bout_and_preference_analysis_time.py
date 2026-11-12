#!/usr/bin/env python3
"""
SFPT motivation analysis
========================
Turns the bout tables produced by ``roi_bout_analysis.py`` into per-animal
motivation metrics, compares the four groups (SHAL, SHFD, GHAL, GHFD) with a
2x2 (Housing x Food state) design, and analyses how behaviour changes across
the session in time bins.

Unit of analysis
    One KEY (e.g. ``MVI_0006_roi1``) = one animal = one 600 s window.
    Nothing is averaged or merged across keys.

Input (per key, all in one folder)
    <key>_bouts_detailed.csv                  states SS, FS, SC, FC, Neutral
    <key>_bouts_collapsed_SC_FC_Neutral.csv   states SC, FC, Neutral
Group file (csv/xlsx)
    columns: key, housing (GH/SH), food_state (AL/FD)

Per-animal metrics
    p_choose_fc, n_choice_points             choice split at Neutral exits
    history_phi (+ chi2, p)                  does the previous chamber predict the next?
    ss_/fs_ n_bouts, mean_duration_s,        bout counts and durations, first/second-half
        first/second-half mean, slope        means, within-session slope
    pref_bin_slope, ss_/fs_logdur_bin_slope  change across time bins

Time-binned analysis (bin length is asked for at start-up; default 60 s)
    Preference index = (SS time - FS time) / (SS time + FS time)  [stimulus zones only]
    SS and FS mean bout duration per bin -> plots (points; points + GEE fit),
    mixed models and GEE (housing x food state x time, animals as clusters).
    Every bin is kept; a trailing remainder is added to the last bin. A bin where a
    value cannot be computed (no zone time / no bout started) is NaN and is simply
    not plotted.

Statistics
    2x2 ANOVA (Type II) with assumption checks, HC3-robust and permutation p-values,
    BH-FDR q-values, outlier sensitivity, four planned comparisons (Welch t, Holm),
    mixed models and GEE for the time courses.

Figure options (see the TUNABLE PARAMETERS block)
    Preference time-course y-axis fixed to -1..+1; SS and FS bout-duration figures
    share one y-axis; GEE line runs from the first to the last bin centre.

Usage
    bout_and_preference_analysis_time.py
    (prompts for the bout folder, output folder, group file and bin length)

Requires: numpy, pandas, scipy, matplotlib, statsmodels (and openpyxl for .xlsx)
"""

import glob
import json
import os
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.formula.api import ols
    from statsmodels.genmod import families as sm_families
    from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable
    from statsmodels.genmod.generalized_estimating_equations import GEE
except ImportError:
    sys.exit("This script needs statsmodels:  pip install statsmodels")

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
# --- time bins ---------------------------------------------------------------
BIN_SECONDS = 60  # default bin width in seconds; main() asks for the value at run time
# A window up to this many seconds SHORT of a whole number of bins still gets that last bin
# (e.g. 599.7 s -> 10 bins of 60 s); a LONGER window has its remainder added to the last bin
# (e.g. 605 s -> 9 x 60 s + one 65 s bin).
BIN_END_TOLERANCE_S = 1.0
STIM_STATES = ("SS", "FS")  # stimulus-zone states in the detailed table
CHAMBER_STATES = ("SC", "FC")  # chamber states in the collapsed table

# --- groups ------------------------------------------------------------------
HOUSING_LEVELS = ("GH", "SH")  # contrasts are reported as first minus second (GH - SH)
FOOD_LEVELS = ("FD", "AL")  # (FD - AL)
GROUP_ORDER = ("SHAL", "SHFD", "GHAL", "GHFD")

# --- statistics --------------------------------------------------------------
ALPHA = 0.05
MIN_PER_CELL = 5  # animals needed in EACH of the 4 groups to test a metric
N_PERMUTATIONS = 5000  # permutations for the permutation p-values
PERM_SEED = 12345  # fixed so results are reproducible
SD_RATIO_LIMIT = 2.0  # flag unequal spread if largest/smallest group SD exceeds this
OUTLIER_Z = 3.0  # |externally studentized residual| above this = outlier (flag + sensitivity test)
PRIMARY_METRICS = ["p_choose_fc", "history_phi", "ss_mean_duration_s", "fs_mean_duration_s"]
LOG_METRIC_PATTERNS = (r"_mean_duration",)  # analysed on the log scale
LOG1P_METRIC_PATTERNS = (r"_n_bouts$", r"^n_choice_points$")  # counts: log(1 + x)
ANOVA_EXCLUDE_PATTERNS = (r"_p_value$", r"_chi2$", r"^window_s$", r"^n_bins$")
WEIGHTED_METRICS = {"history_phi": "n_choice_points"}  # extra weighted (WLS) sensitivity test

GROUP_COLORS = {"GHAL": "#D55E00", "GHFD": "#E69F00", "SHAL": "#155D86", "SHFD": "#39CECE"}

# --- time-course figures -----------------------------------------------------
PLOT_GROUPS = (("GHAL", "GH AL"), ("GHFD", "GH FD"), ("SHAL", "SH AL"), ("SHFD", "SH FD"))  # legend order
MIN_ANIMALS_PER_POINT = 1  # a group mean is plotted in a bin if at least this many animals have a value
GEE_MIN_ANIMALS = 5  # animals (clusters) needed in EACH group to fit the GEE model
# response: (family, working correlation); family is "gaussian" or "gamma_log".
# Correlation can be "exchangeable" or "ar1" (a QIC comparison of both is saved).
GEE_SPECS = {
    "preference_index": ("gaussian", "exchangeable"),
    "ss_mean_duration_s": ("gamma_log", "exchangeable"),
    "fs_mean_duration_s": ("gamma_log", "exchangeable"),
}
GEE_COV_TYPE = "bias_reduced"  # small-sample (Mancl-DeRouen) sandwich covariance; falls back to "robust"
PREF_MIN_ZONE_S = 5.0  # sensitivity analysis for preference: drop bins with less SS+FS time than this
GEE_BOUT_LEVEL = True  # sensitivity analysis for durations: every bout is one observation
POINTS_ONLY_ERRORBARS = True  # points-only figure: draw mean +/- SEM bars
POINTS_ONLY_CONNECT = False  # points-only figure: join the points with thin lines
# True  = GEE line runs from the first to the last bin centre (where the model has data);
# False = full session width 0..end (extrapolates half a bin at each end).
FIT_SPAN_TO_BIN_CENTERS = True
SHARED_DURATION_YAXIS = True  # SS and FS bout-duration figures (points and GEE) share one y-axis range
# None = automatic (highest value drawn in any duration figure, rounded up to a multiple
# of 5); or give a number, e.g. 30.
DURATION_YMAX = None
FIT_FIGURE_ERRORBARS = False  # GEE figure: draw SEM bars on the points (the reference figure has none)
DODGE_FRACTION = 0.06  # horizontal offset between groups, as a fraction of the bin width


# =============================================================================
# Loading
# =============================================================================
def find_keys(bout_dir):
    """Return the keys that have both a detailed and a collapsed bout table in ``bout_dir``."""
    keys = []
    for p in sorted(glob.glob(os.path.join(bout_dir, "*_bouts_detailed.csv"))):
        key = os.path.basename(p)[: -len("_bouts_detailed.csv")]
        if os.path.exists(os.path.join(bout_dir, f"{key}_bouts_collapsed_SC_FC_Neutral.csv")):
            keys.append(key)
        else:
            print(f"  ! Skipping {key}: no matching collapsed bout file")
    return keys


def load_bouts(bout_dir, key):
    """Read the detailed and collapsed bout tables of one key."""
    det = pd.read_csv(os.path.join(bout_dir, f"{key}_bouts_detailed.csv"))
    col = pd.read_csv(os.path.join(bout_dir, f"{key}_bouts_collapsed_SC_FC_Neutral.csv"))
    return det, col


def load_metadata(path):
    """Group file: one row per key with columns key, housing, food_state."""
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["key", "housing", "food_state"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Group file is missing column(s) {missing}; found {list(df.columns)}")
    out = df.rename(columns={cols[c]: c for c in need})[need].copy()
    out["key"] = out["key"].astype(str).str.strip()
    out["housing"] = out["housing"].astype(str).str.strip().str.upper()
    out["food_state"] = out["food_state"].astype(str).str.strip().str.upper()
    dup = out.loc[out["key"].duplicated(keep=False), "key"].unique()
    if len(dup):
        raise ValueError(f"Duplicate key(s) in group file (each key must appear once): {list(dup)}")
    bad_h = set(out["housing"]) - set(HOUSING_LEVELS)
    bad_f = set(out["food_state"]) - set(FOOD_LEVELS)
    if bad_h or bad_f:
        raise ValueError(
            f"Unexpected labels: housing {sorted(bad_h)}, food_state {sorted(bad_f)}. "
            f"Expected housing in {HOUSING_LEVELS} and food_state in {FOOD_LEVELS} "
            f"(edit HOUSING_LEVELS / FOOD_LEVELS at the top of the script)."
        )
    out["group"] = out["housing"] + out["food_state"]
    return out


# =============================================================================
# Per-animal metrics
# =============================================================================
def choice_point_table(collapsed):
    """Every Neutral bout with a chamber on both sides: where the animal came from and went to."""
    neu = collapsed[collapsed["roi"] == "Neutral"]
    neu = neu[neu["from_roi"].isin(CHAMBER_STATES) & neu["to_roi"].isin(CHAMBER_STATES)]
    return neu[["start_time_s", "end_time_s", "from_roi", "to_roi"]].reset_index(drop=True)


def choice_metrics(cp):
    """p_choose_fc and n_choice_points."""
    n = len(cp)
    return {"n_choice_points": n, "p_choose_fc": (cp["to_roi"] == "FC").mean() if n else np.nan}


def history_metrics(cp):
    """
    Does the previous chamber predict the next one?
    2x2 table of previous chamber (rows) x next chamber (columns):
        a = SC->SC   b = SC->FC   c = FC->SC   d = FC->FC     (a, d are 'stays')
    history_phi  = (ad - bc) / sqrt(r1 r2 c1 c2)   signed, no continuity correction
                   > 0 sticks to the same chamber, < 0 alternates, 0 no relation
    history_chi2 / history_p_value: chi-square test of independence (Yates-corrected,
                   descriptive per animal; depends on the number of events).
    """
    out = {"history_phi": np.nan, "history_chi2": np.nan, "history_p_value": np.nan}
    if len(cp) == 0:
        return out
    a = int(((cp.from_roi == "SC") & (cp.to_roi == "SC")).sum())
    b = int(((cp.from_roi == "SC") & (cp.to_roi == "FC")).sum())
    c = int(((cp.from_roi == "FC") & (cp.to_roi == "SC")).sum())
    d = int(((cp.from_roi == "FC") & (cp.to_roi == "FC")).sum())
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if min(r1, r2, c1, c2) == 0:
        return out
    out["history_phi"] = (a * d - b * c) / np.sqrt(r1 * r2 * c1 * c2)
    chi2, p, _, _ = stats.chi2_contingency(np.array([[a, b], [c, d]]))
    out["history_chi2"], out["history_p_value"] = chi2, p
    return out


def engagement_metrics(detailed, modality):
    """Bout-duration summary for one stimulus zone (SS or FS)."""
    pre = modality.lower()
    sub = detailed[detailed["roi"] == modality].sort_values("start_time_s")
    out = {
        f"{pre}_n_bouts": len(sub),
        f"{pre}_mean_duration_s": sub["duration_s"].mean() if len(sub) else np.nan,
        f"{pre}_slope_s_per_s": np.nan,
        f"{pre}_slope_p_value": np.nan,
        f"{pre}_mean_duration_first_half_s": np.nan,
        f"{pre}_mean_duration_second_half_s": np.nan,
    }
    if len(sub) >= 3 and sub["start_time_s"].nunique() > 1:
        lr = stats.linregress(sub["start_time_s"], sub["duration_s"])
        out[f"{pre}_slope_s_per_s"], out[f"{pre}_slope_p_value"] = lr.slope, lr.pvalue
    if len(sub):
        mid = sub["start_time_s"].median()  # halves = equal numbers of bouts
        out[f"{pre}_mean_duration_first_half_s"] = sub[sub.start_time_s <= mid]["duration_s"].mean()
        out[f"{pre}_mean_duration_second_half_s"] = sub[sub.start_time_s > mid]["duration_s"].mean()
    return out


def bin_edges(t0, t_end):
    """
    Bin edges (absolute seconds) for one animal: bins of BIN_SECONDS starting at t0, and the last
    edge is always t_end. So a trailing remainder is ADDED TO THE LAST BIN (nothing is dropped and
    no tiny stub bin appears). A window that is shorter than a whole number of bins by less than
    BIN_END_TOLERANCE_S still gets that last bin.
    """
    n_bins = max(1, int(np.floor((t_end - t0 + BIN_END_TOLERANCE_S) / BIN_SECONDS)))
    edges = t0 + BIN_SECONDS * np.arange(n_bins + 1, dtype=float)
    edges[-1] = t_end
    return edges


def binned_table(collapsed, detailed):
    """
    One row per time bin (bin 1..n, measured from the start of the analysis window). EVERY bin is kept.
      ss_time_s / fs_time_s : time spent in the social / food STIMULUS zones inside the bin
                              (from the detailed table)
      preference_index      : (SS - FS) / (SS + FS)   (+1 = only social zone, -1 = only food zone);
                              NaN if the animal spent no time in either zone in that bin
      sc/fc/neutral_time_s  : chamber / neutral time from the collapsed table (descriptive only)
      coverage              : fraction of the bin covered by the collapsed table (descriptive only)
      ss_/fs_ mean duration : mean duration of the bouts that START inside the bin
                              (NaN if no bout started in the bin)
    """
    t0 = float(collapsed["start_time_s"].min())
    t_end = float(collapsed["end_time_s"].max())
    edges = bin_edges(t0, t_end)
    n_bins = len(edges) - 1
    cst = collapsed["start_time_s"].to_numpy()
    cen = collapsed["end_time_s"].to_numpy()
    croi = collapsed["roi"].to_numpy()
    dst = detailed["start_time_s"].to_numpy()
    den = detailed["end_time_s"].to_numpy()
    droi = detailed["roi"].to_numpy()
    rows = []
    for k in range(n_bins):
        b0, b1 = edges[k], edges[k + 1]
        width = b1 - b0
        last = k == n_bins - 1
        ov_c = np.clip(np.minimum(cen, b1) - np.maximum(cst, b0), 0, None)
        occ = {s: float(ov_c[croi == s].sum()) for s in ("SC", "FC", "Neutral")}
        ov_d = np.clip(np.minimum(den, b1) - np.maximum(dst, b0), 0, None)
        stim = {s: float(ov_d[droi == s].sum()) for s in STIM_STATES}
        denom = stim["SS"] + stim["FS"]
        row = {
            "bin": k + 1,
            "bin_start_rel_s": b0 - t0,
            "bin_end_rel_s": b1 - t0,
            "bin_width_s": width,
            "bin_center_rel_s": (b0 + b1) / 2.0 - t0,
            "coverage": min(1.0, sum(occ.values()) / width) if width > 0 else np.nan,
            "ss_time_s": stim["SS"],
            "fs_time_s": stim["FS"],
            "sc_time_s": occ["SC"],
            "fc_time_s": occ["FC"],
            "neutral_time_s": occ["Neutral"],
            "preference_index": (stim["SS"] - stim["FS"]) / denom if denom > 0 else np.nan,
        }
        in_bin = (detailed["start_time_s"] >= b0) & (
            (detailed["start_time_s"] < b1) | (last & (detailed["start_time_s"] <= b1))
        )
        for m in STIM_STATES:
            sel = detailed[(detailed["roi"] == m) & in_bin]
            row[f"{m.lower()}_n_bouts"] = len(sel)
            row[f"{m.lower()}_mean_duration_s"] = sel["duration_s"].mean() if len(sel) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bin_slope(binned, col, log=False):
    """Slope of a binned measure against bin number (needs >= 3 valid bins)."""
    sub = binned[["bin", col]].dropna()
    if log:
        sub = sub[sub[col] > 0]
    if len(sub) < 3:
        return np.nan
    y = np.log(sub[col]) if log else sub[col]
    return float(stats.linregress(sub["bin"], y).slope)


def analyse_key(bout_dir, key, per_key_dir, make_plots=True):
    """Compute all per-animal metrics and the binned table for one key; optionally save its plots."""
    det, col = load_bouts(bout_dir, key)
    cp = choice_point_table(col)
    binned = binned_table(col, det)
    m = {"key": key, "window_s": float(col["end_time_s"].max() - col["start_time_s"].min())}
    m.update(choice_metrics(cp))
    m.update(history_metrics(cp))
    for mod in STIM_STATES:
        m.update(engagement_metrics(det, mod))
    ss_t, fs_t = float(binned["ss_time_s"].sum()), float(binned["fs_time_s"].sum())
    m["pref_index_overall"] = (ss_t - fs_t) / (ss_t + fs_t) if (ss_t + fs_t) > 0 else np.nan
    m["n_bins"] = len(binned)
    m["pref_bin_slope"] = bin_slope(binned, "preference_index")
    m["ss_logdur_bin_slope"] = bin_slope(binned, "ss_mean_duration_s", log=True)
    m["fs_logdur_bin_slope"] = bin_slope(binned, "fs_mean_duration_s", log=True)
    cp.to_csv(os.path.join(per_key_dir, f"{key}_choice_points.csv"), index=False)
    if make_plots:
        plot_key(key, binned, det, float(col["start_time_s"].min()), per_key_dir)
    return m, binned.assign(key=key)


def plot_key(key, binned, det, t0, out_dir):
    """Save a two-panel summary figure (preference by bin, bout durations over time) for one key."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    if not binned.empty:
        ax.plot(binned["bin_center_rel_s"], binned["preference_index"], marker="o", color="#1f3864")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time from start of window (s)")
    ax.set_ylabel("Preference index (SS vs FS)\n(+1 social, -1 food)")
    ax.set_title(f"{key}\nPreference by time bin")
    ax = axes[1]
    for mod, colr in (("SS", "#6baed6"), ("FS", "#fd8d3c")):
        sub = det[det["roi"] == mod]
        if len(sub):
            ax.scatter(sub["start_time_s"] - t0, sub["duration_s"], s=14, color=colr, label=mod)
            if len(sub) >= 3 and sub["start_time_s"].nunique() > 1:
                mm, bb = np.polyfit(sub["start_time_s"] - t0, sub["duration_s"], 1)
                xs = np.array([0, (sub["start_time_s"] - t0).max()])
                ax.plot(xs, mm * xs + bb, color=colr, lw=1.5)
    ax.set_xlabel("Bout start time from start of window (s)")
    ax.set_ylabel("Bout duration (s)")
    ax.set_title("Stimulus-zone bout durations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{key}_summary.png"), dpi=140)
    plt.close(fig)


# =============================================================================
# Statistics helpers
# =============================================================================
def scale_of(metric):
    """Return the analysis scale of a metric: ``log``, ``log1p`` or ``raw``."""
    if any(re.search(p, metric) for p in LOG_METRIC_PATTERNS):
        return "log"
    if any(re.search(p, metric) for p in LOG1P_METRIC_PATTERNS):
        return "log1p"
    return "raw"


def transform(x, scale):
    """Apply the log / log1p / identity transform named by ``scale``."""
    x = np.asarray(x, dtype=float)
    return np.log(x) if scale == "log" else np.log1p(x) if scale == "log1p" else x


def bh_fdr(p):
    """Benjamini-Hochberg q-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / (np.arange(pv.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def holm(p):
    """Holm step-down adjusted p-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    if ok.size == 0:
        return adj
    order = ok[np.argsort(p[ok])]
    m = ok.size
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def _hat(X):
    """Hat (projection) matrix of the design matrix ``X``."""
    q, _ = np.linalg.qr(X)
    return q @ q.T


def permutation_pvalues(h, f, y, n_perm=N_PERMUTATIONS, seed=PERM_SEED):
    """
    Freedman-Lane permutation p-values for the three Type II terms of a 2x2 design.
    h, f are +/-1 coded factors. For each term: fit the model WITHOUT the term, permute
    its residuals, add them back to its fitted values, and recompute the F statistic.
    The p-value is the share of permuted F values at least as large as the observed F.
    Returns ({term: (F_obs, p)}).
    """
    n = len(y)
    one = np.ones(n)
    X_hf = np.column_stack([one, h, f, h * f])
    H4 = _hat(X_hf)
    df_den = n - 4
    specs = {
        "housing": (np.column_stack([one, f]), np.column_stack([one, h, f])),
        "food_state": (np.column_stack([one, h]), np.column_stack([one, h, f])),
        "interaction": (np.column_stack([one, h, f]), X_hf),
    }
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n, n_perm)), axis=0)
    out = {}
    for term, (X_red, X_ft) in specs.items():
        Hr, Hf = _hat(X_red), _hat(X_ft)

        def fstat(Y):
            r_red = ((Y - Hr @ Y) ** 2).sum(0)
            r_ft = ((Y - Hf @ Y) ** 2).sum(0)
            r_4 = ((Y - H4 @ Y) ** 2).sum(0)
            return (r_red - r_ft) / (r_4 / df_den)

        f_obs = float(fstat(y[:, None])[0])
        fit_red = Hr @ y
        res_red = y - fit_red
        f_perm = fstat(fit_red[:, None] + res_red[idx])
        out[term] = (f_obs, float((1 + np.sum(f_perm >= f_obs - 1e-9)) / (n_perm + 1)))
    return out


def _p_col(table):
    """Name of the p-value column in an ANOVA table."""
    return [c for c in table.columns if c.lower().startswith("pr(")][0]


TERMS = {"housing": "C(housing)", "food_state": "C(food_state)", "interaction": "C(housing):C(food_state)"}


def anova_p(model, robust=None):
    """Type II ANOVA p-values for housing, food state and their interaction."""
    tab = sm.stats.anova_lm(model, typ=2, robust=robust) if robust else sm.stats.anova_lm(model, typ=2)
    col = _p_col(tab)
    return {t: float(tab.loc[lbl, col]) for t, lbl in TERMS.items()}


def welch(a, b):
    """Welch t-test with confidence interval and Hedges' g; None if either group is too small or has no spread."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb, va, vb = a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return None
    diff = ma - mb
    dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), dfw)
    tc = stats.t.ppf(0.975, dfw)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    g = (diff / sp) * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan
    return {
        "n_A": na,
        "n_B": nb,
        "mean_A": ma,
        "mean_B": mb,
        "diff": diff,
        "ci_low": diff - tc * se,
        "ci_high": diff + tc * se,
        "t": t,
        "df": dfw,
        "p_welch": p,
        "hedges_g": g,
    }


def planned_comparisons(d, ycol, scale, interaction_p):
    """
    Four planned simple-effect comparisons (Welch t-tests on the analysis scale):
      FD - AL within SH, FD - AL within GH, GH - SH within AL, GH - SH within FD.
    Holm-adjusted across these four. For log-scale metrics the difference is a log ratio
    and is also reported as a fold-change (ratio of geometric means).
    """
    gh, sh = HOUSING_LEVELS
    fd, al = FOOD_LEVELS
    specs = [
        (f"{fd} - {al} | {sh}", "food_state", fd, al, "housing", sh),
        (f"{fd} - {al} | {gh}", "food_state", fd, al, "housing", gh),
        (f"{gh} - {sh} | {al}", "housing", gh, sh, "food_state", al),
        (f"{gh} - {sh} | {fd}", "housing", gh, sh, "food_state", fd),
    ]
    rows = []
    for label, fac, la, lb, cond_fac, cond_lv in specs:
        sub = d[d[cond_fac] == cond_lv]
        res = welch(sub.loc[sub[fac] == la, ycol], sub.loc[sub[fac] == lb, ycol])
        row = {"comparison": label}
        if res is None:
            row.update({"n_A": np.nan})
        else:
            row.update(res)
            if scale == "log":
                row.update(
                    ratio=np.exp(res["diff"]),
                    ratio_ci_low=np.exp(res["ci_low"]),
                    ratio_ci_high=np.exp(res["ci_high"]),
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_welch"].to_numpy() if "p_welch" in out else np.full(len(out), np.nan))
    out["status"] = (
        "confirmatory" if (not np.isnan(interaction_p) and interaction_p < ALPHA) else "descriptive"
    )
    out["interaction_p_classic"] = interaction_p
    out["scale"] = scale
    return out


def agreement(ps):
    """Label whether classic, robust and permutation p-values agree: solid (sig), solid (ns) or fragile."""
    ps = [p for p in ps if not np.isnan(p)]
    if not ps:
        return ""
    sig = [p < ALPHA for p in ps]
    return "solid (sig)" if all(sig) else "solid (ns)" if not any(sig) else "fragile"


# =============================================================================
# Per-metric 2x2 analysis
# =============================================================================
def analyse_metric(df, col):
    """Returns (anova_row, check_row, comparisons_df, sens_row_or_None, model, data) or (None, reason...)."""
    scale = scale_of(col)
    need = ["key", "housing", "food_state", col] + [
        w for w in WEIGHTED_METRICS.values() if col in WEIGHTED_METRICS
    ]
    sub = df[list(dict.fromkeys(need))].dropna(subset=[col]).copy()
    raw = sub[col].astype(float)
    note = ""
    if scale == "log" and (raw <= 0).any():
        scale, note = "raw", "non-positive values: analysed on raw scale"
    sub["y"] = transform(raw, scale)
    cells = sub.groupby(["housing", "food_state"]).size()
    if len(cells) < 4 or cells.min() < MIN_PER_CELL:
        return (
            None,
            f"fewer than {MIN_PER_CELL} animals in at least one group (cell sizes: {cells.to_dict()})",
        )
    if sub["y"].nunique() < 3:
        return None, "values (almost) constant"
    n = len(sub)
    model = ols("y ~ C(housing) * C(food_state)", data=sub).fit()
    p_cl, p_hc = anova_p(model), anova_p(model, robust="hc3")
    h = np.where(sub["housing"] == HOUSING_LEVELS[0], 1.0, -1.0)
    f = np.where(sub["food_state"] == FOOD_LEVELS[0], 1.0, -1.0)
    perm = permutation_pvalues(h, f, sub["y"].to_numpy())
    p_pm = {t: perm[t][1] for t in TERMS}

    resid = model.resid.to_numpy()
    shapiro_p = float(stats.shapiro(resid).pvalue) if n >= 3 else np.nan
    groups = [g["y"].to_numpy() for _, g in sub.groupby(["housing", "food_state"])]
    levene_p = float(stats.levene(*groups, center="median").pvalue)
    sds = np.array([g.std(ddof=1) for g in groups])
    sd_ratio = float(sds.max() / sds.min()) if sds.min() > 0 else np.inf
    infl = model.get_influence()
    sres = np.asarray(infl.resid_studentized_external)
    cook = np.asarray(infl.cooks_distance[0])
    out_mask = np.abs(sres) > OUTLIER_Z  # Cook's D is reported but does not trigger the flag
    out_keys = sub.loc[out_mask, "key"].tolist()

    flags = []
    if shapiro_p < ALPHA:
        flags.append("non-normal residuals")
    if levene_p < ALPHA or sd_ratio > SD_RATIO_LIMIT:
        flags.append("unequal variances")
    if out_keys:
        flags.append("outliers")

    anova_row = {
        "metric": col,
        "primary": col in PRIMARY_METRICS,
        "scale": scale,
        "n_keys": n,
        "min_cell_n": int(cells.min()),
    }
    for t in TERMS:
        anova_row[f"p_{t}"] = p_cl[t]
    for t in TERMS:
        anova_row[f"p_{t}_hc3"] = p_hc[t]
    for t in TERMS:
        anova_row[f"p_{t}_perm"] = p_pm[t]
    for t in TERMS:
        anova_row[f"agree_{t}"] = agreement([p_cl[t], p_hc[t], p_pm[t]])
    anova_row["note"] = note

    if col in WEIGHTED_METRICS:
        try:
            wcol = WEIGHTED_METRICS[col]
            wls = smf.wls("y ~ C(housing) * C(food_state)", data=sub, weights=sub[wcol]).fit()
            pw = anova_p(wls)
            for t in TERMS:
                anova_row[f"p_{t}_wls"] = pw[t]
        except Exception as exc:  # noqa: BLE001
            anova_row["note"] = (anova_row["note"] + f" WLS failed: {exc}").strip()

    check_row = {
        "metric": col,
        "scale": scale,
        "n_keys": n,
        "shapiro_p": shapiro_p,
        "resid_skew": float(stats.skew(resid)),
        "levene_p": levene_p,
        "sd_ratio": sd_ratio,
        "n_outliers": len(out_keys),
        "outlier_keys": ";".join(out_keys),
        "max_cooks_d": float(cook.max()),
        "n_cooks_gt_4_over_n": int((cook > 4.0 / n).sum()),
        "flag_normality": shapiro_p < ALPHA,
        "flag_variance": bool(levene_p < ALPHA or sd_ratio > SD_RATIO_LIMIT),
        "flag_outliers": bool(out_keys),
        "summary": "; ".join(flags) if flags else "no flags",
    }

    sens = None
    if out_keys:
        keep = sub[~out_mask]
        if (
            keep.groupby(["housing", "food_state"]).size().reindex(cells.index).fillna(0).min() >= 2
            and keep["y"].nunique() > 2
        ):
            m2 = ols("y ~ C(housing) * C(food_state)", data=keep).fit()
            p2 = anova_p(m2)
            sens = {"metric": col, "n_removed": len(out_keys), "keys_removed": ";".join(out_keys)}
            for t in TERMS:
                sens[f"p_{t}_all"] = p_cl[t]
                sens[f"p_{t}_without"] = p2[t]
                sens[f"same_conclusion_{t}"] = (p_cl[t] < ALPHA) == (p2[t] < ALPHA)

    comps = planned_comparisons(sub, "y", scale, p_cl["interaction"])
    comps.insert(0, "metric", col)
    return (anova_row, check_row, comps, sens, model, sub), None


# =============================================================================
# Plots for the 2x2 results
# =============================================================================
def plot_group_metric(df, col, scale, anova_row, out_dir):
    """Save a box-and-points plot of one metric across the four groups."""
    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        vals = df.loc[df["group"] == g, col].dropna().to_numpy()
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.5,
            showfliers=False,
            medianprops=dict(color="black"),
            boxprops=dict(color="grey"),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
        )
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, len(vals)),
            vals,
            s=16,
            color=GROUP_COLORS.get(g, "#555"),
            alpha=0.85,
            zorder=3,
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [f"{g}\n(n={int(df.loc[df['group'] == g, col].notna().sum())})" for g in groups], fontsize=8
    )
    ax.set_ylabel(col)
    ttl = col + ("  (tested on log scale)" if scale in ("log", "log1p") else "")
    ax.set_title(ttl, fontsize=9)
    txt = (
        f"classic p: housing {anova_row['p_housing']:.3g} | food {anova_row['p_food_state']:.3g} | "
        f"inter {anova_row['p_interaction']:.3g}"
    )
    ax.text(0.5, -0.22, txt, transform=ax.transAxes, ha="center", fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{col}.png"), dpi=140)
    plt.close(fig)


def plot_diagnostics(model, col, scale, out_dir):
    """Save a residual QQ plot and a residuals-vs-fitted plot for one ANOVA model."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    stats.probplot(model.resid, dist="norm", plot=axes[0])
    axes[0].set_title("Normal QQ plot of residuals", fontsize=9)
    axes[1].scatter(model.fittedvalues, model.resid, s=16)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs fitted", fontsize=9)
    fig.suptitle(f"{col} ({scale} scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{col}_diagnostics.png"), dpi=130)
    plt.close(fig)


def run_group_stats(animal_metrics, meta, out_dir):
    """Run the 2x2 statistics for every per-animal metric and write the result tables and plots."""
    df = animal_metrics.merge(meta, on="key", how="inner")
    m_keys, g_keys = set(animal_metrics["key"]), set(meta["key"])
    if m_keys - g_keys:
        print(
            f"  ! {len(m_keys - g_keys)} key(s) have no group label and are excluded: {sorted(m_keys - g_keys)}"
        )
    if g_keys - m_keys:
        print(
            f"  ! {len(g_keys - m_keys)} key(s) in the group file have no metrics: {sorted(g_keys - m_keys)}"
        )
    plots_dir = os.path.join(out_dir, "group_plots")
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    cols = [
        c
        for c in animal_metrics.columns
        if c not in ("key",)
        and pd.api.types.is_numeric_dtype(animal_metrics[c])
        and not any(re.search(p, c) for p in ANOVA_EXCLUDE_PATTERNS)
    ]
    anova_rows, check_rows, comp_dfs, sens_rows, skipped = [], [], [], [], []
    for col in cols:
        res, reason = analyse_metric(df, col)
        if res is None:
            skipped.append({"metric": col, "reason": reason})
            continue
        a_row, c_row, comps, sens, model, sub = res
        anova_rows.append(a_row)
        check_rows.append(c_row)
        comp_dfs.append(comps)
        if sens:
            sens_rows.append(sens)
        plot_group_metric(df, col, a_row["scale"], a_row, plots_dir)
        plot_diagnostics(model, col, a_row["scale"], diag_dir)

    anova = pd.DataFrame(anova_rows)
    if not anova.empty:
        for t in TERMS:
            anova[f"q_{t}"] = bh_fdr(anova[f"p_{t}"].to_numpy())
        anova = anova.sort_values(["primary", "metric"], ascending=[False, True]).reset_index(drop=True)
        anova.to_csv(os.path.join(out_dir, "group_2x2_anova.csv"), index=False)
    pd.DataFrame(check_rows).to_csv(os.path.join(out_dir, "assumption_checks.csv"), index=False)
    if comp_dfs:
        pd.concat(comp_dfs, ignore_index=True).to_csv(
            os.path.join(out_dir, "planned_comparisons.csv"), index=False
        )
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(os.path.join(out_dir, "outlier_sensitivity.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(out_dir, "metrics_not_tested.csv"), index=False)
    return {"anova": anova, "checks": pd.DataFrame(check_rows), "skipped": skipped, "df": df}


# =============================================================================
# Time-course (binned) analysis
# =============================================================================
TC_RESPONSES = [
    ("preference_index", "raw", "Preference index (SS - FS)/(SS + FS)", "pref_bin_slope"),
    ("ss_mean_duration_s", "log", "SS bout duration (s)", "ss_logdur_bin_slope"),
    ("fs_mean_duration_s", "log", "FS bout duration (s)", "fs_logdur_bin_slope"),
]
TC_LABELS = {
    "Intercept": "Intercept (grand mean)",
    "h": "Housing (GH - SH)",
    "f": "Food state (FD - AL)",
    "b": "Bin (change per bin, all groups)",
    "h:f": "Housing x Food state",
    "h:b": "Housing x Bin",
    "f:b": "Food state x Bin",
    "h:f:b": "Housing x Food x Bin",
}


def fit_mixed(d):
    """Random-intercept (per animal) model  y ~ h*f*b  with +/-0.5 coded factors and centred bin."""
    last = None
    for method in ("lbfgs", "bfgs", "powell"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                res = smf.mixedlm("y ~ h*f*b", d, groups=d["key"]).fit(reml=True, method=method)
                if res.converged:
                    return res
                last = res
            except Exception as exc:  # noqa: BLE001
                last = exc
    return last if hasattr(last, "params") else None


def run_timecourse(binned_long, meta, out_dir):
    """Analyse the time-binned preference index and bout durations (mixed models, per-bin comparisons)."""
    d0 = binned_long.merge(meta, on="key", how="inner")
    d0["h"] = np.where(d0["housing"] == HOUSING_LEVELS[0], 0.5, -0.5)
    d0["f"] = np.where(d0["food_state"] == FOOD_LEVELS[0], 0.5, -0.5)
    n_bins = int(d0["bin"].max())
    d0["b"] = d0["bin"] - (n_bins + 1) / 2.0
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    term_rows, bin_rows, notes = [], [], []
    for resp, scale, label, _ in TC_RESPONSES:
        d = d0.dropna(subset=[resp]).copy()
        if scale == "log":
            d = d[d[resp] > 0]
        if d["key"].nunique() < 12 or d["bin"].nunique() < 3:
            notes.append(f"{resp}: too few animals/bins for a mixed model")
            continue
        d["y"] = transform(d[resp], scale)
        res = fit_mixed(d)
        if res is None:
            notes.append(f"{resp}: mixed model failed to fit")
            continue
        names = list(res.fe_params.index)
        ci = res.conf_int().loc[names]
        # diagnostics
        fixed = d[["h", "f", "b"]].assign(Intercept=1.0)
        fixed["h:f"], fixed["h:b"], fixed["f:b"] = d.h * d.f, d.h * d.b, d.f * d.b
        fixed["h:f:b"] = d.h * d.f * d.b
        marg = d["y"].to_numpy() - fixed[names].to_numpy() @ res.fe_params.to_numpy()
        try:
            re_map = {k: float(v.iloc[0]) for k, v in res.random_effects.items()}
        except Exception:  # noqa: BLE001  singular random-effects covariance (between-animal variance ~ 0)
            re_map = {k: 0.0 for k in d["key"].unique()}
            notes.append(
                f"{resp}: between-animal variance estimated at ~0 (singular fit); the model "
                f"reduces to an ordinary regression on all bins, so treat its p-values with caution"
            )
        cond = marg - d["key"].map(re_map).to_numpy()
        grp = [cond[(d["group"] == g).to_numpy()] for g in d["group"].unique()]
        sh_p = float(stats.shapiro(cond).pvalue)
        lev_p = float(stats.levene(*grp, center="median").pvalue)
        re_vals = np.array(list(re_map.values()))
        re_flat = bool(np.ptp(re_vals) < 1e-12)
        re_sh = float(stats.shapiro(re_vals).pvalue) if (len(re_vals) >= 3 and not re_flat) else np.nan
        var_re = float(res.cov_re.iloc[0, 0])
        triggered = any(res.pvalues[t] < ALPHA for t in ("h:b", "f:b", "h:f:b"))
        for t in names:
            term_rows.append(
                {
                    "response": resp,
                    "scale": scale,
                    "term": TC_LABELS.get(t, t),
                    "coef": t,
                    "estimate": res.fe_params[t],
                    "se": res.bse_fe[t],
                    "z": res.tvalues[t],
                    "p": res.pvalues[t],
                    "ci_low": ci.loc[t, 0],
                    "ci_high": ci.loc[t, 1],
                    "n_animals": d["key"].nunique(),
                    "n_obs": len(d),
                    "converged": bool(res.converged),
                    "random_intercept_var": var_re,
                    "singular_fit": bool(var_re < 1e-8 or re_flat),
                    "cond_resid_shapiro_p": sh_p,
                    "cond_resid_levene_p": lev_p,
                    "random_effects_shapiro_p": re_sh,
                    "bin_followup_triggered": triggered,
                }
            )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        stats.probplot(cond, dist="norm", plot=axes[0])
        axes[0].set_title("QQ: within-animal residuals", fontsize=9)
        if re_flat:
            axes[1].text(0.5, 0.5, "Random-intercept variance ~ 0\n(singular fit)", ha="center", va="center")
            axes[1].set_axis_off()
        else:
            stats.probplot(re_vals, dist="norm", plot=axes[1])
        axes[1].set_title("QQ: animal random intercepts", fontsize=9)
        fitted = fixed[names].to_numpy() @ res.fe_params.to_numpy()
        axes[2].scatter(fitted, cond, s=10)
        axes[2].axhline(0, color="grey", lw=0.8)
        axes[2].set_title("Residuals vs fitted", fontsize=9)
        fig.suptitle(f"Time-course model: {resp} ({scale} scale)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(diag_dir, f"timecourse_{resp}_diagnostics.png"), dpi=130)
        plt.close(fig)
        if triggered:
            for b, gb in d.groupby("bin"):
                comps = planned_comparisons(gb, "y", scale, np.nan)
                comps.insert(0, "bin", int(b))
                comps.insert(0, "response", resp)
                bin_rows.append(comps)
        else:
            notes.append(f"{resp}: no group x time term below {ALPHA}, bin-by-bin comparisons not run")
    terms = pd.DataFrame(term_rows)
    if not terms.empty:
        terms.to_csv(os.path.join(out_dir, "timecourse_mixed_models.csv"), index=False)
    bins = pd.DataFrame()
    if bin_rows:
        bins = pd.concat(bin_rows, ignore_index=True)
        bins["p_holm"] = np.nan
        for resp, idx in bins.groupby("response").groups.items():
            bins.loc[idx, "p_holm"] = holm(bins.loc[idx, "p_welch"].to_numpy())
        bins["status"] = "follow-up to significant group x time term"
        bins.to_csv(os.path.join(out_dir, "timecourse_bin_comparisons.csv"), index=False)
    return {"terms": terms, "bins": bins, "notes": notes, "data": d0}


def timecourse_crosscheck(tc, stats_out, out_dir):
    """Compare mixed-model group x bin terms with a 2x2 ANOVA on each animal's slope across bins."""
    if tc["terms"].empty or stats_out["anova"].empty:
        return pd.DataFrame()
    an = stats_out["anova"].set_index("metric")
    rows = []
    for resp, _, _, slope_col in TC_RESPONSES:
        t = tc["terms"][tc["terms"]["response"] == resp].set_index("coef")
        if t.empty or slope_col not in an.index:
            continue
        for coef, term in (("h:b", "housing"), ("f:b", "food_state"), ("h:f:b", "interaction")):
            p_mix, p_slope = float(t.loc[coef, "p"]), float(an.loc[slope_col, f"p_{term}"])
            both_sig, both_ns = (p_mix < ALPHA and p_slope < ALPHA), (p_mix >= ALPHA and p_slope >= ALPHA)
            rows.append(
                {
                    "response": resp,
                    "term": TC_LABELS[coef],
                    "p_mixed_model": p_mix,
                    "slope_metric": slope_col,
                    "p_slope_anova": p_slope,
                    "agreement": (
                        "both significant" if both_sig else "both not significant" if both_ns else "DISAGREE"
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "timecourse_crosscheck.csv"), index=False)
    return out


# --- GEE fit of each group's time course (used for the fitted lines) --------------------------
TC_PLOT_SPECS = [  # response, analysis scale, title, y label, file stem
    (
        "preference_index",
        "raw",
        "Preference: social (SS) vs food (FS) zone",
        "Preference index (+1 social, -1 food)",
        "preference_index",
    ),
    (
        "ss_mean_duration_s",
        "log",
        "Bout duration in social stimulus zone (SS)",
        "Mean bout duration (s)",
        "ss_bout_duration",
    ),
    (
        "fs_mean_duration_s",
        "log",
        "Bout duration in food stimulus zone (FS)",
        "Mean bout duration (s)",
        "fs_bout_duration",
    ),
]
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
}


def _log_link():
    """Log link object, compatible with different statsmodels versions."""
    lk = sm_families.links
    return (getattr(lk, "Log", None) or lk.log)()


GEE_LABELS = {
    "Intercept": "Intercept (mid-session, average of the 4 groups)",
    "h": "Housing (GH - SH)",
    "f": "Food state (FD - AL)",
    "h:f": "Housing x Food state",
    "t": "Time: slope per minute (average of the 4 groups)",
    "h:t": "Housing x Time",
    "f:t": "Food state x Time",
    "h:f:t": "Housing x Food x Time",
    "t2": "Time^2",
    "h:t2": "Housing x Time^2",
    "f:t2": "Food state x Time^2",
    "h:f:t2": "Housing x Food x Time^2",
}


def _family(name):
    """Build the GEE family: ``gamma_log``, ``gaussian_log`` or plain Gaussian."""
    if name == "gamma_log":
        return sm_families.Gamma(_log_link())
    if name == "gaussian_log":
        return sm_families.Gaussian(_log_link())
    return sm_families.Gaussian()


def _design(h, f, t, degree=1):
    """Columns: 1, h, f, h*f, then t, h*t, f*t, h*f*t (and the same for t^2 if degree = 2).
    h = +0.5 for GH / -0.5 for SH, f = +0.5 for FD / -0.5 for AL, t = minutes from mid-session."""
    h, f, t = (np.asarray(v, float) for v in (h, f, t))
    cols, names = [np.ones_like(t), h, f, h * f], ["Intercept", "h", "f", "h:f"]
    for pw in range(1, degree + 1):
        tp, nm = t**pw, ("t" if pw == 1 else f"t{pw}")
        cols += [tp, h * tp, f * tp, h * f * tp]
        names += [nm, f"h:{nm}", f"f:{nm}", f"h:f:{nm}"]
    return np.column_stack(cols), names


def _gee_fit(y, X, codes, time, family, corr):
    """One GEE fit. Tries the small-sample (bias-reduced) covariance, then the ordinary robust one; a Gamma
    model that will not converge falls back to Gaussian with a log link. Returns None if everything fails."""
    for fam in [family] + (["gaussian_log"] if family == "gamma_log" else []):
        for cov_type in (GEE_COV_TYPE, "robust"):
            try:
                cs = Autoregressive() if corr == "ar1" else Exchangeable()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = GEE(
                        y,
                        X,
                        groups=codes,
                        time=time if corr == "ar1" else None,
                        family=_family(fam),
                        cov_struct=cs,
                    ).fit(cov_type=cov_type)
                params = np.asarray(res.params, float)
                cov = np.asarray(res.cov_params(), float)
                if (
                    np.all(np.isfinite(params))
                    and np.all(np.isfinite(cov))
                    and cov.shape == (len(params),) * 2
                ):
                    return {
                        "res": res,
                        "params": params,
                        "cov": cov,
                        "family": fam,
                        "link": "log" if fam.endswith("_log") else "identity",
                        "cov_type": cov_type,
                        "corr": corr,
                    }
            except Exception:  # noqa: BLE001
                continue
    return None


def _qic(res, scale=None):
    """QIC and QICu of a fitted GEE (NaN, NaN if they cannot be computed)."""
    for kw in (({"scale": scale} if scale is not None else {}), {}):
        try:
            q = res.qic(**kw)
            return float(q[0]), float(q[1])
        except Exception:  # noqa: BLE001
            continue
    return np.nan, np.nan


def fit_gee_model(df, ycol, tcol, family, corr, mid, degree=1):
    """housing x food state x time, animals (keys) as clusters.
    df needs key, group, h, f, ycol, tcol (minutes)
    and, for AR(1), the column 'bin'."""
    sub = df.dropna(subset=[ycol, tcol]).copy()
    if family != "gaussian":
        sub = sub[sub[ycol] > 0]
    sub = sub.sort_values(["key", tcol])
    if sub.empty:
        return None
    codes, _ = pd.factorize(sub["key"])
    X, names = _design(sub["h"], sub["f"], sub[tcol] - mid, degree)
    time = sub["bin"].to_numpy(float) if "bin" in sub.columns else None
    fit = _gee_fit(sub[ycol].to_numpy(float), X, codes, time, family, corr)
    if fit is None:
        return None
    fit.update(
        {
            "names": names,
            "n_animals": int(sub["key"].nunique()),
            "n_obs": len(sub),
            "group_n": sub.groupby("group")["key"].nunique().to_dict(),
            "degree": degree,
        }
    )
    return fit


def _group_vectors(names, g):
    """Contrast vectors picking a group's intercept and slope per minute out of the coefficient vector."""
    hs = 0.5 if g.startswith(HOUSING_LEVELS[0]) else -0.5
    fs = 0.5 if g.endswith(FOOD_LEVELS[0]) else -0.5
    ix = {n: i for i, n in enumerate(names)}
    a, sl = np.zeros(len(names)), np.zeros(len(names))
    for n, v in (("Intercept", 1.0), ("h", hs), ("f", fs), ("h:f", hs * fs)):
        a[ix[n]] = v
    for n, v in (("t", 1.0), ("h:t", hs), ("f:t", fs), ("h:f:t", hs * fs)):
        sl[ix[n]] = v
    return a, sl


def _tp(est, se, df):
    """t statistic and two-sided p-value for an estimate and its standard error."""
    t = est / se if se and np.isfinite(se) and se > 0 else np.nan
    return t, float(2 * stats.t.sf(abs(t), max(df, 1))) if np.isfinite(t) else np.nan


def gee_tables(fit, response, analysis):
    """Coefficient, per-group slope and slope-contrast tables for one fitted model."""
    names, b, C = fit["names"], fit["params"], fit["cov"]
    log = fit["link"] == "log"
    info = {
        "response": response,
        "analysis": analysis,
        "family": fit["family"],
        "link": fit["link"],
        "correlation": fit["corr"],
        "cov_type": fit["cov_type"],
    }
    df_m = fit["n_animals"] - len(names)
    crit_m = stats.t.ppf(0.975, max(df_m, 1))
    coef = []
    for i, n in enumerate(names):
        se = float(np.sqrt(C[i, i]))
        t, p = _tp(b[i], se, df_m)
        row = {
            **info,
            "term": n,
            "label": GEE_LABELS.get(n, n),
            "estimate": b[i],
            "se": se,
            "t": t,
            "p": p,
            "ci_low": b[i] - crit_m * se,
            "ci_high": b[i] + crit_m * se,
            "n_animals": fit["n_animals"],
            "n_obs": fit["n_obs"],
        }
        if log:
            row.update(
                ratio=np.exp(b[i]),
                ratio_ci_low=np.exp(b[i] - crit_m * se),
                ratio_ci_high=np.exp(b[i] + crit_m * se),
            )
        coef.append(row)
    slopes, contrasts = [], []
    vec = {g: _group_vectors(names, g) for g, _ in PLOT_GROUPS}
    for g, _lab in PLOT_GROUPS:
        a, sl = vec[g]
        est, se = float(sl @ b), float(np.sqrt(sl @ C @ sl))
        n_g = int(fit["group_n"].get(g, 0))
        t, p = _tp(est, se, n_g - 2)
        crit = stats.t.ppf(0.975, max(n_g - 2, 1))
        row = {
            **info,
            "group": g,
            "n_animals": n_g,
            "slope_per_min": est,
            "se": se,
            "t": t,
            "p": p,
            "ci_low": est - crit * se,
            "ci_high": est + crit * se,
        }
        if log:
            row.update(
                ratio_per_min=np.exp(est),
                ratio_ci_low=np.exp(est - crit * se),
                ratio_ci_high=np.exp(est + crit * se),
            )
        slopes.append(row)
    H0, H1, F0, F1 = HOUSING_LEVELS[0], HOUSING_LEVELS[1], FOOD_LEVELS[0], FOOD_LEVELS[1]
    pairs = [
        (f"{F0} - {F1} | {H1}", H1 + F0, H1 + F1),
        (f"{F0} - {F1} | {H0}", H0 + F0, H0 + F1),
        (f"{H0} - {H1} | {F1}", H0 + F1, H1 + F1),
        (f"{H0} - {H1} | {F0}", H0 + F0, H1 + F0),
    ]
    for lab, g1, g2 in pairs:
        L = vec[g1][1] - vec[g2][1]
        est, se = float(L @ b), float(np.sqrt(L @ C @ L))
        n1, n2 = int(fit["group_n"].get(g1, 0)), int(fit["group_n"].get(g2, 0))
        t, p = _tp(est, se, n1 + n2 - 4)
        crit = stats.t.ppf(0.975, max(n1 + n2 - 4, 1))
        row = {
            **info,
            "comparison": lab,
            "n_A": n1,
            "n_B": n2,
            "slope_difference": est,
            "se": se,
            "t": t,
            "p": p,
            "ci_low": est - crit * se,
            "ci_high": est + crit * se,
        }
        if log:
            row.update(
                ratio_of_ratios_per_min=np.exp(est),
                ratio_ci_low=np.exp(est - crit * se),
                ratio_ci_high=np.exp(est + crit * se),
            )
        contrasts.append(row)
    return coef, slopes, contrasts


def group_line(fit, g, mid):
    """Intercept + slope (and their 2x2 covariance) of one group's line, for the fitted curve and band."""
    a, sl = _group_vectors(fit["names"], g)
    A = np.vstack([a, sl])
    crit = stats.t.ppf(0.975, max(fit["n_animals"] - len(fit["names"]), 1))
    return {
        "link": fit["link"],
        "params": A @ fit["params"],
        "cov": A @ fit["cov"] @ A.T,
        "mid": mid,
        "crit": float(crit),
    }


def predict_gee(fit, xs):
    """Fitted curve and 95% band on the response scale; xs in minutes from the start of the session."""
    Xp = np.column_stack([np.ones_like(xs), xs - fit.get("mid", 0.0)])
    eta = Xp @ fit["params"]
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xp, fit["cov"], Xp))
    crit = fit.get("crit", 1.96)
    inv = np.exp if fit["link"] == "log" else (lambda v: v)
    return inv(eta), inv(eta - crit * se), inv(eta + crit * se)


def build_bout_level(bout_dir, keys, meta):
    """Every SS / FS bout as one row: key, zone, start (minutes from the start of the window), duration."""
    rows = []
    for key in keys:
        try:
            det, col = load_bouts(bout_dir, key)
        except Exception:  # noqa: BLE001
            continue
        t0 = float(col["start_time_s"].min())
        for zone in STIM_STATES:
            sel = det[det["roi"] == zone]
            rows.append(
                pd.DataFrame(
                    {
                        "key": key,
                        "zone": zone,
                        "start_min": (sel["start_time_s"] - t0) / 60.0,
                        "duration_s": sel["duration_s"],
                    }
                )
            )
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).merge(meta[["key", "housing", "food_state", "group"]], on="key")
    out["h"] = np.where(out["housing"] == HOUSING_LEVELS[0], 0.5, -0.5)
    out["f"] = np.where(out["food_state"] == FOOD_LEVELS[0], 0.5, -0.5)
    return out


def run_gee_fits(d, out_dir, bout_dir=None):
    """
    One GEE per response (housing x food state x time, animals as clusters), plus sensitivity fits.
      preference_index   : Gaussian, identity link
      SS / FS duration   : Gamma, log link   (curve = mean duration in seconds; slopes are ratios per minute)
    Primary analysis uses each animal's value in each bin. Sensitivity analyses:
      preference -> drop bins with < PREF_MIN_ZONE_S seconds of SS+FS time
      durations  -> bout level (every bout one observation, clustered by animal; no empty bins)
    Also saves a QIC comparison of linear vs quadratic time and of exchangeable vs AR(1) correlation.
    Returns (fits for plotting, dict of result tables).
    """
    d = d.copy()
    d["t_min"] = d["bin_center_rel_s"] / 60.0
    mid = float(d.groupby("bin")["bin_end_rel_s"].median().max()) / 60.0 / 2.0  # session midpoint (minutes)
    meta = d[["key", "housing", "food_state", "group"]].drop_duplicates("key")
    bouts = (
        build_bout_level(bout_dir, d["key"].unique(), meta)
        if (bout_dir and GEE_BOUT_LEVEL)
        else pd.DataFrame()
    )
    coef_rows, slope_rows, contrast_rows, qic_rows, plot_fits = [], [], [], [], {}
    for resp, scale, *_ in TC_PLOT_SPECS:
        family, corr = GEE_SPECS[resp]
        analyses = [("primary: animal x bin", d, resp, "t_min", corr)]
        if resp == "preference_index":
            keep = (d["ss_time_s"] + d["fs_time_s"]) >= PREF_MIN_ZONE_S
            analyses.append(
                (f"sensitivity: bins with >= {PREF_MIN_ZONE_S:g} s in SS+FS", d[keep], resp, "t_min", corr)
            )
        elif not bouts.empty:
            zone = "SS" if resp.startswith("ss_") else "FS"
            analyses.append(
                (
                    "sensitivity: bout level",
                    bouts[bouts["zone"] == zone],
                    "duration_s",
                    "start_min",
                    "exchangeable",
                )
            )
        for label, df, ycol, tcol, cr in analyses:
            n_g = df.dropna(subset=[ycol]).groupby("group")["key"].nunique()
            if any(n_g.get(g, 0) < GEE_MIN_ANIMALS for g, _ in PLOT_GROUPS):
                print(f"  ! GEE {resp} [{label}]: a group has fewer than {GEE_MIN_ANIMALS} animals - skipped")
                continue
            fit = fit_gee_model(df, ycol, tcol, family, cr, mid, degree=1)
            if fit is None:
                print(f"  ! GEE {resp} [{label}]: model did not converge - skipped")
                continue
            if fit["family"] != family:
                print(f"  note: GEE {resp} [{label}]: Gamma did not converge, used Gaussian with log link")
            if fit["cov_type"] != GEE_COV_TYPE:
                print(
                    f"  note: GEE {resp} [{label}]: bias-reduced covariance unavailable, used ordinary robust"
                )
            c, sl, ct = gee_tables(fit, resp, label)
            coef_rows += c
            slope_rows += sl
            contrast_rows += ct
            if label.startswith("primary"):
                plot_fits[resp] = {g: group_line(fit, g, mid) for g, _ in PLOT_GROUPS}
                # does a straight line (on the link scale) and the chosen correlation structure hold up?
                alt = fit_gee_model(df, ycol, tcol, family, cr, mid, degree=2)
                scale_q = getattr(alt["res"], "scale", None) if alt else None
                other = "ar1" if cr == "exchangeable" else "exchangeable"
                alt_corr = fit_gee_model(df, ycol, tcol, family, other, mid, degree=1)
                for name, m in (
                    (f"linear time | {cr}", fit),
                    ("quadratic time | " + cr, alt),
                    (f"linear time | {other}", alt_corr),
                ):
                    if m is None:
                        qic_rows.append(
                            {
                                "response": resp,
                                "model": name,
                                "n_params": np.nan,
                                "QIC": np.nan,
                                "QICu": np.nan,
                            }
                        )
                        continue
                    q, qu = _qic(m["res"], scale_q)
                    qic_rows.append(
                        {"response": resp, "model": name, "n_params": len(m["names"]), "QIC": q, "QICu": qu}
                    )
    tables = {
        "coef": pd.DataFrame(coef_rows),
        "slopes": pd.DataFrame(slope_rows),
        "contrasts": pd.DataFrame(contrast_rows),
        "qic": pd.DataFrame(qic_rows),
        "mid": mid,
    }
    for key in ("slopes", "contrasts"):
        t = tables[key]
        if not t.empty:
            t["p_holm"] = np.nan
            for _, idx in t.groupby(["response", "analysis"]).groups.items():
                t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"].to_numpy())
    for name, key in (
        ("timecourse_gee_model.csv", "coef"),
        ("timecourse_gee_slopes.csv", "slopes"),
        ("timecourse_gee_slope_contrasts.csv", "contrasts"),
        ("timecourse_gee_time_shape_qic.csv", "qic"),
    ):
        if not tables[key].empty:
            tables[key].to_csv(os.path.join(out_dir, name), index=False)
    return plot_fits, tables


def print_gee_summary(tables):
    """Print the GEE coefficients and per-group slopes to the console."""
    c, sl, ct = tables["coef"], tables["slopes"], tables["contrasts"]
    if c.empty:
        return
    print("\n" + "=" * 78)
    print(" GEE time courses (housing x food state x time; animals as clusters)")
    print("=" * 78)
    for (resp, an), g in c.groupby(["response", "analysis"], sort=False):
        print(
            f" {resp}  [{an}]  {g['family'].iloc[0]}, {g['link'].iloc[0]} link, {g['correlation'].iloc[0]}, "
            f"{g['cov_type'].iloc[0]}; {g['n_animals'].iloc[0]} animals, {g['n_obs'].iloc[0]} obs"
        )
        for r in g[g["term"].isin(["t", "h:t", "f:t", "h:f:t"])].itertuples():
            ratio = (
                f"  (x{r.ratio:.3f}/min)"
                if hasattr(r, "ratio") and pd.notna(getattr(r, "ratio", np.nan))
                else ""
            )
            print(f"    {r.label:<52} est {r.estimate:+.4f}{ratio}  p = {r.p:.4f}")
        for r in sl[(sl["response"] == resp) & (sl["analysis"] == an)].itertuples():
            ratio = (
                f"  (x{r.ratio_per_min:.3f}/min)"
                if hasattr(r, "ratio_per_min") and pd.notna(getattr(r, "ratio_per_min", np.nan))
                else ""
            )
            print(
                f"    slope {r.group:<6} {r.slope_per_min:+.4f}{ratio}  p = {r.p:.4f}  Holm p = {r.p_holm:.4f}"
            )
    print(" (full tables: timecourse_gee_model / _slopes / _slope_contrasts / _time_shape_qic .csv)")


# --- figures --------------------------------------------------------------------------------
def _fmt_min(v):
    """Format a number of minutes for an axis label."""
    return f"{round(float(v), 1):g}"


def _axis_info(d):
    """Bin centres / edges in minutes (median across animals) and the tick labels such as '0–1', '1–2'."""
    b = (
        d.groupby("bin")
        .agg(
            start=("bin_start_rel_s", "median"),
            end=("bin_end_rel_s", "median"),
            width=("bin_width_s", "median"),
        )
        .sort_index()
        / 60.0
    )
    centers = ((b["start"] + b["end"]) / 2).to_numpy()
    labels = [f"{_fmt_min(a)}\u2013{_fmt_min(z)}" for a, z in zip(b["start"], b["end"])]
    return {
        "centers": centers,
        "labels": labels,
        "total": float(b["end"].max()),
        "bin_min": float(BIN_SECONDS) / 60.0,
    }


def _bin_points(d, resp):
    """Group mean, SD, SEM and animal count per bin for plotting."""
    sub = d.dropna(subset=[resp])
    pts = (
        sub.groupby(["group", "bin"])
        .agg(x=("bin_center_rel_s", "mean"), mean=(resp, "mean"), sd=(resp, "std"), n=("key", "nunique"))
        .reset_index()
    )
    pts["x"] = pts["x"] / 60.0
    pts["sem"] = pts["sd"] / np.sqrt(pts["n"])
    return pts[pts["n"] >= MIN_ANIMALS_PER_POINT]


def _fit_xs(info):
    """x positions (minutes) at which the GEE line is evaluated."""
    if FIT_SPAN_TO_BIN_CENTERS and len(info["centers"]) > 1:
        return np.linspace(float(info["centers"][0]), float(info["centers"][-1]), 200)
    return np.linspace(0, info["total"], 200)


def _shared_duration_ymax(d, gee, info):
    """One y-axis maximum for all bout-duration figures (so SS and FS are directly comparable)."""
    if DURATION_YMAX is not None:
        return float(DURATION_YMAX)
    top = 0.0
    xs = _fit_xs(info)
    for resp, _s, _t, _y, _stem in TC_PLOT_SPECS:
        if resp == "preference_index":
            continue
        pts = _bin_points(d, resp)
        if not pts.empty:
            top = max(top, float((pts["mean"] + np.nan_to_num(pts["sem"])).max()))
        for g, fit in (gee or {}).get(resp, {}).items():
            _mu, _lo, hi = predict_gee(fit, xs)
            top = max(top, float(np.nanmax(hi)))
    return float(np.ceil(top * 1.05 / 5.0) * 5.0) if top > 0 else None


def _draw_timecourse(pts, fits, spec, info, with_fit, out_base, ymax=None):
    """Draw one time-course figure (points, optionally with the GEE line and 95% band) and save it."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    resp, scale, title, ylab, _stem = spec
    show_err = FIT_FIGURE_ERRORBARS if with_fit else POINTS_ONLY_ERRORBARS
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    vals, handles = [], []
    xs = _fit_xs(info)
    for i, (g, lab) in enumerate(PLOT_GROUPS):
        col = GROUP_COLORS[g]
        p = pts[pts["group"] == g].sort_values("x")
        if p.empty and not (with_fit and g in fits):
            continue
        off = (i - 1.5) * DODGE_FRACTION * info["bin_min"]
        if with_fit and g in fits:
            mu, lo, hi = predict_gee(fits[g], xs)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.22, lw=0, zorder=1)
            ax.plot(xs, mu, color=col, lw=2.8, solid_capstyle="round", zorder=2)
            vals += [lo, hi]
        if not p.empty:
            if (not with_fit) and POINTS_ONLY_CONNECT:
                ax.plot(p["x"] + off, p["mean"], color=col, lw=1.0, alpha=0.6, zorder=2)
            yerr = np.nan_to_num(p["sem"].to_numpy()) if show_err else None
            ax.errorbar(
                p["x"] + off,
                p["mean"],
                yerr=yerr,
                fmt="o",
                ms=4.5 if with_fit else 6,
                color=col,
                ecolor=col,
                elinewidth=1.3,
                capsize=2.5,
                alpha=0.85,
                linestyle="none",
                zorder=3,
            )
            vals.append(p["mean"].to_numpy())
            if show_err:
                vals += [
                    (p["mean"] - np.nan_to_num(p["sem"])).to_numpy(),
                    (p["mean"] + np.nan_to_num(p["sem"])).to_numpy(),
                ]
        if with_fit and g in fits:
            handles.append(Line2D([0], [0], color=col, lw=3.2, label=lab))
        else:
            handles.append(Line2D([0], [0], color=col, marker="o", ms=6, lw=0, label=lab))
    if not vals:
        plt.close(fig)
        return False
    allv = np.concatenate([np.ravel(v) for v in vals])
    allv = allv[np.isfinite(allv)]
    lo, hi = float(allv.min()), float(allv.max())
    if resp == "preference_index":
        ax.axhline(0, color="#9a9a9a", lw=1.0, ls="--", zorder=0)
        ax.set_ylim(-1.0, 1.0)  # index is bounded: show exactly -1 to +1
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    elif ymax is not None:
        ax.set_ylim(0, ymax)
    else:
        rng = max(hi, 1e-6)
        ax.set_ylim(0, hi + 0.5 * rng)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlim(0, info["total"])
    step = int(np.ceil(len(info["centers"]) / 12.0))
    ax.set_xticks(info["centers"])
    ax.set_xticklabels([l if i % step == 0 else "" for i, l in enumerate(info["labels"])])
    ax.tick_params(axis="both", direction="out", length=5, width=1.5, labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5)
    ax.set_xlabel("Time in session (minutes)", fontsize=12.5, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=12.5, fontweight="bold")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=10)
    ax.legend(
        handles=handles,
        loc="lower right" if resp == "preference_index" else "upper right",
        frameon=False,
        prop={"weight": "bold", "size": 11},
        handlelength=1.8,
        labelspacing=0.5,
    )
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    return True


def plot_timecourse(tc_data, out_dir, gee=None):
    """
    For each response two figures, same style (bold Arial, open axes, minute-range bin labels):
      timecourse_<stem>_points.png : group means per bin (points + SEM bars), no fit
      timecourse_<stem>_GEE.png    : group means per bin + GEE line and 95% band per group
    Bins with no group mean are skipped.
    """
    d = tc_data
    info = _axis_info(d)
    dur_ymax = _shared_duration_ymax(d, gee, info) if SHARED_DURATION_YAXIS else None
    with plt.rc_context(PLOT_STYLE):
        for spec in TC_PLOT_SPECS:
            resp, _scale, _title, _ylab, stem = spec
            pts = _bin_points(d, resp)
            ym = dur_ymax if resp != "preference_index" else None
            _draw_timecourse(
                pts, {}, spec, info, False, os.path.join(out_dir, f"timecourse_{stem}_points"), ymax=ym
            )
            fits = (gee or {}).get(resp, {})
            if fits:
                _draw_timecourse(
                    pts, fits, spec, info, True, os.path.join(out_dir, f"timecourse_{stem}_GEE"), ymax=ym
                )
            else:
                print(f"  ! No GEE fit available for {resp}: GEE figure not drawn")


# =============================================================================
# Pipeline
# =============================================================================
def set_bin_seconds(value):
    """Set the global time-bin width (seconds)."""
    global BIN_SECONDS
    BIN_SECONDS = float(value)


def run_pipeline(bout_dir, out_dir, group_file=None, make_key_plots=True, bin_seconds=None):
    """Run the whole analysis: per-animal metrics, group statistics, time courses and GEE fits."""
    if bin_seconds is not None:
        set_bin_seconds(bin_seconds)
    print(f"Time-bin length: {BIN_SECONDS:g} s")
    os.makedirs(out_dir, exist_ok=True)
    per_key_dir = os.path.join(out_dir, "per_key")
    os.makedirs(per_key_dir, exist_ok=True)
    meta = load_metadata(group_file) if group_file else None

    keys = find_keys(bout_dir)
    if not keys:
        raise SystemExit("No matching bout file pairs found in that folder.")
    print(f"Found {len(keys)} key(s).")
    metrics, binned = [], []
    for key in keys:
        try:
            m, b = analyse_key(bout_dir, key, per_key_dir, make_key_plots)
            metrics.append(m)
            binned.append(b)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {key} FAILED: {type(exc).__name__}: {exc}")
    am = pd.DataFrame(metrics)
    bl = pd.concat(binned, ignore_index=True) if binned else pd.DataFrame()
    if not bl.empty:
        nb = bl.groupby("key")["bin"].max().value_counts().sort_index()
        print("Bins per animal: " + ", ".join(f"{int(k)} bins: {int(v)} animal(s)" for k, v in nb.items()))
        short = bl[bl["bin_width_s"] < 0.98 * BIN_SECONDS]
        longer = bl[bl["bin_width_s"] > 1.02 * BIN_SECONDS]
        if len(short) or len(longer):
            print(
                f"  note: {longer['key'].nunique()} animal(s) have a last bin longer than {BIN_SECONDS:g} s "
                f"(remainder added) and {short['key'].nunique()} a last bin slightly shorter."
            )
    am_out = am.merge(meta, on="key", how="left") if meta is not None else am
    bl_out = bl.merge(meta, on="key", how="left") if meta is not None else bl
    am_out.to_csv(os.path.join(out_dir, "animal_metrics.csv"), index=False)
    bl_out.to_csv(os.path.join(out_dir, "binned_long.csv"), index=False)
    with open(os.path.join(out_dir, "analysis_settings.json"), "w") as fh:
        json.dump(
            {
                k: v
                for k, v in globals().items()
                if k.isupper() and isinstance(v, (int, float, str, list, tuple, dict))
            },
            fh,
            indent=1,
            default=str,
        )
    print(f"Per-animal metrics: {len(am)} key(s) -> animal_metrics.csv; binned data -> binned_long.csv")
    if meta is None:
        print("No group file given: group statistics skipped.")
        return {"animal_metrics": am}

    st = run_group_stats(am, meta, out_dir)
    tc = run_timecourse(bl, meta, out_dir)
    gee, gee_tables = run_gee_fits(tc["data"], out_dir, bout_dir)
    plot_timecourse(tc["data"], out_dir, gee)
    cc = timecourse_crosscheck(tc, st, out_dir)
    print_summary(st, tc, cc)
    print_gee_summary(gee_tables)
    return {
        "animal_metrics": am,
        "binned": bl,
        "stats": st,
        "timecourse": tc,
        "crosscheck": cc,
        "gee": gee_tables,
    }


def print_summary(st, tc, cc):
    """Print a console summary of the ANOVA, time-course and cross-check results."""
    a = st["anova"]
    print("\n" + "=" * 78)
    print(" 2x2 ANOVA: primary metrics (classic p | HC3 p | permutation p -> agreement)")
    print("=" * 78)
    for r in a[a["primary"]].itertuples():
        print(f" {r.metric}  (scale: {r.scale}, n={r.n_keys})")
        for t, lab in (("housing", "Housing"), ("food_state", "Food state"), ("interaction", "Interaction")):
            print(
                f"    {lab:<12} {getattr(r, 'p_' + t):.4f} | {getattr(r, 'p_' + t + '_hc3'):.4f} | "
                f"{getattr(r, 'p_' + t + '_perm'):.4f}  -> {getattr(r, 'agree_' + t)}"
            )
    if st["skipped"]:
        print(f"\n {len(st['skipped'])} metric(s) not tested (see metrics_not_tested.csv)")
    flagged = st["checks"][st["checks"]["summary"] != "no flags"]
    print(
        f"\n Assumption flags on {len(flagged)} of {len(st['checks'])} tested metrics (see assumption_checks.csv)"
    )
    for n in tc["notes"]:
        print("  note:", n)
    if not cc.empty:
        print("\n Time-course vs slope-ANOVA cross-check:")
        print(
            cc[["response", "term", "p_mixed_model", "p_slope_anova", "agreement"]]
            .round(4)
            .to_string(index=False)
        )


def main():
    """Interactive entry point: ask for the input/output paths and bin length, then run the pipeline."""
    print("=" * 70)
    print(
        " SFPT motivation analysis (bout and preference)"
    )
    print("=" * 70)
    while True:
        bout_dir = input("Folder with the bout CSVs from roi_bout_analysis.py: ").strip().strip("'\"")
        if os.path.isdir(bout_dir):
            break
        print("  That folder does not exist.")
    out_dir = input("Output folder [default: <bout folder>/motivation_output_v3]: ").strip().strip("'\"")
    out_dir = out_dir or os.path.join(bout_dir, "bout_and_preference_output")
    group_file = input("Group file (key, housing, food_state) -- press Enter to skip group statistics: ")
    group_file = group_file.strip().strip("'\"") or None
    while True:
        raw = input(f"Time-bin length in seconds [default {BIN_SECONDS:g}]: ").strip()
        if not raw:
            bin_s = BIN_SECONDS
            break
        try:
            bin_s = float(raw)
            if bin_s > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive number of seconds.")
    run_pipeline(bout_dir, out_dir, group_file, bin_seconds=bin_s)
    print(f"\nAll outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
