#!/usr/bin/env python3
"""
ROI bout / transition analysis for DeepLabCut output.

For every DLC CSV in an input folder this script
  1. reads the arena/ROI polygons straight from the CSV itself
     (columns `all_roi_vertices` and/or `roi_<NAME>_vertices`)  -> no JSON files needed,
  2. asks which body part(s) and which condition (cascade / any / all / single)
     define "the animal is inside an ROI" (same logic as the ROI placement tool),
  3. assigns each frame to one state (SS / FS / SC / FC / Neutral),
  4. merges short exits (grace period) into bouts,
  5. writes a bout table (start frame, end frame, duration, transitions),
  6. writes a second bout table with SS->SC and FS->FC collapsed (SC / FC / Neutral),
  7. draws a timeline-strip ethogram with time spent in each region
     (plot title = group name only, e.g. GHAL; the PNG file name keeps the movie name),
  8. optionally sorts every video's outputs into one sub-folder per group (GHAL, GHFD, ...)
     using a master sheet with a `key` column and `housing` + `food_state` (or `group`) columns.

Run:  python roi_bout_analysis.py
Requires: numpy, pandas, matplotlib, (openpyxl only if your start-frame sheet is .xlsx)
"""
import ast
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.path import Path

# =============================================================================
# TUNABLE PARAMETERS (can also be changed at run time when the script asks)
# =============================================================================
GRACE_FRAMES = 10             # an exit of <= this many frames is absorbed into the bout
GAP_FRACTION_THRESHOLD = 0.5  # flag bouts where MORE than this fraction of frames are gap frames
WINDOW_FRAMES = 15000         # frames analysed after the start frame
GAP_QUALITY_LABELS = ("long_gap_filled", "interpolated_short_gap")  # qualities counted as "gap"
QUALITY_BODYPARTS = ("nose", "center")  # a frame is a gap frame if any of these is a gap
FPS = 25.0                    # frames per second (asked at run time)
NEUTRAL = "Neutral"

# Frame -> state priority. Later entries win when a frame is inside several regions.
# (SC/FC are the chambers; SS/FS are the small zones, so the small zone wins.)
PRIORITY_LOW_TO_HIGH = ["SC", "FC", "SS", "FS"]
COLLAPSE_MAP = {"SS": "SC", "FS": "FC"}

COLORS = {"SC": "#1f77b4", "SS": "#9ecae1", "FC": "#d95f02", "FS": "#fdb863", NEUTRAL: "#bdbdbd"}

# Same options as the ROI placement tool
CONDITION_KEYS = ["cascade", "any", "all", "single"]
CONDITION_DESCS = [
    "cascade - use the first selected bodypart that has a valid coordinate (best for small zones)",
    "any     - frame counts if ANY selected bodypart is inside",
    "all     - frame counts only if ALL selected bodyparts are inside",
    "single  - use only the first selected bodypart",
]


# =============================================================================
# Small interactive helpers
# =============================================================================
def ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    if ans.startswith(("'", '"')) and ans.endswith(("'", '"')) and len(ans) > 1:
        ans = ans[1:-1]
    return ans if ans else (default if default is not None else "")


def ask_choice(prompt, options, default=None):
    """options: list of (key, label). Returns the chosen key (as a string)."""
    print(prompt)
    for key, label in options:
        print(f"   {key}) {label}")
    valid = {str(k) for k, _ in options}
    while True:
        ans = ask("   Choice", default=str(default if default is not None else options[0][0]))
        if ans in valid:
            return ans
        print("   Please enter one of:", ", ".join(sorted(valid)))


def ask_yn(prompt, default="y"):
    return ask(prompt + " (y/n)", default=default).lower().strip() in ("y", "yes", "1", "true")


def ask_int(prompt, default):
    while True:
        ans = ask(prompt, default=str(default))
        try:
            return int(float(ans))
        except ValueError:
            print("   Please enter a whole number.")


def ask_float(prompt, default):
    while True:
        ans = ask(prompt, default=str(default))
        try:
            return float(ans)
        except ValueError:
            print("   Please enter a number.")


def ask_multi(prompt, choices):
    """Pick several items by number. Order of entry is kept (it matters for 'cascade')."""
    print(f"\n  {prompt}")
    for i, c in enumerate(choices, 1):
        print(f"    {i}. {c}")
    print("  Enter numbers separated by commas (e.g. 5,1 = nose first, then center) or 'all'.")
    print("  The order you type is the priority order used by 'cascade'.")
    while True:
        raw = ask("  Selection", default="").lower()
        if raw == "all":
            return list(choices)
        try:
            idx = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
            if idx and all(0 <= i < len(choices) for i in idx):
                seen, out = set(), []
                for i in idx:
                    if i not in seen:
                        seen.add(i)
                        out.append(choices[i])
                return out
        except ValueError:
            pass
        print(f"    Enter comma-separated numbers from 1 to {len(choices)}")


# =============================================================================
# File-name / key handling
# =============================================================================
def csv_key(path):
    """MVI_0618_roi1DLC_Hrnet...cleaned.csv -> MVI_0618_roi1"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "DLC" in stem:
        return stem.split("DLC")[0]
    m = re.match(r"(MVI_\d+_roi\w*)", stem, re.I)
    return m.group(1) if m else stem


def base_video(key):
    m = re.match(r"(MVI_\d+)", key, re.I)
    return m.group(1).lower() if m else key.lower()


def normalise_video_name(text):
    """'MVI_0618.MP4' / 'MVI_0618_roi1' / 'C:/x/MVI_0618.mp4' -> 'mvi_0618' / 'mvi_0618_roi1'"""
    name = os.path.basename(str(text).strip().replace("\\", "/"))
    name = re.sub(r"\.(mp4|avi|mov|mkv|csv)$", "", name, flags=re.I)
    if "DLC" in name:
        name = name.split("DLC")[0]
    return name.lower()


# =============================================================================
# ROIs and body parts, read from the DLC CSV itself
# =============================================================================
ROI_COL_RE = re.compile(r"^roi_(.+)_vertices$")


def _parse_vertices(val):
    """'[[x, y], ...]' (JSON or Python literal) -> list of (x, y) floats, or None if unusable."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        obj = json.loads(val) if isinstance(val, str) else val
    except (ValueError, TypeError):
        try:
            obj = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    try:
        pts = [(float(x), float(y)) for x, y in obj]
    except (TypeError, ValueError):
        return None
    return pts if len(pts) >= 3 else None


def _first_valid(series):
    for v in series:
        if isinstance(v, str) and v.strip():
            return v
    return None


def read_rois_from_csv(csv_path):
    """Return {roi_name: [(x, y), ...]} read from the CSV's vertex columns."""
    header = pd.read_csv(csv_path, nrows=0).columns
    per_roi_cols = {m.group(1): c for c in header if (m := ROI_COL_RE.match(c))}
    use = list(per_roi_cols.values()) + (["all_roi_vertices"] if "all_roi_vertices" in header else [])
    if not use:
        raise KeyError("CSV has no ROI vertex columns ('all_roi_vertices' or 'roi_<NAME>_vertices')")
    head = pd.read_csv(csv_path, usecols=use, nrows=500)  # vertices are constant, a few rows is enough

    rois = {}
    if "all_roi_vertices" in head.columns:
        raw = _first_valid(head["all_roi_vertices"])
        if raw:
            try:
                d = json.loads(raw)
            except ValueError:
                d = ast.literal_eval(raw)
            for name, verts in d.items():
                pts = _parse_vertices(verts)
                if pts:
                    rois[name] = pts
    for name, col in per_roi_cols.items():  # fill in / cross-check from the per-ROI columns
        if name not in rois:
            pts = _parse_vertices(_first_valid(head[col]))
            if pts:
                rois[name] = pts
    if not rois:
        raise ValueError("ROI vertex columns exist but contain no usable polygons")
    return rois


def detect_bodyparts(csv_path):
    header = pd.read_csv(csv_path, nrows=0).columns
    xs = {c[:-len("_x_final")] for c in header if c.endswith("_x_final")}
    ys = {c[:-len("_y_final")] for c in header if c.endswith("_y_final")}
    return sorted(xs & ys)


def configure_bodyparts(roi_names, avail_bps):
    """Same prompts as the ROI placement tool. Returns {roi_name: (condition, [bodyparts])}."""
    print("\n" + "-" * 24 + " Bodypart & Condition Setup " + "-" * 24)
    print(f"  ROIs found: {', '.join(roi_names)}")

    def one(label):
        bps = ask_multi(f"Select bodyparts for {label}", avail_bps)
        if len(bps) > 1:
            k = ask_choice("  Condition type", [(i + 1, d) for i, d in enumerate(CONDITION_DESCS)], default=1)
            cond = CONDITION_KEYS[int(k) - 1]
        else:
            cond = "single"
        return cond, bps

    if ask_yn("Use the same bodyparts & condition for all ROIs?", default="y"):
        rule = one("ALL ROIs")
        rules = {n: rule for n in roi_names}
    else:
        rules = {}
        for n in roi_names:
            print(f"\n  -- ROI: '{n}' --")
            rules[n] = one(f"'{n}'")
    print("\n  Rules that will be used:")
    for n, (cond, bps) in rules.items():
        print(f"    {n}: {cond}  ({' > '.join(bps) if cond == 'cascade' else ' + '.join(bps)})")
    return rules


# =============================================================================
# Start-frame sheet
# =============================================================================
VIDEO_COLUMN_NAMES = ("VIDEO FILE", "MOVIE", "VIDEO", "FILE", "FILENAME")
START_COLUMN_NAMES = ("START FRAME", "START")


def load_start_table(path):
    """Reads a sheet with a movie-name column and a start-frame column (any other columns are ignored)."""
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    cols = {re.sub(r"[\s_]+", " ", str(c)).strip().upper(): c for c in df.columns}
    vcol = next((cols[n] for n in VIDEO_COLUMN_NAMES if n in cols), None)
    scol = next((cols[n] for n in START_COLUMN_NAMES if n in cols), None)
    if vcol is None or scol is None:
        raise ValueError(f"Could not find a movie column {VIDEO_COLUMN_NAMES} and a start column "
                         f"{START_COLUMN_NAMES}; found {list(df.columns)}")
    names = df[vcol].map(normalise_video_name)
    names = names.map(lambda n: "mvi_" + n if re.match(r"\d", n) else n)  # '0618_roi1' -> 'mvi_0618_roi1'
    out = pd.DataFrame({"video": names, "start": pd.to_numeric(df[scol], errors="coerce")})
    return out.dropna(subset=["start"]).reset_index(drop=True)


def choose_start(key, table, fallback_mode, total_frames):
    """Return (start_frame, use_whole_csv, note) or None to skip."""
    if table is not None:
        hits = table[table["video"] == key.lower()]
        if hits.empty:
            hits = table[table["video"] == base_video(key)]
        values = sorted({int(v) for v in hits["start"]})
        if len(values) == 1:
            return values[0], False, ""
        if len(values) > 1:
            print(f"\n  ! Several different start frames listed for {key}: {values}")
            options = [(i + 1, f"start frame {v}") for i, v in enumerate(values)] + [
                ("m", "enter manually"), ("w", "analyse the whole CSV"), ("s", "skip this file")]
            ch = ask_choice("  Which one?", options)
            if ch.isdigit():
                return values[int(ch) - 1], False, "FLAG: several start frames listed, user chose one"
        else:
            print(f"\n  ! No start frame found in the sheet for {key}.")
            ch = ask_choice("  What now?", [("m", "enter start frame manually"),
                                             ("w", "analyse the whole CSV"), ("s", "skip this file")])
        if ch == "s":
            return None
        if ch == "w":
            return 0, True, "FLAG: no start frame in sheet, whole CSV analysed"
        return ask_int(f"  Start frame for {key} (CSV has {total_frames} frames)", 0), False, \
            "FLAG: start frame entered manually"
    if fallback_mode == "w":
        return 0, True, "whole CSV analysed"
    return ask_int(f"  Start frame for {key} (CSV has {total_frames} frames)", 0), False, "start frame entered manually"


# =============================================================================
# Group (master) sheet:  key -> GHAL / GHFD / SHAL / SHFD ...
# =============================================================================
GROUP_KEY_COLUMNS = ("KEY", "VIDEO FILE", "MOVIE", "VIDEO", "FILE", "FILENAME")
GROUP_PART_COLUMNS = ("HOUSING", "FOOD STATE")   # joined into the group name, e.g. GH + AL -> GHAL
GROUP_COLUMN = "GROUP"                           # if the sheet has a ready-made group column, it wins


def load_group_table(path):
    """Returns {normalised key (e.g. 'mvi_0618_roi1'): group name (e.g. 'GHAL')}."""
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    cols = {re.sub(r"[\s_]+", " ", str(c)).strip().upper(): c for c in df.columns}
    kcol = next((cols[n] for n in GROUP_KEY_COLUMNS if n in cols), None)
    if kcol is None:
        raise ValueError(f"Could not find a key column {GROUP_KEY_COLUMNS}; found {list(df.columns)}")
    if GROUP_COLUMN in cols:
        groups = df[cols[GROUP_COLUMN]].astype(str).str.strip()
    elif all(n in cols for n in GROUP_PART_COLUMNS):
        groups = df[[cols[n] for n in GROUP_PART_COLUMNS]].astype(str).apply(
            lambda r: "".join(v.strip() for v in r), axis=1)
    else:
        raise ValueError(f"Need a 'group' column or both {GROUP_PART_COLUMNS} columns; found {list(df.columns)}")
    keys = df[kcol].map(normalise_video_name)
    keys = keys.map(lambda n: "mvi_" + n if re.match(r"\d", n) else n)
    return {k: g for k, g in zip(keys, groups) if g and g.lower() != "nan"}


def find_group(key, group_map):
    """Exact key match first (roi1 and roi2 of one video can be in different groups);
    falls back to the video number only if every ROI of that video is in the same group."""
    k = key.lower()
    if k in group_map:
        return group_map[k]
    base = base_video(key)
    found = {g for kk, g in group_map.items() if base_video(kk) == base}
    return found.pop() if len(found) == 1 else None


def safe_folder(name):
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_") or "UNASSIGNED"


# =============================================================================
# Core analysis
# =============================================================================
def region_mask(df, vertices, cond, parts):
    """Boolean mask: which frames count as 'inside' this polygon under the chosen condition."""
    path = Path(np.asarray(vertices, dtype=float))
    n = len(df)

    def pts(bp):
        return df[[f"{bp}_x_final", f"{bp}_y_final"]].to_numpy(dtype=float)

    if cond == "single":
        return path.contains_points(pts(parts[0]))  # NaN coordinates -> False
    if cond == "any":
        return np.any([path.contains_points(pts(bp)) for bp in parts], axis=0)
    if cond == "all":
        return np.all([path.contains_points(pts(bp)) for bp in parts], axis=0)
    if cond == "cascade":
        result = np.zeros(n, dtype=bool)
        decided = np.zeros(n, dtype=bool)
        for bp in parts:
            p = pts(bp)
            use = np.isfinite(p).all(axis=1) & ~decided
            if use.any():
                result[use] = path.contains_points(p[use])
            decided |= use
        return result
    raise ValueError(f"unknown condition '{cond}'")


def frame_states(df, rois, rules):
    """One state label per frame (object array)."""
    inside = {}
    for name, verts in rois.items():
        if name not in rules:
            raise KeyError(f"no bodypart/condition rule set for ROI '{name}'")
        cond, parts = rules[name]
        inside[name] = region_mask(df, verts, cond, parts)
    order = [r for r in inside if r not in PRIORITY_LOW_TO_HIGH] + [r for r in PRIORITY_LOW_TO_HIGH if r in inside]
    labels = np.full(len(df), NEUTRAL, dtype=object)
    for name in order:
        labels[inside[name]] = name
    return labels


def run_length(labels):
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append({"label": labels[start], "start": start, "end": i - 1})
            start = i
    return runs


def grace_merge(runs, grace):
    """Absorb any run of <= grace frames that sits between two runs with the same label."""
    runs = [dict(r) for r in runs]
    i = 1
    while i < len(runs) - 1:
        cur = runs[i]
        if cur["end"] - cur["start"] + 1 <= grace and runs[i - 1]["label"] == runs[i + 1]["label"]:
            merged = {"label": runs[i - 1]["label"], "start": runs[i - 1]["start"], "end": runs[i + 1]["end"]}
            runs[i - 1:i + 2] = [merged]
            i = max(1, i - 1)
        else:
            i += 1
    return runs


def build_bout_table(runs, raw, gap, win_start, total_frames, fps):
    rows = []
    n_win = len(raw)
    for k, r in enumerate(runs):
        s, e, lab = r["start"], r["end"], r["label"]
        n = e - s + 1
        gap_frac = float(gap[s:e + 1].mean())
        rows.append({
            "bout": k + 1,
            "roi": lab,
            "start_frame": win_start + s,
            "end_frame": win_start + e,
            "duration_frames": n,
            "duration_s": round(n / fps, 3),
            "start_time_s": round((win_start + s) / fps, 3),
            "end_time_s": round((win_start + e + 1) / fps, 3),
            "from_roi": runs[k - 1]["label"] if k > 0 else "START",
            "to_roi": runs[k + 1]["label"] if k < len(runs) - 1 else "END",
            "pct_gap_frames": round(100 * gap_frac, 1),
            "low_tracking_quality": gap_frac > GAP_FRACTION_THRESHOLD,
            "grace_absorbed_frames": int((raw[s:e + 1] != lab).sum()),
            "truncated_at_window_start": bool(s == 0 and win_start > 0),
            "truncated_at_window_end": bool(e == n_win - 1 and win_start + n_win < total_frames),
        })
    return pd.DataFrame(rows)


def draw_ethogram(bouts_detail, bouts_collapsed, win_start, n_win, fps, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 3.9), sharex=True)
    for ax, table, name in ((axes[0], bouts_detail, "SS / FS / SC / FC / Neutral"),
                            (axes[1], bouts_collapsed, "SC / FC / Neutral")):
        totals = table.groupby("roi")["duration_frames"].sum()
        for lab, grp in table.groupby("roi"):
            spans = [((r.start_frame - win_start) / fps, r.duration_frames / fps) for r in grp.itertuples()]
            ax.broken_barh(spans, (0, 1), facecolors=COLORS.get(lab, "#888888"),
                           edgecolor="none", linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        handles = [Patch(facecolor=COLORS.get(lab, "#888888"),
                         label=f"{lab}: {frames / fps:.1f} s ({100 * frames / n_win:.1f}%)")
                   for lab, frames in totals.items()]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.05), fontsize=8, frameon=False)
    axes[1].set_xlim(0, n_win / fps)
    axes[1].set_xlabel("Time from start of analysed window (s)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 0.95 if title else 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_one(csv_path, key, rules, fps, start, whole, out_dir, group=None):
    rois = read_rois_from_csv(csv_path)  # per-video arena, read from this CSV
    used_parts = {bp for name in rois if name in rules for bp in rules[name][1]}
    coord_cols = [f"{bp}_{ax}_final" for bp in used_parts for ax in ("x", "y")]
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in coord_cols if c not in header]
    if missing:
        raise KeyError(f"CSV lacks required columns: {missing[:4]}")
    qual_cols = [f"{bp}_quality" for bp in QUALITY_BODYPARTS if f"{bp}_quality" in header]
    df = pd.read_csv(csv_path, usecols=list(dict.fromkeys(coord_cols + qual_cols)))
    total = len(df)

    notes = []
    if start >= total:
        raise ValueError(f"start frame {start} is beyond the end of the CSV ({total} frames)")
    end = total if whole else min(start + WINDOW_FRAMES, total)
    if not whole and end - start < WINDOW_FRAMES:
        notes.append(f"FLAG: only {end - start} frames available after start frame (wanted {WINDOW_FRAMES})")
    win = df.iloc[start:end].reset_index(drop=True)
    n_win = len(win)

    raw = frame_states(win, rois, rules)
    if qual_cols:
        gap = np.any([win[c].isin(GAP_QUALITY_LABELS).to_numpy() for c in qual_cols], axis=0)
    else:
        gap = np.zeros(n_win, dtype=bool)
        notes.append("FLAG: no quality columns found, gap fraction not computed")

    detail_runs = grace_merge(run_length(raw), GRACE_FRAMES)
    detail = build_bout_table(detail_runs, raw, gap, start, total, fps)

    raw_c = np.array([COLLAPSE_MAP.get(x, x) for x in raw], dtype=object)
    coll_runs = grace_merge(run_length(raw_c), GRACE_FRAMES)
    coll = build_bout_table(coll_runs, raw_c, gap, start, total, fps)

    detail.to_csv(os.path.join(out_dir, f"{key}_bouts_detailed.csv"), index=False)
    coll.to_csv(os.path.join(out_dir, f"{key}_bouts_collapsed_SC_FC_Neutral.csv"), index=False)
    draw_ethogram(detail, coll, start, n_win, fps,
                  group,  # title = group only (None -> no title); file name still carries the movie key
                  os.path.join(out_dir, f"{key}_ethogram.png"))

    totals = detail.groupby("roi")["duration_frames"].sum()
    summary = ", ".join(f"{lab} {fr / fps:.1f}s" for lab, fr in totals.items())
    n_low = int(detail["low_tracking_quality"].sum())
    print(f"    ROIs from CSV: {', '.join(rois)}")
    print(f"    {len(detail)} bouts ({n_low} flagged low tracking quality) | {summary}")
    return {"start_frame": start, "end_frame": end - 1, "frames_analysed": n_win,
            "n_bouts_detailed": len(detail), "n_bouts_collapsed": len(coll),
            "n_low_quality_bouts": n_low}, notes


# =============================================================================
# Main
# =============================================================================
def main():
    global GRACE_FRAMES, GAP_FRACTION_THRESHOLD, WINDOW_FRAMES, FPS
    print("=" * 70)
    print(" ROI bout / transition analysis")
    print("=" * 70)

    while True:
        in_dir = ask("Input folder with the DLC CSV files")
        if os.path.isdir(in_dir):
            break
        print("  That folder does not exist.")
    csvs = sorted(p for p in glob.glob(os.path.join(in_dir, "*.csv")) if not os.path.basename(p).startswith("._"))
    if not csvs:
        sys.exit("No CSV files found in that folder.")
    print(f"  Found {len(csvs)} CSV file(s).")

    out_dir = ask("Output folder", default=os.path.join(in_dir, "roi_bout_output"))
    os.makedirs(out_dir, exist_ok=True)

    FPS = ask_float("Frames per second (FPS)", FPS)
    print(f"\nCurrent parameters: grace = {GRACE_FRAMES} frames, low-quality threshold = "
          f"{GAP_FRACTION_THRESHOLD:.0%} gap frames, window = {WINDOW_FRAMES} frames")
    if ask("Change any of these? (y/N)", default="n").lower().startswith("y"):
        GRACE_FRAMES = ask_int("  Grace frames", GRACE_FRAMES)
        GAP_FRACTION_THRESHOLD = ask_float("  Gap-fraction threshold (0-1)", GAP_FRACTION_THRESHOLD)
        WINDOW_FRAMES = ask_int("  Frames to analyse after the start frame", WINDOW_FRAMES)

    # ---- ROIs + body parts, taken from the first CSV --------------------------------
    print(f"\nReading ROIs and body parts from {os.path.basename(csvs[0])} ...")
    try:
        sample_rois = read_rois_from_csv(csvs[0])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not read ROIs from the first CSV: {exc}")
    avail_bps = detect_bodyparts(csvs[0])
    if not avail_bps:
        sys.exit("No '<bodypart>_x_final' / '<bodypart>_y_final' columns found in the first CSV.")
    print(f"  ROIs: {', '.join(sample_rois)}")
    print(f"  Bodyparts: {', '.join(avail_bps)}")
    rules = configure_bodyparts(list(sample_rois), avail_bps)

    # ---- start frames ---------------------------------------------------------------
    table, fallback_mode = None, None
    sf_path = ask("\nPath to start-frame file (.xlsx/.csv); "
                  "press Enter if you don't have one", default="")
    if sf_path:
        try:
            table = load_start_table(sf_path)
            print(f"  Read {len(table)} start-frame rows.")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Could not use that file: {exc}")
    if table is None:
        fallback_mode = ask_choice("No start-frame table available. How should start frames be set?",
                                   [("m", "enter a start frame manually for each video"),
                                    ("w", "analyse the whole CSV for every video")])

    group_map = None
    gp = ask("\nPath to group/master file (.xlsx/.csv) to sort outputs into one folder per group; "
             "press Enter to skip", default="")
    if gp:
        try:
            group_map = load_group_table(gp)
            counts = pd.Series(list(group_map.values())).value_counts()
            print(f"  Read {len(group_map)} keys in groups: " + ", ".join(f"{g} ({n})" for g, n in counts.items()))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Could not use that file, outputs will not be grouped: {exc}")

    rule_text = "; ".join(f"{n}={c}:{'>'.join(b)}" for n, (c, b) in rules.items())
    report = []
    for i, csv_path in enumerate(csvs, 1):
        key = csv_key(csv_path)
        print(f"\n[{i}/{len(csvs)}] {os.path.basename(csv_path)}  (key: {key})")
        row = {"csv": os.path.basename(csv_path), "video_key": key, "group": "", "status": "",
               "rules": rule_text, "notes": ""}
        notes = []
        try:
            group, key_out_dir = None, out_dir
            if group_map is not None:
                group = find_group(key, group_map)
                if group is None:
                    notes.append("FLAG: key not found in group file, saved in UNASSIGNED")
                key_out_dir = os.path.join(out_dir, safe_folder(group) if group else "UNASSIGNED")
                os.makedirs(key_out_dir, exist_ok=True)
                row["group"] = group or "UNASSIGNED"
            n_frames = sum(1 for _ in open(csv_path, "rb")) - 1
            choice = choose_start(key, table, fallback_mode, n_frames)
            if choice is None:
                notes.append("skipped by user (no start frame)")
                row.update(status="SKIPPED", notes="; ".join(notes))
                report.append(row)
                continue
            start, whole, start_note = choice
            if start_note:
                notes.append(start_note)

            info, more_notes = process_one(csv_path, key, rules, FPS, start, whole, key_out_dir, group)
            row.update(info)
            row.update(status="PROCESSED")
            notes += more_notes
        except Exception as exc:  # noqa: BLE001
            row.update(status="FAILED")
            notes.append(f"{type(exc).__name__}: {exc}")
        row["notes"] = "; ".join(notes)
        for n in notes:
            print("   ", n)
        report.append(row)

    rep = pd.DataFrame(report)
    rep_path = os.path.join(out_dir, "run_report.csv")
    rep.to_csv(rep_path, index=False)
    print("\n" + "=" * 70)
    print(rep["status"].value_counts().to_string())
    print(f"Outputs and run_report.csv are in: {out_dir}")


if __name__ == "__main__":
    main()
