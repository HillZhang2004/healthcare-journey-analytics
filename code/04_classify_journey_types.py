"""
04_classify_journey_types.py
-----------------------------
Classify patient journeys into descriptive type indicators and one
mutually exclusive primary_journey_type category.

Input:
    outputs/tables/journeys_clean_diagnosisvalue.parquet

Outputs:
    outputs/audit/journey_type_counts.csv
    outputs/audit/primary_journey_type_counts.csv
    outputs/audit/journey_type_metric_summary.csv
    outputs/audit/journey_type_by_groupname_top.csv
    outputs/audit/journey_type_validation_notes.txt

    outputs/tables/journeys_classified.parquet   ← journey table with type columns added

Run from project root:
    python code/04_classify_journey_types.py

No raw patient-level or journey-level rows are printed or exported.
All printed output is aggregate only.

INTERPRETATION CAUTION
These categories are descriptive, not clinical quality labels.
One-visit journeys are not inherently bad. Acute, screening, procedural,
newborn, obstetric, and administrative episodes may naturally produce
only one observed encounter. Use language such as 'associated with',
'observed among', or 'linked to' — never causal framing.
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

TABLES_DIR     = Path("outputs/tables")
AUDIT_DIR      = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True,  exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

JOURNEY_PARQUET     = TABLES_DIR / "journeys_clean_diagnosisvalue.parquet"
CLASSIFIED_PARQUET  = TABLES_DIR / "journeys_classified.parquet"

# Priority order for the single mutually exclusive label (first match wins).
PRIMARY_PRIORITY = [
    ("one_visit",          "one_visit_journey"),
    ("hospital_involved",  "hospital_involved_journey"),
    ("long_journey",       "long_journey"),
    ("lab_heavy",          "lab_heavy_journey"),
    ("imaging_heavy",      "imaging_heavy_journey"),
    ("multi_department",   "multi_department_journey"),
    ("short_followup",     "short_followup_journey"),
]
PRIMARY_DEFAULT = "other_multi_encounter"

# Human-readable descriptions for each indicator (used in output files).
INDICATOR_DESCRIPTIONS = {
    "one_visit_journey":          "n_encounters == 1",
    "short_followup_journey":     "n_encounters >= 2 AND journey_duration_days <= 30",
    "long_journey":               "n_encounters >= 2 AND journey_duration_days >= 365",
    "lab_heavy_journey":          "(n_lab_encounters >= 3 OR lab_share >= 0.5) AND n_encounters >= 2",
    "imaging_heavy_journey":      "(n_imaging_encounters >= 2 OR imaging_share >= 0.5) AND n_encounters >= 2",
    "hospital_involved_journey":  "has_ed OR has_hospital_admission OR has_inpatient_admission OR has_observation",
    "multi_department_journey":   "n_unique_departments >= 3",
    "mixed_setting_journey":      "n_unique_types >= 3 OR n_unique_departments >= 3",
}


# ---------------------------------------------------------------------------
# 1. Load journey table
# ---------------------------------------------------------------------------

def load_journeys(path: Path) -> pd.DataFrame:
    jdf = pd.read_parquet(path)
    print(f"  Loaded: {path}")
    print(f"  Rows (journeys): {len(jdf):,}")
    print(f"  Columns        : {list(jdf.columns)}")
    return jdf


# ---------------------------------------------------------------------------
# 2. Classify indicators
# ---------------------------------------------------------------------------

def classify_indicators(jdf: pd.DataFrame) -> pd.DataFrame:
    """
    Add eight boolean indicator columns to the journey table.
    All logic operates on existing numeric / boolean columns only.
    """
    jdf = jdf.copy()
    n   = len(jdf)

    enc = jdf["n_encounters"]
    dur = jdf["journey_duration_days"]

    # --- 1. one_visit_journey ---
    jdf["one_visit_journey"] = enc == 1

    # --- 2. short_followup_journey ---
    jdf["short_followup_journey"] = (enc >= 2) & (dur <= 30)

    # --- 3. long_journey ---
    jdf["long_journey"] = (enc >= 2) & (dur >= 365)

    # --- 4. lab_heavy_journey ---
    # Ratio guard: enc >= 2 prevents division by zero and limits to multi-enc.
    lab     = jdf["n_lab_encounters"]
    multi   = enc >= 2
    lab_rat = lab / enc.where(enc > 0, other=1)   # safe division
    jdf["lab_heavy_journey"] = multi & ((lab >= 3) | (lab_rat >= 0.5))

    # --- 5. imaging_heavy_journey ---
    img     = jdf["n_imaging_encounters"]
    img_rat = img / enc.where(enc > 0, other=1)
    jdf["imaging_heavy_journey"] = multi & ((img >= 2) | (img_rat >= 0.5))

    # --- 6. hospital_involved_journey ---
    hosp_cols = ["has_ed", "has_hospital_admission",
                 "has_inpatient_admission", "has_observation"]
    present   = [c for c in hosp_cols if c in jdf.columns]
    if present:
        jdf["hospital_involved_journey"] = jdf[present].any(axis=1)
    else:
        jdf["hospital_involved_journey"] = False
        print("  WARNING: no hospital flag columns found; hospital_involved_journey = False.")

    # --- 7. multi_department_journey ---
    jdf["multi_department_journey"] = jdf["n_unique_departments"] >= 3

    # --- 8. mixed_setting_journey ---
    if "n_unique_types" in jdf.columns:
        jdf["mixed_setting_journey"] = (jdf["n_unique_types"] >= 3) | (jdf["n_unique_departments"] >= 3)
    else:
        jdf["mixed_setting_journey"] = jdf["n_unique_departments"] >= 3
        print("  NOTE: n_unique_types not found; mixed_setting_journey uses n_unique_departments only.")

    return jdf


# ---------------------------------------------------------------------------
# 3. Assign primary_journey_type (mutually exclusive, priority order)
# ---------------------------------------------------------------------------

def assign_primary_type(jdf: pd.DataFrame) -> pd.DataFrame:
    """
    Assign one label per journey using the priority list.
    The first condition that is True for a row wins.
    """
    conditions = [jdf[ind_col] for _, ind_col in PRIMARY_PRIORITY]
    choices    = [label        for label, _    in PRIMARY_PRIORITY]
    jdf["primary_journey_type"] = np.select(conditions, choices, default=PRIMARY_DEFAULT)
    return jdf


# ---------------------------------------------------------------------------
# 4. Aggregate outputs
# ---------------------------------------------------------------------------

def indicator_counts(jdf: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage for each of the 8 binary indicators."""
    n = len(jdf)
    rows = []
    for ind, desc in INDICATOR_DESCRIPTIONS.items():
        if ind not in jdf.columns:
            continue
        n_true = int(jdf[ind].sum())
        rows.append({
            "indicator":    ind,
            "description":  desc,
            "n_journeys":   n_true,
            "pct_journeys": round(n_true / n * 100, 2) if n else 0.0,
        })
    return pd.DataFrame(rows)


def primary_type_counts(jdf: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage for each primary_journey_type value."""
    n = len(jdf)
    # Preserve the declared priority order in the output.
    type_order = [l for l, _ in PRIMARY_PRIORITY] + [PRIMARY_DEFAULT]
    vc = jdf["primary_journey_type"].value_counts()
    rows = []
    for t in type_order:
        cnt = int(vc.get(t, 0))
        rows.append({
            "primary_journey_type": t,
            "n_journeys":           cnt,
            "pct_journeys":         round(cnt / n * 100, 2) if n else 0.0,
        })
    return pd.DataFrame(rows)


def metric_summary_by_type(jdf: pd.DataFrame) -> pd.DataFrame:
    """
    Per primary_journey_type:
      n_journeys, pct_journeys, median/mean n_encounters,
      pct multi-encounter, median/mean duration, median first-to-second gap.
    """
    n_total = len(jdf)
    rows = []

    for ptype, grp in jdf.groupby("primary_journey_type", sort=False):
        n_grp    = len(grp)
        enc      = grp["n_encounters"]
        dur      = grp["journey_duration_days"]
        gap_col  = "first_to_second_gap_days"
        gap      = grp[gap_col].dropna() if gap_col in grp.columns else pd.Series(dtype=float)
        n_follow = int(grp["has_follow_up"].sum()) if "has_follow_up" in grp.columns else 0

        rows.append({
            "primary_journey_type":          ptype,
            "n_journeys":                    n_grp,
            "pct_journeys":                  round(n_grp / n_total * 100, 2),
            "median_n_encounters":           round(enc.median(), 1),
            "mean_n_encounters":             round(enc.mean(), 2),
            "p90_n_encounters":              int(enc.quantile(0.90)),
            "pct_has_follow_up":             round(n_follow / n_grp * 100, 2) if n_grp else 0.0,
            "median_journey_duration_days":  int(dur.median()),
            "mean_journey_duration_days":    round(dur.mean(), 1),
            "p90_journey_duration_days":     int(dur.quantile(0.90)),
            "median_first_to_second_gap_days": round(gap.median(), 1) if len(gap) else np.nan,
        })

    df = pd.DataFrame(rows)
    # Sort by the declared priority order for readability.
    type_order = [l for l, _ in PRIMARY_PRIORITY] + [PRIMARY_DEFAULT]
    df["_sort"] = df["primary_journey_type"].map({t: i for i, t in enumerate(type_order)})
    df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return df


def top_groups_by_type(jdf: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    For each primary_journey_type, the top_n GroupName values by journey count.
    Returns a tidy DataFrame; journeys with null GroupName are labeled '(unlabeled)'.
    """
    if "GroupName" not in jdf.columns:
        print("  NOTE: GroupName not in journey table; skipping by-groupname output.")
        return pd.DataFrame()

    work = jdf[["primary_journey_type", "GroupName"]].copy()
    work["GroupName"] = work["GroupName"].fillna("(unlabeled)").astype(str)

    rows = []
    for ptype, grp in work.groupby("primary_journey_type", sort=False):
        n_type = len(grp)
        top    = grp["GroupName"].value_counts().head(top_n).reset_index()
        top.columns = ["GroupName", "n_journeys"]
        top["primary_journey_type"] = ptype
        top["pct_of_type_journeys"] = (top["n_journeys"] / n_type * 100).round(2)
        rows.append(top)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    # Sort by type priority, then count descending within type.
    type_order = {t: i for i, t in enumerate([l for l, _ in PRIMARY_PRIORITY] + [PRIMARY_DEFAULT])}
    result["_sort"] = result["primary_journey_type"].map(type_order)
    result = (
        result.sort_values(["_sort", "n_journeys"], ascending=[True, False])
              .drop(columns="_sort")
              .reset_index(drop=True)
    )
    return result[["primary_journey_type", "GroupName", "n_journeys", "pct_of_type_journeys"]]


def indicator_overlap_matrix(jdf: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair of indicators, count journeys where both are True.
    Saved inside the validation notes — not a standalone CSV.
    """
    ind_cols = list(INDICATOR_DESCRIPTIONS.keys())
    present  = [c for c in ind_cols if c in jdf.columns]
    n = len(jdf)
    rows = []
    for i, a in enumerate(present):
        for b in present[i:]:
            both = int((jdf[a] & jdf[b]).sum())
            rows.append({
                "indicator_A": a,
                "indicator_B": b,
                "n_both_true":    both,
                "pct_of_total":   round(both / n * 100, 2),
            })
    return pd.DataFrame(rows)


def build_validation_notes(
    jdf:        pd.DataFrame,
    ind_counts: pd.DataFrame,
    ptype_counts: pd.DataFrame,
    overlap:    pd.DataFrame,
) -> str:
    """Produce the text content for journey_type_validation_notes.txt."""
    n = len(jdf)
    sep = "=" * 64
    lines = [
        sep,
        "  JOURNEY TYPE CLASSIFICATION — VALIDATION NOTES",
        f"  Generated by: code/04_classify_journey_types.py",
        sep,
        "",
        f"  Journey table rows : {n:,}",
        f"  Columns            : {len(jdf.columns)}",
        "",
        "  INDICATOR DEFINITIONS",
        "  " + "-"*60,
    ]
    for ind, desc in INDICATOR_DESCRIPTIONS.items():
        lines.append(f"  {ind:<32}: {desc}")

    lines += [
        "",
        "  INDICATOR COUNTS",
        "  " + "-"*60,
    ]
    for _, row in ind_counts.iterrows():
        lines.append(
            f"  {row['indicator']:<32}: {row['n_journeys']:>10,}  ({row['pct_journeys']:>5.1f}%)"
        )

    lines += [
        "",
        "  PRIMARY JOURNEY TYPE — PRIORITY ORDER (first match wins)",
        "  " + "-"*60,
    ]
    for _, row in ptype_counts.iterrows():
        lines.append(
            f"  {row['primary_journey_type']:<26}: {row['n_journeys']:>10,}  ({row['pct_journeys']:>5.1f}%)"
        )

    lines += [
        "",
        "  INDICATOR OVERLAP (journeys where both indicators are True)",
        "  " + "-"*60,
    ]
    off_diag = overlap[overlap["indicator_A"] != overlap["indicator_B"]]
    notable  = off_diag[off_diag["pct_of_total"] >= 1.0].sort_values("pct_of_total", ascending=False)
    if notable.empty:
        lines.append("  No indicator pair with >= 1% co-occurrence of total journeys.")
    else:
        for _, row in notable.iterrows():
            lines.append(
                f"  {row['indicator_A']} ∩ {row['indicator_B']}: "
                f"{row['n_both_true']:,}  ({row['pct_of_total']}% of all journeys)"
            )

    lines += [
        "",
        sep,
        "  INTERPRETATION CAUTIONS",
        sep,
        "",
        textwrap.fill(
            "These journey-type categories are descriptive and structural, "
            "not clinical quality assessments. They describe the observable "
            "pattern of encounters linked to a given diagnosis, not whether "
            "care was appropriate or complete.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        textwrap.fill(
            "One-visit journeys are NOT inherently problematic. Acute illness, "
            "screening, preventive care, newborn, obstetric, procedural, and "
            "administrative episodes may naturally produce only one observed "
            "encounter within the data window.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        textwrap.fill(
            "Left/right censoring: journeys beginning before Jan 2022 or "
            "extending past Dec 2025 will appear shorter or less complex "
            "than they truly are. This affects duration and encounter-count "
            "metrics, particularly for 'long_journey' and 'multi_department' "
            "categories.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        textwrap.fill(
            "The hospital_involved_journey indicator captures any encounter "
            "flagged as ED, hospital admission, inpatient admission, or "
            "observation. A single ED visit within an otherwise outpatient "
            "journey will cause it to be classified as hospital_involved. "
            "Interpret counts accordingly.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        textwrap.fill(
            "Lab and imaging classification is based on keyword matching "
            "against Type and VisitTypeDescription fields. Coverage may be "
            "incomplete; encounter type labeling in the source data may vary "
            "by department or provider. Treat these counts as approximations.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        textwrap.fill(
            "Use careful language in all downstream analysis: "
            "'associated with', 'observed among', 'linked to', 'suggests' — "
            "not 'causes', 'leads to', or 'results in'.",
            width=64, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(
    jdf:          pd.DataFrame,
    ind_counts:   pd.DataFrame,
    ptype_counts: pd.DataFrame,
    metric_df:    pd.DataFrame,
) -> None:
    sep = "=" * 64
    n   = len(jdf)

    print(f"\n{sep}")
    print("  JOURNEY TYPE CLASSIFICATION — AGGREGATE SUMMARY")
    print(sep)

    print(f"\n  Total journeys classified: {n:,}")

    print(f"\n  INDICATOR COUNTS (overlapping — one journey may match multiple)")
    print(f"  {'Indicator':<34} {'N':>10}  {'%':>6}")
    print(f"  {'-'*54}")
    for _, row in ind_counts.iterrows():
        print(f"  {row['indicator']:<34} {row['n_journeys']:>10,}  {row['pct_journeys']:>5.1f}%")

    print(f"\n  PRIMARY JOURNEY TYPE (mutually exclusive, priority order)")
    print(f"  {'Type':<28} {'N':>10}  {'%':>6}")
    print(f"  {'-'*48}")
    for _, row in ptype_counts.iterrows():
        print(f"  {row['primary_journey_type']:<28} {row['n_journeys']:>10,}  {row['pct_journeys']:>5.1f}%")

    print(f"\n  MEDIAN ENCOUNTERS AND DURATION BY PRIMARY TYPE")
    print(f"  {'Type':<28} {'Med Enc':>8}  {'Med Dur(d)':>10}  {'Med 1st→2nd Gap':>16}")
    print(f"  {'-'*68}")
    for _, row in metric_df.iterrows():
        gap_str = f"{row['median_first_to_second_gap_days']:.0f}d" \
                  if pd.notna(row["median_first_to_second_gap_days"]) else "   —"
        print(
            f"  {row['primary_journey_type']:<28} "
            f"{row['median_n_encounters']:>8.1f}  "
            f"{row['median_journey_duration_days']:>10}  "
            f"{gap_str:>16}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 64)
    print("  CLASSIFY JOURNEY TYPES — DataFest 2026")
    print("=" * 64)

    # Load
    print("\n  Loading journey table …")
    jdf = load_journeys(JOURNEY_PARQUET)

    # Classify
    print("\n  Classifying journey-type indicators …")
    jdf = classify_indicators(jdf)
    print("  Indicators added: " + ", ".join(INDICATOR_DESCRIPTIONS.keys()))

    print("  Assigning primary_journey_type …")
    jdf = assign_primary_type(jdf)

    # Aggregate outputs
    print("\n  Computing aggregate outputs …")
    ind_counts_df   = indicator_counts(jdf)
    ptype_counts_df = primary_type_counts(jdf)
    metric_df       = metric_summary_by_type(jdf)
    top_groups_df   = top_groups_by_type(jdf, top_n=10)
    overlap_df      = indicator_overlap_matrix(jdf)

    # Save CSVs
    ind_counts_df.to_csv(  AUDIT_DIR / "journey_type_counts.csv",             index=False)
    ptype_counts_df.to_csv(AUDIT_DIR / "primary_journey_type_counts.csv",     index=False)
    metric_df.to_csv(      AUDIT_DIR / "journey_type_metric_summary.csv",     index=False)
    if not top_groups_df.empty:
        top_groups_df.to_csv(AUDIT_DIR / "journey_type_by_groupname_top.csv", index=False)

    # Validation notes text file
    notes = build_validation_notes(jdf, ind_counts_df, ptype_counts_df, overlap_df)
    with open(AUDIT_DIR / "journey_type_validation_notes.txt", "w") as fh:
        fh.write(notes)

    for fname in [
        "journey_type_counts.csv",
        "primary_journey_type_counts.csv",
        "journey_type_metric_summary.csv",
        "journey_type_by_groupname_top.csv",
        "journey_type_validation_notes.txt",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    # Save classified journey table
    jdf.to_parquet(CLASSIFIED_PARQUET, index=False)
    print(f"  Saved: {CLASSIFIED_PARQUET}")

    # Print aggregate summary
    print_summary(jdf, ind_counts_df, ptype_counts_df, metric_df)

    print(f"""
{'='*64}
  WARNING — DESCRIPTIVE CLASSIFICATION ONLY
{'='*64}

These journey-type labels describe observable care structure.
They are not clinical quality assessments.

One-visit journeys are not automatically incomplete or poor care.
Acute, screening, procedural, newborn, obstetric, and administrative
episodes may naturally produce one observed encounter.

All downstream analysis should use careful language:
  "associated with" / "observed among" / "linked to" / "suggests"
  — not "causes", "leads to", or "results in".

SDOH comparisons should be framed as patterns among patients with
recorded responses, not conclusions about the full population.
{'='*64}
""")


if __name__ == "__main__":
    main()
