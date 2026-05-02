"""
08_screened_only_sdoh_analysis.py
-----------------------------------
Compare journey types and metrics only among journeys linked to patients
who have at least one clean SDOH screening response.

The full journey table mixes two very different populations:
  (a) Patients who were screened and had no positive core need.
  (b) Patients who were never screened (no clean SDOH data at all).
Keeping them in a single '0 needs' group conflates absence-of-need with
absence-of-data.  This script restricts to screened patients and uses
'screened_no_positive_need' to label the zero-need screened subgroup.

Input:
    outputs/tables/journeys_with_core_sdoh_need_flags.parquet

Outputs to outputs/audit/:
    screened_core_need_group_counts.csv
    screened_journey_type_by_core_need_group.csv
    screened_journey_type_by_any_core_need.csv
    screened_core_need_metric_summary.csv
    screened_specific_need_by_journey_type.csv
    screened_sdoh_analysis_notes.txt

Run from project root:
    python code/08_screened_only_sdoh_analysis.py

No raw patient-level or journey-level rows are printed or exported.
All printed output is aggregate only.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

IN_PARQUET = TABLES_DIR / "journeys_with_core_sdoh_need_flags.parquet"

CORE_NEED_FLAGS = [
    "transportation_need",
    "food_insecurity_need",
    "housing_instability_need",
    "financial_strain_need",
    "ipv_need",
    "utilities_need",
]

SCREENED_GROUP_ORDER = [
    "screened_no_positive_need",
    "1 need",
    "2-3 needs",
    "4+ needs",
]

PRIMARY_TYPE_ORDER = [
    "one_visit", "hospital_involved", "long_journey", "lab_heavy",
    "imaging_heavy", "multi_department", "short_followup", "other_multi_encounter",
]


# ---------------------------------------------------------------------------
# 1. Load, filter, and label
# ---------------------------------------------------------------------------

def load_screened(path: Path) -> pd.DataFrame:
    """
    Load the journey table, inspect SDOH screening column, and filter to
    journeys where the patient has at least one clean SDOH response.

    Adds 'screened_core_need_group' with the four screened-only labels.
    """
    jdf = pd.read_parquet(path)

    # Verify the screening indicator column.
    if "has_clean_sdoh_response" in jdf.columns:
        screen_col = "has_clean_sdoh_response"
    elif "has_any_sdoh_response" in jdf.columns:
        screen_col = "has_any_sdoh_response"
        print(f"  NOTE: 'has_clean_sdoh_response' not found; using '{screen_col}'.")
    else:
        raise KeyError(
            "Neither 'has_clean_sdoh_response' nor 'has_any_sdoh_response' "
            "found in parquet.  Cannot identify screened journeys."
        )

    n_total    = len(jdf)
    screened   = jdf[jdf[screen_col] == True].copy()
    n_screened = len(screened)
    n_dropped  = n_total - n_screened

    print(f"  Total journeys in file       : {n_total:,}")
    print(f"  Journeys with screening data : {n_screened:,}  "
          f"({round(n_screened/n_total*100, 1)}%)")
    print(f"  Journeys without (excluded)  : {n_dropped:,}  "
          f"({round(n_dropped/n_total*100, 1)}%)")

    # Build screened_core_need_group.
    screened["screened_core_need_group"] = pd.cut(
        screened["core_need_count"],
        bins=[-1, 0, 1, 3, len(CORE_NEED_FLAGS)],
        labels=SCREENED_GROUP_ORDER,
        right=True,
    ).astype(str)

    return screened


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_type(df: pd.DataFrame, col: str = "primary_journey_type") -> pd.DataFrame:
    order = {t: i for i, t in enumerate(PRIMARY_TYPE_ORDER)}
    df["_ts"] = df[col].map(order).fillna(99)
    return df.sort_values("_ts").drop(columns="_ts").reset_index(drop=True)


def _sort_group(df: pd.DataFrame, col: str = "screened_core_need_group") -> pd.DataFrame:
    order = {g: i for i, g in enumerate(SCREENED_GROUP_ORDER)}
    df["_gs"] = df[col].map(order).fillna(99)
    return df.sort_values("_gs").drop(columns="_gs").reset_index(drop=True)


def _pct(num, den, decimals: int = 2) -> float:
    return round(num / den * 100, decimals) if den else 0.0


# ---------------------------------------------------------------------------
# 2. Output builders
# ---------------------------------------------------------------------------

def save_group_counts(screened: pd.DataFrame) -> None:
    n = len(screened)
    vc = screened["screened_core_need_group"].value_counts()
    rows = []
    for g in SCREENED_GROUP_ORDER:
        cnt = int(vc.get(g, 0))
        rows.append({
            "screened_core_need_group": g,
            "n_journeys":              cnt,
            "pct_screened_journeys":   _pct(cnt, n),
        })
    _sort_group(pd.DataFrame(rows)).to_csv(
        AUDIT_DIR / "screened_core_need_group_counts.csv", index=False
    )


def save_type_by_need_group(screened: pd.DataFrame) -> None:
    """
    primary_journey_type × screened_core_need_group with three percentage columns:
      pct_within_type       : % of this journey type in this need group
      pct_within_need_group : % of this need group in this journey type
    """
    ct = (
        screened
        .groupby(["primary_journey_type", "screened_core_need_group"], dropna=False)
        .size()
        .reset_index(name="n_journeys")
    )
    type_totals  = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    group_totals = ct.groupby("screened_core_need_group")["n_journeys"].transform("sum")
    ct["pct_within_type"]       = (ct["n_journeys"] / type_totals  * 100).round(2)
    ct["pct_within_need_group"] = (ct["n_journeys"] / group_totals * 100).round(2)

    type_order  = {t: i for i, t in enumerate(PRIMARY_TYPE_ORDER)}
    group_order = {g: i for i, g in enumerate(SCREENED_GROUP_ORDER)}
    ct["_ts"] = ct["primary_journey_type"].map(type_order).fillna(99)
    ct["_gs"] = ct["screened_core_need_group"].map(group_order).fillna(99)
    (
        ct.sort_values(["_gs", "_ts"])
          .drop(columns=["_ts", "_gs"])
          .to_csv(AUDIT_DIR / "screened_journey_type_by_core_need_group.csv", index=False)
    )


def save_type_by_any_need(screened: pd.DataFrame) -> None:
    ct = (
        screened
        .groupby(["primary_journey_type", "has_any_core_need"], dropna=False)
        .size()
        .reset_index(name="n_journeys")
    )
    type_totals = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    ct["pct_within_type"] = (ct["n_journeys"] / type_totals * 100).round(2)
    _sort_type(ct).to_csv(
        AUDIT_DIR / "screened_journey_type_by_any_core_need.csv", index=False
    )


def save_metric_summary(screened: pd.DataFrame) -> None:
    gap_col = "first_to_second_gap_days"
    rows = []
    n_total = len(screened)

    for group in SCREENED_GROUP_ORDER:
        grp = screened[screened["screened_core_need_group"] == group]
        if grp.empty:
            continue
        n   = len(grp)
        enc = grp["n_encounters"]
        dur = grp["journey_duration_days"]
        gap = grp[gap_col].dropna() if gap_col in grp.columns else pd.Series(dtype=float)

        rows.append({
            "screened_core_need_group":        group,
            "n_journeys":                      n,
            "pct_of_screened_journeys":        _pct(n, n_total),
            "median_n_encounters":             round(float(enc.median()), 1),
            "mean_n_encounters":               round(float(enc.mean()),   2),
            "pct_multi_encounter":             _pct(int((enc > 1).sum()), n),
            "median_journey_duration_days":    int(dur.median()),
            "mean_journey_duration_days":      round(float(dur.mean()), 1),
            "p90_journey_duration_days":       int(dur.quantile(0.90)),
            "median_first_to_second_gap_days": round(float(gap.median()), 1) if len(gap) else np.nan,
            "pct_hospital_involved":           _pct(int(grp["hospital_involved_journey"].sum()), n)
                                               if "hospital_involved_journey" in grp.columns else np.nan,
            "pct_long_journey":                _pct(int(grp["long_journey"].sum()), n)
                                               if "long_journey" in grp.columns else np.nan,
            "pct_one_visit":                   _pct(int(grp["one_visit_journey"].sum()), n)
                                               if "one_visit_journey" in grp.columns else np.nan,
        })

    _sort_group(pd.DataFrame(rows)).to_csv(
        AUDIT_DIR / "screened_core_need_metric_summary.csv", index=False
    )


def save_specific_need_by_type(screened: pd.DataFrame) -> None:
    """
    For each of the 6 core need flags × each primary_journey_type:
      n_journeys_with_need, n_type_screened_total, n_need_screened_total,
      pct_of_type_with_need, pct_of_need_in_this_type.
    """
    type_totals = screened.groupby("primary_journey_type").size().rename("n_type_screened_total")
    rows = []

    for flag in CORE_NEED_FLAGS:
        if flag not in screened.columns:
            continue
        n_need_total = int(screened[flag].sum())
        grp = (
            screened[screened[flag]]
            .groupby("primary_journey_type")
            .size()
            .reset_index(name="n_journeys_with_need")
        )
        grp = grp.merge(type_totals.reset_index(), on="primary_journey_type", how="right")
        grp["n_journeys_with_need"] = grp["n_journeys_with_need"].fillna(0).astype(int)
        grp["need_flag"]            = flag
        grp["n_need_screened_total"] = n_need_total
        grp["pct_of_type_with_need"] = (
            grp["n_journeys_with_need"] / grp["n_type_screened_total"] * 100
        ).round(2)
        grp["pct_of_need_in_this_type"] = (
            grp["n_journeys_with_need"] / n_need_total * 100
        ).round(2).where(n_need_total > 0, other=0.0)
        rows.append(grp)

    if rows:
        result = pd.concat(rows, ignore_index=True)
        flag_order = {f: i for i, f in enumerate(CORE_NEED_FLAGS)}
        type_order = {t: i for i, t in enumerate(PRIMARY_TYPE_ORDER)}
        result["_fs"] = result["need_flag"].map(flag_order).fillna(99)
        result["_ts"] = result["primary_journey_type"].map(type_order).fillna(99)
        (
            result
            .sort_values(["_fs", "_ts"])
            .drop(columns=["_fs", "_ts"])
            [["need_flag", "primary_journey_type", "n_type_screened_total",
              "n_journeys_with_need", "n_need_screened_total",
              "pct_of_type_with_need", "pct_of_need_in_this_type"]]
            .to_csv(AUDIT_DIR / "screened_specific_need_by_journey_type.csv", index=False)
        )


def save_notes(screened: pd.DataFrame) -> None:
    n   = len(screened)
    n_any = int(screened["has_any_core_need"].sum())
    n_pat = int(screened["PatientDurableKey"].nunique())
    SEP  = "=" * 68
    DASH = "-" * 68
    W    = 66

    def wrap(text):
        return textwrap.fill(text, width=W, initial_indent="  ",
                             subsequent_indent="  ")

    grp_vc = screened["screened_core_need_group"].value_counts()

    lines = [
        SEP,
        "  SCREENED-ONLY SDOH ANALYSIS — SCOPE AND INTERPRETATION NOTES",
        "  Generated by: code/08_screened_only_sdoh_analysis.py",
        SEP,
        "",
        "  ANALYSIS SCOPE",
        DASH,
        wrap(
            "All outputs in this directory are restricted to journeys linked to "
            "patients who have at least one clean SDOH screening response in "
            "social_determinants.csv (after excluding DisplayName='*Unspecified' "
            "and Domain='NA' rows). This removes journeys where we have no "
            "SDOH information at all, enabling a within-screened comparison."
        ),
        "",
        f"  Screened journeys included     : {n:,}",
        f"  Unique screened patients       : {n_pat:,}",
        f"  Journeys with any core need    : {n_any:,}  ({_pct(n_any, n, 1)}%)",
        "",
        "  NEED GROUP DEFINITIONS",
        DASH,
        "  screened_no_positive_need : core_need_count == 0",
        "    Patient was screened but no core domain returned a positive screen.",
        "    Does NOT mean the patient has no unmet need — they may have needs",
        "    not captured by the six core flags (e.g., depression, alcohol use).",
        "",
        "  1 need   : core_need_count == 1",
        "  2-3 needs: core_need_count in [2, 3]",
        "  4+ needs : core_need_count >= 4  (max possible: 6)",
        "",
        "  NEED GROUP COUNTS IN THIS ANALYSIS",
        DASH,
    ]
    for g in SCREENED_GROUP_ORDER:
        cnt = int(grp_vc.get(g, 0))
        lines.append(
            f"  {g:<35}: {cnt:>10,}  ({_pct(cnt, n, 1):>5.1f}% of screened journeys)"
        )

    lines += [
        "",
        "  CORE NEED FLAGS INCLUDED",
        DASH,
        "  transportation_need         — 'Yes' to either transportation question",
        "  food_insecurity_need        — 'Sometimes/Often true' on food security Qs",
        "  housing_instability_need    — homeless/shelter 'Yes' OR moved count >= 2",
        "  financial_strain_need       — 'Yes' to rent/mortgage OR utility shutoff Q",
        "  ipv_need                    — 'Yes' to any IPV question",
        "  utilities_need              — 'Somewhat/Hard/Very hard' on basics Q",
        "",
        "  FLAGS EXCLUDED FROM core_need_count (not yet coded)",
        DASH,
        "  Alcohol Use (AUDIT-C composite), Depression (PHQ-2/Edinburgh),",
        "  Physical Activity (weekly volume), Social Connections (multi-item).",
        "",
        "  INTERPRETATION REQUIREMENTS",
        DASH,
        "",
        wrap(
            "1. All comparisons below are within the screened subset only. "
            "Results cannot be generalised to the full journey cohort without "
            "acknowledging that screened patients may differ systematically "
            "from unscreened patients (e.g., by encounter type, time period, "
            "or care setting)."
        ),
        "",
        wrap(
            "2. 'Screened with no recorded positive core need' does not mean "
            "no need exists. Use that phrase exactly — not 'no social needs' "
            "or 'low-need patients'."
        ),
        "",
        wrap(
            "3. For patients with positive screens, the flag reflects any "
            "response across all of the patient's encounters in the data window. "
            "Frame as 'patients with an ever-recorded positive screen' rather "
            "than 'patients currently experiencing this need'."
        ),
        "",
        wrap(
            "4. Use careful language throughout: 'associated with', "
            "'observed among', 'linked to', 'suggests' — "
            "not 'causes', 'leads to', or 'results in'."
        ),
        "",
        SEP,
    ]

    with open(AUDIT_DIR / "screened_sdoh_analysis_notes.txt", "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# 3. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(screened: pd.DataFrame) -> None:
    sep = "=" * 68
    n   = len(screened)

    print(f"\n{sep}")
    print("  SCREENED-ONLY SDOH ANALYSIS — AGGREGATE SUMMARY")
    print(sep)

    n_any = int(screened["has_any_core_need"].sum())
    n_no  = n - n_any
    print(f"\n  Screened journeys total               : {n:,}")
    print(f"  With any core need (positive screen)  : {n_any:,}  ({_pct(n_any, n, 1)}%)")
    print(f"  Screened, no positive core need       : {n_no:,}  ({_pct(n_no, n, 1)}%)")

    print(f"\n  screened_core_need_group distribution:")
    grp_vc = screened["screened_core_need_group"].value_counts()
    for g in SCREENED_GROUP_ORDER:
        cnt = int(grp_vc.get(g, 0))
        print(f"  {g:<35}: {cnt:>10,}  ({_pct(cnt, n, 1):>5.1f}%)")

    print(f"\n  % hospital_involved by screened_core_need_group:")
    if "hospital_involved_journey" in screened.columns:
        for g in SCREENED_GROUP_ORDER:
            grp   = screened[screened["screened_core_need_group"] == g]
            n_grp = len(grp)
            n_h   = int(grp["hospital_involved_journey"].sum())
            print(f"  {g:<35}: {_pct(n_h, n_grp, 1):>5.1f}%  "
                  f"({n_h:,} / {n_grp:,})")

    print(f"\n  primary_journey_type distribution by screened_core_need_group")
    print(f"  {'Type':<28}  {'No pos need':>11}  {'1 need':>8}  "
          f"{'2-3':>7}  {'4+':>6}  {'Total':>8}")
    print(f"  {'-'*72}")
    for ptype in PRIMARY_TYPE_ORDER:
        sub = screened[screened["primary_journey_type"] == ptype]
        if sub.empty:
            continue
        vals = {g: int((sub["screened_core_need_group"] == g).sum())
                for g in SCREENED_GROUP_ORDER}
        print(
            f"  {ptype:<28}  "
            f"{vals['screened_no_positive_need']:>11,}  "
            f"{vals['1 need']:>8,}  "
            f"{vals['2-3 needs']:>7,}  "
            f"{vals['4+ needs']:>6,}  "
            f"{len(sub):>8,}"
        )

    print(f"\n  Core need flags — journeys with positive screen (screened subset):")
    print(f"  {'Flag':<32}  {'N journeys':>10}  {'%':>6}")
    print(f"  {'-'*52}")
    for flag in CORE_NEED_FLAGS:
        if flag not in screened.columns:
            continue
        n_flag = int(screened[flag].sum())
        print(f"  {flag:<32}  {n_flag:>10,}  {_pct(n_flag, n, 1):>5.1f}%")

    print(f"""
  INTERPRETATION REMINDER
  All results above are within the screened patient subset only.
  'Screened with no recorded positive core need' does not mean
  no unmet need exists. Comparisons should not be generalised
  to the full journey cohort without appropriate caveats.
  Use 'associated with' / 'observed among' throughout.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 68)
    print("  SCREENED-ONLY SDOH ANALYSIS — DataFest 2026")
    print("=" * 68)

    print(f"\n  Loading {IN_PARQUET} …")
    screened = load_screened(IN_PARQUET)
    print(f"  Unique screened patients: {screened['PatientDurableKey'].nunique():,}")

    print("\n  Building aggregate outputs …")
    save_group_counts(screened)
    save_type_by_need_group(screened)
    save_type_by_any_need(screened)
    save_metric_summary(screened)
    save_specific_need_by_type(screened)
    save_notes(screened)

    for fname in [
        "screened_core_need_group_counts.csv",
        "screened_journey_type_by_core_need_group.csv",
        "screened_journey_type_by_any_core_need.csv",
        "screened_core_need_metric_summary.csv",
        "screened_specific_need_by_journey_type.csv",
        "screened_sdoh_analysis_notes.txt",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    print_summary(screened)


if __name__ == "__main__":
    main()
