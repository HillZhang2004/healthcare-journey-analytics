"""
09_validation_checks_before_plots.py
--------------------------------------
Targeted validation of pipeline outputs before creating final slide-ready
plots. Covers seven checks:
  1. Exclusion profile — how excluded encounters differ from retained ones
  2. Journey metric sanity — negative / extreme values
  3. Journey type overlap — pairwise overlap among boolean indicators
  4. Primary type priority sensitivity — alternative priority order
  5. SDOH screened vs. unscreened journey comparison
  6. Screened-only SDOH robustness — hospital_involved share by need group
  7. Validation summary notes

No patient-level or journey-level rows are printed or exported.
All outputs are aggregate summaries.

Run from project root:
    python code/09_validation_checks_before_plots.py
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants matching pipeline definitions
# ---------------------------------------------------------------------------

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

# Overlapping boolean indicators to test pairwise overlap
OVERLAP_INDICATORS = [
    "long_journey",
    "lab_heavy_journey",
    "imaging_heavy_journey",
    "hospital_involved_journey",
    "multi_department_journey",
    "mixed_setting_journey",
]

# (boolean_column, primary_type_label) — order defines priority
CURRENT_PRIORITY = [
    ("one_visit_journey",         "one_visit"),
    ("hospital_involved_journey", "hospital_involved"),
    ("long_journey",              "long_journey"),
    ("lab_heavy_journey",         "lab_heavy"),
    ("imaging_heavy_journey",     "imaging_heavy"),
    ("multi_department_journey",  "multi_department"),
    ("short_followup_journey",    "short_followup"),
]

ALTERNATIVE_PRIORITY = [
    ("one_visit_journey",         "one_visit"),
    ("long_journey",              "long_journey"),
    ("hospital_involved_journey", "hospital_involved"),
    ("multi_department_journey",  "multi_department"),
    ("lab_heavy_journey",         "lab_heavy"),
    ("imaging_heavy_journey",     "imaging_heavy"),
    ("short_followup_journey",    "short_followup"),
]

PRIMARY_TYPE_ORDER = [
    "one_visit", "hospital_involved", "long_journey", "lab_heavy",
    "imaging_heavy", "multi_department", "short_followup", "other_multi_encounter",
]

ENCOUNTER_FLAGS = [
    "IsEdVisit",
    "IsHospitalAdmission",
    "IsHospitalOutpatientVisit",
    "IsInpatientAdmission",
    "IsObservation",
    "IsOutpatientFaceToFaceVisit",
]

EXCL_CATS = [
    "retained_for_clean_journey",
    "excluded_minus1",
    "excluded_ambiguous_or_unmatched",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def safe_read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False, **kwargs)


def normalize_numeric_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _pct(num, den, decimals: int = 2) -> float:
    return round(num / den * 100, decimals) if den else 0.0


def _build_primary_type(jdf: pd.DataFrame, priority: list) -> pd.Series:
    conditions = [jdf[col] for col, _ in priority]
    choices    = [label for _, label in priority]
    return np.select(conditions, choices, default="other_multi_encounter")


def _screened_col(jdf: pd.DataFrame) -> str:
    if "has_clean_sdoh_response" in jdf.columns:
        return "has_clean_sdoh_response"
    if "has_any_sdoh_response" in jdf.columns:
        return "has_any_sdoh_response"
    raise KeyError("No SDOH screening indicator column found in parquet.")


# ---------------------------------------------------------------------------
# Check 1: Exclusion profile for the clean journey table
# ---------------------------------------------------------------------------

def check_exclusion_profile(results: dict) -> None:
    print("\n[1] Exclusion profile …")

    enc  = safe_read(DATA_DIR / "encounters.csv")
    diag = safe_read(DATA_DIR / "diagnosis.csv")

    enc["_pk"]  = normalize_numeric_key(enc["PrimaryDiagnosisKey"])
    diag["_dk"] = normalize_numeric_key(diag["DiagnosisKey"])

    # Build the same clean-mapping key set used in 03_build_journeys.py:
    # only DiagnosisKeys with exactly one unique DiagnosisValue.
    clean_dk = (
        diag.groupby("_dk")["DiagnosisValue"]
        .nunique()
        .pipe(lambda s: s[s == 1])
        .index
    )
    clean_key_set = set(clean_dk.dropna())

    pk = enc["_pk"]
    enc["_excl"] = np.select(
        [pk == -1, pk.isin(clean_key_set)],
        ["excluded_minus1", "retained_for_clean_journey"],
        default="excluded_ambiguous_or_unmatched",
    )

    n_total = len(enc)
    excl_vc = enc["_excl"].value_counts()
    results["n_encounters_total"]   = n_total
    results["n_retained"]           = int(excl_vc.get("retained_for_clean_journey", 0))
    results["n_excluded_minus1"]    = int(excl_vc.get("excluded_minus1", 0))
    results["n_excluded_ambiguous"] = int(excl_vc.get("excluded_ambiguous_or_unmatched", 0))

    # -- 1a: by Type ---------------------------------------------------
    grp = enc.groupby(["Type", "_excl"]).size().unstack(fill_value=0)
    grp = grp.reindex(columns=EXCL_CATS, fill_value=0)
    grp["n_total"] = grp.sum(axis=1)
    for c in EXCL_CATS:
        grp[f"pct_{c}"] = (grp[c] / grp["n_total"] * 100).round(2)
    grp.reset_index().sort_values("n_total", ascending=False).to_csv(
        AUDIT_DIR / "validation_excluded_vs_retained_by_type.csv", index=False
    )

    # store top excluded types for notes
    excl_only = enc[enc["_excl"] != "retained_for_clean_journey"]
    results["top_excluded_types"] = (
        excl_only.groupby("Type").size().sort_values(ascending=False).head(5).to_dict()
    )

    # -- 1b: by VisitTypeDescription ------------------------------------
    grp2 = enc.groupby(["VisitTypeDescription", "_excl"]).size().unstack(fill_value=0)
    grp2 = grp2.reindex(columns=EXCL_CATS, fill_value=0)
    grp2["n_total"] = grp2.sum(axis=1)
    for c in EXCL_CATS:
        grp2[f"pct_{c}"] = (grp2[c] / grp2["n_total"] * 100).round(2)
    grp2.reset_index().sort_values("n_total", ascending=False).to_csv(
        AUDIT_DIR / "validation_excluded_vs_retained_by_visit_description.csv", index=False
    )

    # -- 1c: by year (parsed from Date column, format MM/DD/YY) ---------
    enc["_year"] = pd.to_datetime(enc["Date"], format="%m/%d/%y", errors="coerce").dt.year
    grp3 = enc.groupby(["_year", "_excl"]).size().unstack(fill_value=0)
    grp3 = grp3.reindex(columns=EXCL_CATS, fill_value=0)
    grp3["n_total"] = grp3.sum(axis=1)
    for c in EXCL_CATS:
        grp3[f"pct_{c}"] = (grp3[c] / grp3["n_total"] * 100).round(2)
    grp3.reset_index().rename(columns={"_year": "year"}).sort_values("year").to_csv(
        AUDIT_DIR / "validation_excluded_vs_retained_by_year.csv", index=False
    )

    # -- 1d: by encounter flag columns ----------------------------------
    flag_rows = []
    for cat in EXCL_CATS:
        sub   = enc[enc["_excl"] == cat]
        n_sub = len(sub)
        for flag in ENCOUNTER_FLAGS:
            n_flag = int((pd.to_numeric(sub[flag], errors="coerce") == 1).sum())
            flag_rows.append({
                "exclusion_category": cat,
                "flag":               flag,
                "n_encounters":       n_sub,
                "n_flag_true":        n_flag,
                "pct_flag_true":      _pct(n_flag, n_sub),
            })
    pd.DataFrame(flag_rows).to_csv(
        AUDIT_DIR / "validation_excluded_vs_retained_by_flags.csv", index=False
    )

    print(f"  Total encounters           : {n_total:,}")
    print(f"  retained_for_clean_journey : {results['n_retained']:,}  "
          f"({_pct(results['n_retained'], n_total, 1)}%)")
    print(f"  excluded_minus1            : {results['n_excluded_minus1']:,}  "
          f"({_pct(results['n_excluded_minus1'], n_total, 1)}%)")
    print(f"  excluded_ambiguous         : {results['n_excluded_ambiguous']:,}  "
          f"({_pct(results['n_excluded_ambiguous'], n_total, 1)}%)")
    print(f"  Top Type values among excluded encounters:")
    for t, n in list(results["top_excluded_types"].items()):
        print(f"    {t:<32}: {n:,}")


# ---------------------------------------------------------------------------
# Check 2: Journey metric sanity checks
# ---------------------------------------------------------------------------

def check_metric_sanity(jdf_clean: pd.DataFrame, results: dict) -> None:
    print("\n[2] Journey metric sanity checks …")

    n = len(jdf_clean)

    checks = [
        ("negative_journey_duration_days",
         int((jdf_clean["journey_duration_days"] < 0).sum())),
        ("negative_first_to_second_gap_days",
         int((jdf_clean["first_to_second_gap_days"].dropna() < 0).sum())),
        ("negative_median_gap_days",
         int((jdf_clean["median_gap_days"].dropna() < 0).sum())),
        ("negative_max_gap_days",
         int((jdf_clean["max_gap_days"].dropna() < 0).sum())),
        ("journey_duration_days_gt_1460",
         int((jdf_clean["journey_duration_days"] > 1460).sum())),
        ("first_to_second_gap_days_gt_1460",
         int((jdf_clean["first_to_second_gap_days"].dropna() > 1460).sum())),
        ("n_encounters_lte_0",
         int((jdf_clean["n_encounters"] <= 0).sum())),
    ]

    rows = []
    n_failed = 0
    for name, count in checks:
        passed = count == 0
        if not passed:
            n_failed += 1
        rows.append({
            "check":            name,
            "n_flagged":        count,
            "pct_of_journeys":  _pct(count, n),
            "passed":           passed,
        })

    pd.DataFrame(rows).to_csv(AUDIT_DIR / "validation_journey_metric_sanity.csv", index=False)
    results["metric_sanity_n_failed"] = n_failed
    results["metric_sanity_rows"]     = rows

    print(f"  Journeys checked : {n:,}")
    print(f"  Checks failed    : {n_failed}/{len(checks)}")
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['n_flagged']:,} flagged  ({r['pct_of_journeys']}%)")


# ---------------------------------------------------------------------------
# Check 3: Journey type pairwise overlap
# ---------------------------------------------------------------------------

def check_type_overlap(jdf_cl: pd.DataFrame) -> None:
    print("\n[3] Journey type overlap check …")

    indicators = [c for c in OVERLAP_INDICATORS if c in jdf_cl.columns]
    rows = []

    for i, a in enumerate(indicators):
        for b in indicators[i + 1:]:
            n_a    = int(jdf_cl[a].sum())
            n_b    = int(jdf_cl[b].sum())
            n_both = int((jdf_cl[a] & jdf_cl[b]).sum())
            rows.append({
                "indicator_A":   a,
                "indicator_B":   b,
                "n_A":           n_a,
                "n_B":           n_b,
                "n_both":        n_both,
                "pct_A_also_B":  _pct(n_both, n_a),
                "pct_B_also_A":  _pct(n_both, n_b),
            })

    pd.DataFrame(rows).to_csv(AUDIT_DIR / "validation_journey_type_overlap.csv", index=False)
    print(f"  Computed {len(rows)} pairwise overlaps across {len(indicators)} indicators.")


# ---------------------------------------------------------------------------
# Check 4: Primary journey type priority sensitivity
# ---------------------------------------------------------------------------

def check_priority_sensitivity(jdf_cl: pd.DataFrame) -> None:
    print("\n[4] Primary journey type priority sensitivity …")

    jdf_cl = jdf_cl.copy()
    alt    = _build_primary_type(jdf_cl, ALTERNATIVE_PRIORITY)
    n      = len(jdf_cl)
    n_same = int((jdf_cl["primary_journey_type"] == alt).sum())

    rows = []
    for t in PRIMARY_TYPE_ORDER:
        n_curr = int((jdf_cl["primary_journey_type"] == t).sum())
        n_alt  = int((alt == t).sum())
        rows.append({
            "primary_journey_type":       t,
            "n_current_priority":         n_curr,
            "pct_current_priority":       _pct(n_curr, n),
            "n_alternative_priority":     n_alt,
            "pct_alternative_priority":   _pct(n_alt, n),
            "n_difference":               n_alt - n_curr,
            "pct_difference":             round(_pct(n_alt, n) - _pct(n_curr, n), 2),
        })

    rows.append({
        "primary_journey_type":     "_TOTAL_SAME_ASSIGNMENT",
        "n_current_priority":       n,
        "pct_current_priority":     100.0,
        "n_alternative_priority":   n_same,
        "pct_alternative_priority": _pct(n_same, n),
        "n_difference":             n_same - n,
        "pct_difference":           round(_pct(n_same, n) - 100.0, 2),
    })

    pd.DataFrame(rows).to_csv(
        AUDIT_DIR / "validation_primary_type_priority_sensitivity.csv", index=False
    )
    print(f"  Same assignment under both priorities: {n_same:,} / {n:,}  "
          f"({_pct(n_same, n, 1)}%)")
    print(f"  Reclassified                         : {n - n_same:,}  "
          f"({_pct(n - n_same, n, 1)}%)")
    print(f"  Type-level changes:")
    for r in rows[:-1]:
        if r["n_difference"] != 0:
            print(f"    {r['primary_journey_type']:<28}: "
                  f"{r['n_current_priority']:>8,} → {r['n_alternative_priority']:>8,}  "
                  f"(Δ {r['n_difference']:+,})")


# ---------------------------------------------------------------------------
# Check 5: SDOH screened vs. unscreened comparison
# ---------------------------------------------------------------------------

def check_screened_vs_unscreened(jdf_need: pd.DataFrame, results: dict) -> None:
    print("\n[5] Screened vs. unscreened comparison …")

    scol  = _screened_col(jdf_need)
    jdf   = jdf_need.copy()
    jdf["_screened"] = jdf[scol].map({True: "screened", False: "unscreened"})
    jdf["_year"]     = jdf["first_encounter_date"].dt.year

    n_sc = int((jdf["_screened"] == "screened").sum())
    n_un = int((jdf["_screened"] == "unscreened").sum())

    # Overall hospital share for notes
    sc_sub = jdf[jdf["_screened"] == "screened"]
    un_sub = jdf[jdf["_screened"] == "unscreened"]
    results["screened_pct_hospital"]   = _pct(int(sc_sub["hospital_involved_journey"].sum()), n_sc, 1)
    results["unscreened_pct_hospital"] = _pct(int(un_sub["hospital_involved_journey"].sum()), n_un, 1)
    results["n_screened"]   = n_sc
    results["n_unscreened"] = n_un

    rows = []
    for status, sub in [("screened", sc_sub), ("unscreened", un_sub)]:
        n_grp = len(sub)
        # Overall row for this screening status
        rows.append({
            "screening_status":           status,
            "primary_journey_type":       "ALL",
            "n_journeys":                 n_grp,
            "pct_of_screening_group":     100.0,
            "pct_hospital_involved":      _pct(int(sub["hospital_involved_journey"].sum()), n_grp),
            "median_n_encounters":        round(float(sub["n_encounters"].median()), 1),
            "median_journey_duration_days": int(sub["journey_duration_days"].median()),
            "median_first_encounter_year": int(sub["_year"].dropna().median())
                                           if not sub["_year"].dropna().empty else np.nan,
        })
        # Per journey type
        for t in PRIMARY_TYPE_ORDER:
            sub_t = sub[sub["primary_journey_type"] == t]
            n_t   = len(sub_t)
            rows.append({
                "screening_status":           status,
                "primary_journey_type":       t,
                "n_journeys":                 n_t,
                "pct_of_screening_group":     _pct(n_t, n_grp),
                "pct_hospital_involved":      _pct(int(sub_t["hospital_involved_journey"].sum()), n_t)
                                              if n_t > 0 else np.nan,
                "median_n_encounters":        round(float(sub_t["n_encounters"].median()), 1)
                                              if n_t > 0 else np.nan,
                "median_journey_duration_days": int(sub_t["journey_duration_days"].median())
                                              if n_t > 0 else np.nan,
                "median_first_encounter_year": int(sub_t["_year"].dropna().median())
                                              if n_t > 0 and not sub_t["_year"].dropna().empty
                                              else np.nan,
            })

    pd.DataFrame(rows).to_csv(AUDIT_DIR / "validation_screened_vs_unscreened.csv", index=False)

    # Console: type distribution comparison
    sc_vc = sc_sub["primary_journey_type"].value_counts()
    un_vc = un_sub["primary_journey_type"].value_counts()
    results["type_diff"] = {}
    print(f"  Screened journeys   : {n_sc:,}   Unscreened : {n_un:,}")
    print(f"  hospital_involved%  : screened={results['screened_pct_hospital']}%  "
          f"unscreened={results['unscreened_pct_hospital']}%")
    print(f"\n  Journey type distribution (screened vs. unscreened):")
    print(f"  {'Type':<28}  {'Screened%':>10}  {'Unscreened%':>12}  {'Δpp':>6}")
    for t in PRIMARY_TYPE_ORDER:
        sc_p = _pct(int(sc_vc.get(t, 0)), n_sc, 1)
        un_p = _pct(int(un_vc.get(t, 0)), n_un, 1)
        diff = round(sc_p - un_p, 1)
        results["type_diff"][t] = diff
        print(f"  {t:<28}  {sc_p:>9.1f}%  {un_p:>11.1f}%  {diff:>+5.1f}")


# ---------------------------------------------------------------------------
# Check 6: Screened-only SDOH robustness
# ---------------------------------------------------------------------------

def check_screened_robustness(jdf_need: pd.DataFrame, results: dict) -> None:
    print("\n[6] Screened-only SDOH robustness …")

    scol     = _screened_col(jdf_need)
    screened = jdf_need[jdf_need[scol] == True].copy()

    screened["screened_core_need_group"] = pd.cut(
        screened["core_need_count"],
        bins=[-1, 0, 1, 3, len(CORE_NEED_FLAGS)],
        labels=SCREENED_GROUP_ORDER,
        right=True,
    ).astype(str)

    rows = []

    # By screened_core_need_group
    for g in SCREENED_GROUP_ORDER:
        sub = screened[screened["screened_core_need_group"] == g]
        n   = len(sub)
        n_h = int(sub["hospital_involved_journey"].sum())
        rows.append({
            "stratification":        "screened_core_need_group",
            "stratum":               g,
            "n_journeys":            n,
            "n_hospital_involved":   n_h,
            "pct_hospital_involved": _pct(n_h, n),
        })

    # By has_any_core_need
    for val, label in [(False, "has_any_core_need=False"), (True, "has_any_core_need=True")]:
        sub = screened[screened["has_any_core_need"] == val]
        n   = len(sub)
        n_h = int(sub["hospital_involved_journey"].sum())
        rows.append({
            "stratification":        "has_any_core_need",
            "stratum":               label,
            "n_journeys":            n,
            "n_hospital_involved":   n_h,
            "pct_hospital_involved": _pct(n_h, n),
        })

    # By each specific core need flag
    for flag in CORE_NEED_FLAGS:
        if flag not in screened.columns:
            continue
        for val, label in [(False, f"{flag}=False"), (True, f"{flag}=True")]:
            sub = screened[screened[flag] == val]
            n   = len(sub)
            n_h = int(sub["hospital_involved_journey"].sum())
            rows.append({
                "stratification":        f"specific_need:{flag}",
                "stratum":               label,
                "n_journeys":            n,
                "n_hospital_involved":   n_h,
                "pct_hospital_involved": _pct(n_h, n),
            })

    pd.DataFrame(rows).to_csv(
        AUDIT_DIR / "validation_screened_sdoh_hospital_share.csv", index=False
    )

    # Store need-group breakdown for notes
    grp_rows = [r for r in rows if r["stratification"] == "screened_core_need_group"]
    results["hospital_by_need_group"] = {
        r["stratum"]: r["pct_hospital_involved"] for r in grp_rows
    }

    print(f"  Screened journeys : {len(screened):,}")
    print(f"\n  hospital_involved% by screened_core_need_group:")
    for r in grp_rows:
        print(f"  {r['stratum']:<35}: {r['pct_hospital_involved']:>5.1f}%  "
              f"(n={r['n_journeys']:,})")


# ---------------------------------------------------------------------------
# Check 7: Validation notes
# ---------------------------------------------------------------------------

def write_validation_notes(results: dict) -> None:
    SEP  = "=" * 68
    DASH = "-" * 68
    W    = 66

    def wrap(text: str) -> str:
        return textwrap.fill(text, width=W, initial_indent="  ", subsequent_indent="  ")

    n_total    = results.get("n_encounters_total", 0)
    n_retained = results.get("n_retained", 0)
    n_excl_m1  = results.get("n_excluded_minus1", 0)
    n_excl_amb = results.get("n_excluded_ambiguous", 0)
    n_failed   = results.get("metric_sanity_n_failed", 0)
    n_sc       = results.get("n_screened", 0)
    n_un       = results.get("n_unscreened", 0)
    sc_hosp    = results.get("screened_pct_hospital", 0.0)
    un_hosp    = results.get("unscreened_pct_hospital", 0.0)
    hosp_by_grp = results.get("hospital_by_need_group", {})
    type_diff  = results.get("type_diff", {})
    sanity_rows = results.get("metric_sanity_rows", [])

    failed_checks = [r for r in sanity_rows if not r["passed"]]
    max_type_diff = max((abs(v) for v in type_diff.values()), default=0.0)

    # Assess monotone pattern for safe-to-visualise judgement
    grp_values = [hosp_by_grp.get(g, None) for g in SCREENED_GROUP_ORDER]
    grp_values = [v for v in grp_values if v is not None]
    pattern_monotone = (
        len(grp_values) == len(SCREENED_GROUP_ORDER)
        and all(grp_values[i] <= grp_values[i + 1] for i in range(len(grp_values) - 1))
    )

    lines = [
        SEP,
        "  VALIDATION NOTES BEFORE SLIDES — DataFest 2026",
        "  Generated by: code/09_validation_checks_before_plots.py",
        SEP,
        "",
        "  1. ENCOUNTER EXCLUSION PROFILE",
        DASH,
        f"  Total encounters               : {n_total:,}",
        f"  Retained (clean journey)       : {n_retained:,}  "
        f"({_pct(n_retained, n_total, 1)}%)",
        f"  Excluded (PrimaryDiagnosisKey == -1) : {n_excl_m1:,}  "
        f"({_pct(n_excl_m1, n_total, 1)}%)",
        f"  Excluded (ambiguous/unmatched) : {n_excl_amb:,}  "
        f"({_pct(n_excl_amb, n_total, 1)}%)",
        "",
    ]

    top_types = results.get("top_excluded_types", {})
    if top_types:
        lines.append("  Top encounter Type values among excluded encounters:")
        for t, n in list(top_types.items()):
            lines.append(f"    {t:<32}: {n:,}")
        lines.append("")

    lines += [
        wrap(
            "CAVEAT: Excluded encounters are not uniformly distributed across "
            "encounter types. The retained cohort represents diagnosis-anchored "
            "journeys only. Any slide claiming completeness of coverage should "
            "note that encounters without a clean primary diagnosis code were "
            "excluded. Frame results as 'among encounters with a clean diagnosis "
            "code' rather than 'all encounters'."
        ),
        "",
        "  2. JOURNEY METRIC SANITY",
        DASH,
        f"  Checks run    : {len(sanity_rows)}",
        f"  Checks FAILED : {n_failed}",
        "",
    ]

    if n_failed == 0:
        lines.append("  STATUS: PASS — all metric sanity checks passed. "
                     "Gap and duration metrics are safe to display.")
    else:
        lines.append("  STATUS: ISSUES FOUND — see below before finalising slides.")
        for r in failed_checks:
            lines.append(
                f"    FAIL  {r['check']:<42}: "
                f"n={r['n_flagged']:,}  ({r['pct_of_journeys']}%)"
            )
        lines += [
            "",
            wrap(
                "Negative gap values may indicate same-day encounters ordered "
                "differently across sources, or data-entry anomalies. "
                "Journeys with duration > 1460 days span the full observation "
                "window and may represent long-standing chronic conditions rather "
                "than a discrete care episode. Consider capping display at the "
                "95th or 99th percentile for visualisations."
            ),
        ]

    lines += [
        "",
        "  3. JOURNEY TYPE OVERLAP",
        DASH,
        wrap(
            "The six indicator flags (long_journey, lab_heavy_journey, "
            "imaging_heavy_journey, hospital_involved_journey, "
            "multi_department_journey, mixed_setting_journey) are overlapping "
            "booleans — a single journey can satisfy multiple criteria. "
            "Pairwise overlap counts are in "
            "validation_journey_type_overlap.csv. The primary_journey_type "
            "column assigns each journey to exactly one type via a fixed "
            "priority order, so all charts using primary_journey_type are "
            "mutually exclusive by construction."
        ),
        "",
        "  4. PRIMARY JOURNEY TYPE PRIORITY SENSITIVITY",
        DASH,
        wrap(
            "An alternative priority order was tested in which long_journey is "
            "promoted above hospital_involved and multi_department is promoted "
            "above lab_heavy. The percentage of journeys assigned to the same "
            "type under both orderings, and the magnitude of type-level shifts, "
            "are in validation_primary_type_priority_sensitivity.csv."
        ),
        "",
        wrap(
            "CAVEAT: Any claim about the relative prevalence of 'hospital-involved' "
            "vs 'long journey' types should acknowledge that counts depend on the "
            "chosen priority order. The one_visit category is unaffected."
        ),
        "",
        "  5. SCREENED VS. UNSCREENED JOURNEYS",
        DASH,
        f"  Screened journeys           : {n_sc:,}",
        f"  Unscreened journeys         : {n_un:,}",
        f"  Screened   hospital_involved: {sc_hosp}%",
        f"  Unscreened hospital_involved: {un_hosp}%",
        "",
    ]

    if max_type_diff > 5.0:
        lines += [
            wrap(
                f"NOTE: The largest screened vs. unscreened journey type difference "
                f"is {max_type_diff:.1f} percentage points. Screened and unscreened "
                f"patients have meaningfully different care-setting profiles. "
                f"This likely reflects systematic differences in who receives SDOH "
                f"screening (e.g., primary care encounters are more likely to "
                f"include a screening than ED visits). All SDOH-based findings "
                f"must be interpreted within the screened subset only, not "
                f"generalised to the full journey cohort."
            ),
        ]
    else:
        lines += [
            wrap(
                f"Journey type profiles are broadly similar between screened and "
                f"unscreened patients (largest difference: {max_type_diff:.1f}pp). "
                f"However, screened patients should still be treated as a "
                f"distinct analytical subgroup because SDOH screening is not "
                f"a random process."
            ),
        ]

    lines += [
        "",
        "  6. SCREENED-ONLY SDOH ROBUSTNESS",
        DASH,
        "  hospital_involved% by screened_core_need_group:",
    ]
    for g in SCREENED_GROUP_ORDER:
        pct = hosp_by_grp.get(g, float("nan"))
        lines.append(f"    {g:<35}: {pct:.1f}%")

    lines += [""]
    if pattern_monotone:
        lines += [
            "  STATUS: PATTERN IS MONOTONE.",
            "",
            wrap(
                "SAFE TO VISUALISE: The share of hospital_involved journeys "
                "increases consistently with core need count among screened "
                "patients. This gradient holds across need groups and is "
                "robust within the screened-patient subset. It is suitable "
                "for slide presentation with the caveats below."
            ),
        ]
    else:
        lines += [
            "  STATUS: PATTERN IS NOT STRICTLY MONOTONE — review values above.",
            "",
            wrap(
                "CAUTION: The hospital_involved share does not increase "
                "monotonically with core need count. Consider using "
                "has_any_core_need as a binary comparison or collapsing "
                "the '2-3 needs' and '4+ needs' groups before presenting."
            ),
        ]

    lines += [
        "",
        "  7. REQUIRED INTERPRETATION LANGUAGE",
        DASH,
        wrap(
            "Use 'associated with', 'observed among', 'linked to', or 'suggests' "
            "throughout. Never 'causes', 'leads to', or 'results in'."
        ),
        "",
        wrap(
            "For SDOH findings: 'screened patients with a recorded positive screen' "
            "and 'screened with no recorded positive core need'. "
            "Never 'patients with no social needs' or 'low-need patients'. "
            "A negative screen does not indicate absence of need."
        ),
        "",
        wrap(
            "For journey metrics: results describe diagnosis-anchored journeys "
            "among encounters with a clean primary diagnosis code. Encounters "
            "without a clean code were excluded and are not represented."
        ),
        "",
        wrap(
            "For screened-vs-unscreened comparisons: screened patients are not a "
            "random sample. SDOH patterns within the screened subset cannot be "
            "generalised to the full cohort without acknowledging selection into "
            "screening."
        ),
        "",
        SEP,
    ]

    with open(AUDIT_DIR / "validation_notes_before_slides.txt", "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEP = "=" * 68
    print(f"\n{SEP}")
    print("  VALIDATION CHECKS BEFORE SLIDES — DataFest 2026")
    print(SEP)

    results = {}

    print("\n  Loading journey parquet tables …")
    jdf_clean = pd.read_parquet(TABLES_DIR / "journeys_clean_diagnosisvalue.parquet")
    jdf_cl    = pd.read_parquet(TABLES_DIR / "journeys_classified.parquet")
    jdf_need  = pd.read_parquet(TABLES_DIR / "journeys_with_core_sdoh_need_flags.parquet")
    print(f"  journeys_clean      : {len(jdf_clean):>10,} rows")
    print(f"  journeys_classified : {len(jdf_cl):>10,} rows")
    print(f"  journeys_need_flags : {len(jdf_need):>10,} rows")

    check_exclusion_profile(results)
    check_metric_sanity(jdf_clean, results)
    check_type_overlap(jdf_cl)
    check_priority_sensitivity(jdf_cl)
    check_screened_vs_unscreened(jdf_need, results)
    check_screened_robustness(jdf_need, results)
    write_validation_notes(results)

    print(f"\n{SEP}")
    print("  VALIDATION COMPLETE — SHORT SUMMARY")
    print(SEP)

    print(f"\n  Metric sanity failures  : {results['metric_sanity_n_failed']}")

    top_types = results.get("top_excluded_types", {})
    if top_types:
        top_t, top_n = next(iter(top_types.items()))
        print(f"  Top excluded Type       : {top_t}  ({top_n:,} excluded encounters)")

    print(f"\n  Screened vs. unscreened hospital_involved share:")
    print(f"    Screened   : {results['screened_pct_hospital']}%")
    print(f"    Unscreened : {results['unscreened_pct_hospital']}%")

    print(f"\n  hospital_involved% by screened_core_need_group (screened only):")
    for g, pct in results.get("hospital_by_need_group", {}).items():
        print(f"    {g:<35}: {pct:.1f}%")

    saved_files = [
        "validation_excluded_vs_retained_by_type.csv",
        "validation_excluded_vs_retained_by_visit_description.csv",
        "validation_excluded_vs_retained_by_year.csv",
        "validation_excluded_vs_retained_by_flags.csv",
        "validation_journey_metric_sanity.csv",
        "validation_journey_type_overlap.csv",
        "validation_primary_type_priority_sensitivity.csv",
        "validation_screened_vs_unscreened.csv",
        "validation_screened_sdoh_hospital_share.csv",
        "validation_notes_before_slides.txt",
    ]
    print(f"\n  Outputs saved to outputs/audit/:")
    for f in saved_files:
        print(f"    {f}")


if __name__ == "__main__":
    main()
