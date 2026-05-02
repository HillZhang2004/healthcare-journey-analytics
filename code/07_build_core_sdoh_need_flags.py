"""
07_build_core_sdoh_need_flags.py
---------------------------------
Build conservative patient-level SDOH need/risk flags, then attach them
to the journey table for aggregate comparison by journey type.

Each flag is a positive screen only — a patient who was never asked, who
declined to answer, or who has incomplete SDOH data will have the flag set
to False.  False does NOT mean no need.

Inputs:
    outputs/tables/journeys_classified_with_patient_sdoh.parquet
    data/social_determinants.csv

Outputs:
    outputs/tables/journeys_with_core_sdoh_need_flags.parquet
    outputs/audit/core_sdoh_need_patient_counts.csv
    outputs/audit/core_sdoh_need_journey_counts.csv
    outputs/audit/journey_type_by_core_need_group.csv
    outputs/audit/journey_type_by_any_core_need.csv
    outputs/audit/journey_type_by_specific_core_need.csv
    outputs/audit/core_need_group_metric_summary.csv
    outputs/audit/core_sdoh_need_flags_notes.txt

Run from project root:
    python code/07_build_core_sdoh_need_flags.py

No raw patient-level or journey-level rows are printed or exported.
All printed output is aggregate only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True,  exist_ok=True)

IN_PARQUET  = TABLES_DIR / "journeys_classified_with_patient_sdoh.parquet"
OUT_PARQUET = TABLES_DIR / "journeys_with_core_sdoh_need_flags.parquet"

# ---------------------------------------------------------------------------
# Domain normalization — identical to scripts 05 and 06
# ---------------------------------------------------------------------------

DOMAIN_CORRECTION_MAP = {
    "alcohol use":                "Alcohol Use",
    "depression":                 "Depression",
    "financial resource strain":  "Financial Resource Strain",
    "food insecurity":            "Food Insecurity",
    "housing stability":          "Housing Stability",
    "intimate partner violence":  "Intimate Partner Violence",
    "intimate partner violance":  "Intimate Partner Violence",
    "physical activity":          "Physical Activity",
    "social connections":         "Social Connections",
    "stress":                     "Stress",
    "transportation needs":       "Transportation Needs",
    "utilities":                  "Utilities",
}

# Six core need flags that count toward core_need_count.
CORE_NEED_FLAGS = [
    "transportation_need",
    "food_insecurity_need",
    "housing_instability_need",
    "financial_strain_need",
    "ipv_need",
    "utilities_need",
]

# One additional sensitivity/context flag — not counted in core_need_count.
SENSITIVITY_FLAGS = ["elevated_stress_flag"]

ALL_FLAGS = CORE_NEED_FLAGS + SENSITIVITY_FLAGS

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def safe_read(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path, keep_default_na=False, na_values=[], low_memory=False, usecols=usecols
    )


def normalize_numeric_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def apply_domain_corrections(domain_raw: pd.Series) -> pd.Series:
    stripped = domain_raw.astype(str).str.strip()
    lower    = stripped.str.lower()
    return lower.map(DOMAIN_CORRECTION_MAP).fillna(stripped.str.title())


# ---------------------------------------------------------------------------
# 1. Load and clean SDOH
# ---------------------------------------------------------------------------

def load_clean_sdoh() -> tuple[pd.DataFrame, dict]:
    """
    Load social_determinants.csv, normalize keys, apply domain corrections,
    and exclude *Unspecified / NA rows.
    """
    sdoh = safe_read(DATA_DIR / "social_determinants.csv")
    sdoh["PatientDurableKey"] = normalize_numeric_key(sdoh["PatientDurableKey"])
    sdoh = sdoh[sdoh["PatientDurableKey"].notna()]

    n_total = len(sdoh)
    sdoh["Domain_raw"]   = sdoh["Domain"].astype(str).str.strip()
    sdoh["Domain_clean"] = apply_domain_corrections(sdoh["Domain"])

    mask_unspec    = sdoh["DisplayName"] == "*Unspecified"
    mask_na_domain = sdoh["Domain_raw"]  == "NA"
    n_excl         = int((mask_unspec | mask_na_domain).sum())

    clean = sdoh[~mask_unspec & ~mask_na_domain].copy()

    counts = {
        "n_raw_sdoh_rows":           n_total,
        "n_rows_excluded":           n_excl,
        "n_rows_retained":           len(clean),
        "n_patients_in_clean_rows":  int(clean["PatientDurableKey"].nunique()),
    }
    return clean, counts


# ---------------------------------------------------------------------------
# 2. Compute individual need-flag patient sets
# ---------------------------------------------------------------------------

def _patient_set(df: pd.DataFrame, condition: pd.Series) -> set:
    """Return the set of Int64 PatientDurableKey values where condition is True."""
    return set(df.loc[condition, "PatientDurableKey"].dropna().unique())


def compute_transportation_need(sdoh: pd.DataFrame) -> set:
    """
    Domain: Transportation Needs
    Positive: AnswerText == "Yes" on either transportation-barrier question.
    """
    return _patient_set(
        sdoh,
        (sdoh["Domain_clean"] == "Transportation Needs") &
        (sdoh["AnswerText"] == "Yes"),
    )


def compute_food_insecurity_need(sdoh: pd.DataFrame) -> set:
    """
    Domain: Food Insecurity
    Positive: AnswerText in {"Sometimes true", "Often true"}.
    (Standard HRSN protocol — either food-worry or food-insufficiency question.)
    """
    return _patient_set(
        sdoh,
        (sdoh["Domain_clean"] == "Food Insecurity") &
        (sdoh["AnswerText"].isin({"Sometimes true", "Often true"})),
    )


def compute_housing_instability_need(sdoh: pd.DataFrame) -> tuple[set, dict]:
    """
    Domain: Housing Stability
    Positive if:
      (a) Any homeless/shelter question answered "Yes", OR
      (b) Any moved-count question has a numeric answer >= 2.
    Patient declined, unable to answer, *Unknown are excluded (non-responses).
    """
    housing = sdoh[sdoh["Domain_clean"] == "Housing Stability"].copy()
    dn_lower = housing["DisplayName"].str.lower()

    # (a) Homeless / shelter question
    homeless_mask = (
        (dn_lower.str.contains("homeless") | dn_lower.str.contains("shelter")) &
        (housing["AnswerText"] == "Yes")
    )
    patients_homeless = _patient_set(housing, homeless_mask)

    # (b) Moved-count question: numeric answer >= 2
    moved_mask = (
        dn_lower.str.contains("how many times have you moved") |
        dn_lower.str.contains("how many places have you lived")
    )
    moved_rows = housing[moved_mask].copy()
    moved_numeric = pd.to_numeric(moved_rows["AnswerText"], errors="coerce")
    patients_moved = set(
        moved_rows.loc[moved_numeric >= 2, "PatientDurableKey"].dropna().unique()
    )

    diagnostics = {
        "n_rows_homeless_question":        int(moved_mask.sum() + homeless_mask.sum()),
        "n_positive_homeless_patients":    len(patients_homeless),
        "n_rows_moved_question":           int(moved_mask.sum()),
        "n_positive_moved_gte2_patients":  len(patients_moved),
    }
    return patients_homeless | patients_moved, diagnostics


def compute_financial_strain_need(sdoh: pd.DataFrame) -> tuple[set, dict]:
    """
    Domain: Financial Resource Strain
    Positive if:
      (a) Rent/mortgage question answered "Yes", OR
      (b) Utility shutoff question answered "Yes" or "Already shut off".
    Both questions are confirmed to be in the Financial Resource Strain domain.
    """
    fin = sdoh[sdoh["Domain_clean"] == "Financial Resource Strain"].copy()
    dn_lower = fin["DisplayName"].str.lower()

    # (a) Rent/mortgage question
    rent_mask = (
        dn_lower.str.contains("mortgage or rent") &
        (fin["AnswerText"] == "Yes")
    )
    patients_rent = _patient_set(fin, rent_mask)

    # (b) Utility / electric shutoff question
    shutoff_mask = (
        dn_lower.str.contains("electric") &
        (fin["AnswerText"].isin({"Yes", "Already shut off"}))
    )
    patients_shutoff = _patient_set(fin, shutoff_mask)

    diagnostics = {
        "n_positive_rent_mortgage_patients":   len(patients_rent),
        "n_positive_utility_shutoff_patients": len(patients_shutoff),
    }
    return patients_rent | patients_shutoff, diagnostics


def compute_ipv_need(sdoh: pd.DataFrame) -> set:
    """
    Domain: Intimate Partner Violence
    Positive: AnswerText == "Yes" on any IPV question.
    (Questions cover: humiliation/abuse, fear, physical harm, forced sexual activity.)
    """
    return _patient_set(
        sdoh,
        (sdoh["Domain_clean"] == "Intimate Partner Violence") &
        (sdoh["AnswerText"] == "Yes"),
    )


def compute_utilities_need(sdoh: pd.DataFrame) -> set:
    """
    Domain: Utilities
    Question: 'How hard is it for you to pay for the very basics like food, housing,
              medical care, and heating?'
    Positive: AnswerText in {"Somewhat hard", "Hard", "Very hard"}.
    (Excludes "Not hard at all" and "Not very hard" as non-need screens.)
    """
    return _patient_set(
        sdoh,
        (sdoh["Domain_clean"] == "Utilities") &
        (sdoh["AnswerText"].isin({"Somewhat hard", "Hard", "Very hard"})),
    )


def compute_elevated_stress_flag(sdoh: pd.DataFrame) -> set:
    """
    Domain: Stress
    Positive: AnswerText in {"Rather much", "Very much"}.
    This is a sensitivity/context flag and is NOT included in core_need_count.
    The 5-level scale is: Not at all / Only a little / To some extent /
                          Rather much / Very much.
    Threshold rationale: "Rather much" and above represents a meaningful
    stress burden; "To some extent" is left below the threshold as a
    conservative choice.
    """
    return _patient_set(
        sdoh,
        (sdoh["Domain_clean"] == "Stress") &
        (sdoh["AnswerText"].isin({"Rather much", "Very much"})),
    )


# ---------------------------------------------------------------------------
# 3. Build patient-level feature table
# ---------------------------------------------------------------------------

def build_patient_features(sdoh_clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One row per patient who appears in clean SDOH data.
    Columns: PatientDurableKey, has_clean_sdoh_response, [6 core flags],
             elevated_stress_flag, has_any_core_need, core_need_count,
             core_need_group.
    """
    all_patients = pd.DataFrame(
        {"PatientDurableKey": sdoh_clean["PatientDurableKey"].dropna().unique()}
    )
    all_patients["has_clean_sdoh_response"] = True

    diagnostics = {}

    # Compute each flag.
    t_patients = compute_transportation_need(sdoh_clean)
    f_patients = compute_food_insecurity_need(sdoh_clean)
    h_patients, h_diag = compute_housing_instability_need(sdoh_clean)
    fin_patients, fin_diag = compute_financial_strain_need(sdoh_clean)
    ipv_patients = compute_ipv_need(sdoh_clean)
    u_patients   = compute_utilities_need(sdoh_clean)
    s_patients   = compute_elevated_stress_flag(sdoh_clean)

    diagnostics.update(h_diag)
    diagnostics.update(fin_diag)

    flag_sets = {
        "transportation_need":      t_patients,
        "food_insecurity_need":     f_patients,
        "housing_instability_need": h_patients,
        "financial_strain_need":    fin_patients,
        "ipv_need":                 ipv_patients,
        "utilities_need":           u_patients,
        "elevated_stress_flag":     s_patients,
    }

    for col, patient_set in flag_sets.items():
        all_patients[col] = all_patients["PatientDurableKey"].isin(patient_set)
        diagnostics[f"n_patients_{col}"] = len(patient_set)

    # Aggregate fields.
    all_patients["core_need_count"] = (
        all_patients[CORE_NEED_FLAGS].sum(axis=1).astype(int)
    )
    all_patients["has_any_core_need"] = all_patients["core_need_count"] > 0
    all_patients["core_need_group"] = pd.cut(
        all_patients["core_need_count"],
        bins=[-1, 0, 1, 3, len(CORE_NEED_FLAGS)],
        labels=["0 needs", "1 need", "2-3 needs", "4+ needs"],
        right=True,
    ).astype(str)

    return all_patients, diagnostics


# ---------------------------------------------------------------------------
# 4. Join features to journey table
# ---------------------------------------------------------------------------

def join_to_journeys(
    jdf:      pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join patient SDOH need flags onto the journey table.
    Patients not in the features table get False / 0 / '0 needs'.
    """
    jdf = jdf.merge(features, on="PatientDurableKey", how="left")

    jdf["has_clean_sdoh_response"] = jdf["has_clean_sdoh_response"].fillna(False)
    jdf["has_any_core_need"]       = jdf["has_any_core_need"].fillna(False)
    jdf["core_need_count"]         = jdf["core_need_count"].fillna(0).astype(int)
    jdf["core_need_group"]         = jdf["core_need_group"].fillna("0 needs")

    for col in ALL_FLAGS:
        if col in jdf.columns:
            jdf[col] = jdf[col].fillna(False)

    return jdf


# ---------------------------------------------------------------------------
# 5. Aggregate output builders
# ---------------------------------------------------------------------------

NEED_GROUP_ORDER = ["0 needs", "1 need", "2-3 needs", "4+ needs"]

PRIMARY_TYPE_ORDER = [
    "one_visit", "hospital_involved", "long_journey", "lab_heavy",
    "imaging_heavy", "multi_department", "short_followup", "other_multi_encounter",
]


def save_patient_counts(features: pd.DataFrame, n_journey_patients: int) -> None:
    n_sdoh = len(features)
    rows = []
    for flag in ALL_FLAGS:
        n_pos = int(features[flag].sum())
        rows.append({
            "need_flag":                         flag,
            "n_patients_with_positive_screen":   n_pos,
            "n_patients_with_sdoh_response":     n_sdoh,
            "pct_of_sdoh_response_patients":     round(n_pos / n_sdoh * 100, 2) if n_sdoh else 0.0,
            "n_all_journey_patients":            n_journey_patients,
            "pct_of_all_journey_patients":       round(n_pos / n_journey_patients * 100, 2)
                                                  if n_journey_patients else 0.0,
        })
    pd.DataFrame(rows).to_csv(AUDIT_DIR / "core_sdoh_need_patient_counts.csv", index=False)


def save_journey_counts(jdf: pd.DataFrame) -> None:
    n_total = len(jdf)
    n_sdoh  = int(jdf["has_clean_sdoh_response"].sum())
    rows = []
    for flag in ALL_FLAGS:
        if flag not in jdf.columns:
            continue
        n_pos = int(jdf[flag].sum())
        rows.append({
            "need_flag":                        flag,
            "n_journeys_with_positive_screen":  n_pos,
            "n_journeys_with_sdoh_response":    n_sdoh,
            "pct_of_sdoh_response_journeys":    round(n_pos / n_sdoh * 100, 2) if n_sdoh else 0.0,
            "n_total_journeys":                 n_total,
            "pct_of_all_journeys":              round(n_pos / n_total * 100, 2) if n_total else 0.0,
        })
    pd.DataFrame(rows).to_csv(AUDIT_DIR / "core_sdoh_need_journey_counts.csv", index=False)


def save_type_by_need_group(jdf: pd.DataFrame) -> None:
    n_total = len(jdf)
    ct = (
        jdf.groupby(["primary_journey_type", "core_need_group"], dropna=False)
           .size()
           .reset_index(name="n_journeys")
    )
    type_totals  = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    group_totals = ct.groupby("core_need_group")["n_journeys"].transform("sum")
    ct["pct_of_type"]       = (ct["n_journeys"] / type_totals  * 100).round(2)
    ct["pct_of_need_group"] = (ct["n_journeys"] / group_totals * 100).round(2)

    type_sort  = {t: i for i, t in enumerate(PRIMARY_TYPE_ORDER)}
    group_sort = {g: i for i, g in enumerate(NEED_GROUP_ORDER)}
    ct["_ts"] = ct["primary_journey_type"].map(type_sort).fillna(99)
    ct["_gs"] = ct["core_need_group"].map(group_sort).fillna(99)
    (
        ct.sort_values(["_ts", "_gs"])
          .drop(columns=["_ts", "_gs"])
          .to_csv(AUDIT_DIR / "journey_type_by_core_need_group.csv", index=False)
    )


def save_type_by_any_need(jdf: pd.DataFrame) -> None:
    ct = (
        jdf.groupby(["primary_journey_type", "has_any_core_need"], dropna=False)
           .size()
           .reset_index(name="n_journeys")
    )
    type_totals = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    ct["pct_of_type"] = (ct["n_journeys"] / type_totals * 100).round(2)
    ct.sort_values(["primary_journey_type", "has_any_core_need"]).to_csv(
        AUDIT_DIR / "journey_type_by_any_core_need.csv", index=False
    )


def save_type_by_specific_need(jdf: pd.DataFrame) -> None:
    type_totals = jdf.groupby("primary_journey_type").size().rename("n_type_total")
    rows = []
    for flag in CORE_NEED_FLAGS:
        if flag not in jdf.columns:
            continue
        grp = (
            jdf[jdf[flag]]
               .groupby("primary_journey_type")
               .size()
               .reset_index(name="n_journeys_with_need")
        )
        grp["need_flag"] = flag
        grp = grp.merge(type_totals.reset_index(), on="primary_journey_type", how="right")
        grp["n_journeys_with_need"] = grp["n_journeys_with_need"].fillna(0).astype(int)
        grp["need_flag"]             = grp["need_flag"].fillna(flag)
        grp["pct_of_type_with_need"] = (
            grp["n_journeys_with_need"] / grp["n_type_total"] * 100
        ).round(2)
        rows.append(grp)

    if rows:
        result = pd.concat(rows, ignore_index=True)
        type_sort = {t: i for i, t in enumerate(PRIMARY_TYPE_ORDER)}
        flag_sort = {f: i for i, f in enumerate(CORE_NEED_FLAGS)}
        result["_ts"] = result["primary_journey_type"].map(type_sort).fillna(99)
        result["_fs"] = result["need_flag"].map(flag_sort).fillna(99)
        (
            result.sort_values(["_fs", "_ts"])
                  .drop(columns=["_ts", "_fs"])
                  [["need_flag", "primary_journey_type",
                    "n_type_total", "n_journeys_with_need", "pct_of_type_with_need"]]
                  .to_csv(AUDIT_DIR / "journey_type_by_specific_core_need.csv", index=False)
        )


def save_need_group_metrics(jdf: pd.DataFrame) -> None:
    gap_col = "first_to_second_gap_days"
    rows = []
    n_total = len(jdf)
    for group in NEED_GROUP_ORDER:
        grp = jdf[jdf["core_need_group"] == group]
        if grp.empty:
            continue
        n_grp     = len(grp)
        enc       = grp["n_encounters"]
        dur       = grp["journey_duration_days"]
        gap       = grp[gap_col].dropna() if gap_col in grp.columns else pd.Series(dtype=float)
        n_multi   = int((enc > 1).sum())
        n_hosp    = int(grp["hospital_involved_journey"].sum()) if "hospital_involved_journey" in grp.columns else 0

        rows.append({
            "core_need_group":                group,
            "n_journeys":                     n_grp,
            "pct_of_all_journeys":            round(n_grp / n_total * 100, 2),
            "median_n_encounters":            round(enc.median(), 1),
            "mean_n_encounters":              round(enc.mean(), 2),
            "pct_multi_encounter":            round(n_multi / n_grp * 100, 2) if n_grp else 0.0,
            "median_journey_duration_days":   int(dur.median()),
            "mean_journey_duration_days":     round(dur.mean(), 1),
            "p90_journey_duration_days":      int(dur.quantile(0.90)),
            "median_first_to_second_gap_days": round(gap.median(), 1) if len(gap) else np.nan,
            "pct_hospital_involved":          round(n_hosp / n_grp * 100, 2) if n_grp else 0.0,
        })

    pd.DataFrame(rows).to_csv(
        AUDIT_DIR / "core_need_group_metric_summary.csv", index=False
    )


def save_notes(features: pd.DataFrame, jdf: pd.DataFrame, diag: dict) -> None:
    n_sdoh   = len(features)
    n_any    = int(features["has_any_core_need"].sum())
    n_j      = len(jdf)
    n_j_any  = int(jdf["has_any_core_need"].sum())
    SEP      = "=" * 68
    DASH     = "-" * 68

    lines = [
        SEP,
        "  CORE SDOH NEED FLAGS — CODING RULES AND LIMITATIONS",
        "  Generated by: code/07_build_core_sdoh_need_flags.py",
        SEP,
        "",
        "  PURPOSE",
        DASH,
        "  This file documents the coding rules used to build each",
        "  patient-level need flag and notes what is NOT captured.",
        "",
        f"  Patients with clean SDOH responses : {n_sdoh:,}",
        f"  Patients with any core need        : {n_any:,}  ({round(n_any/n_sdoh*100,1) if n_sdoh else 0}%)",
        f"  Journeys with any core need        : {n_j_any:,}  ({round(n_j_any/n_j*100,1) if n_j else 0}%)",
        "",
        "  CODING RULES",
        DASH,
        "",
        "  1. transportation_need",
        "     Domain  : Transportation Needs",
        "     Trigger : AnswerText = 'Yes' to either transportation-barrier question.",
        "     Q1: '...kept you from medical appointments or from getting medications?'",
        "     Q2: '...kept you from meetings, work, or from getting things needed...'",
        f"     Patients flagged: {diag.get('n_patients_transportation_need', '?'):,}",
        "",
        "  2. food_insecurity_need",
        "     Domain  : Food Insecurity",
        "     Trigger : AnswerText in {'Sometimes true', 'Often true'} on either",
        "               food-security screening question (HRSN protocol).",
        "     Q1: '...worried that your food would run out...'",
        "     Q2: '...the food you bought just didn't last...'",
        f"     Patients flagged: {diag.get('n_patients_food_insecurity_need', '?'):,}",
        "",
        "  3. housing_instability_need",
        "     Domain  : Housing Stability",
        "     Trigger : (a) Homeless/shelter question answered 'Yes', OR",
        "               (b) Moved-count question has numeric answer >= 2.",
        "     Non-responses (Patient declined, unable to answer, *Unknown)",
        "     are treated as negative — they do NOT trigger the flag.",
        f"     Patients flagged via (a) homeless: {diag.get('n_positive_homeless_patients', '?'):,}",
        f"     Patients flagged via (b) moves >=2: {diag.get('n_positive_moved_gte2_patients', '?'):,}",
        f"     Patients flagged total (a OR b): {diag.get('n_patients_housing_instability_need', '?'):,}",
        "",
        "  4. financial_strain_need",
        "     Domain  : Financial Resource Strain",
        "     Trigger : (a) Rent/mortgage question answered 'Yes', OR",
        "               (b) Utility shutoff question answered 'Yes' or 'Already shut off'.",
        "     CONFIRMED: both questions are in the Financial Resource Strain domain.",
        f"     Patients flagged via (a) rent/mortgage: {diag.get('n_positive_rent_mortgage_patients', '?'):,}",
        f"     Patients flagged via (b) utility shutoff: {diag.get('n_positive_utility_shutoff_patients', '?'):,}",
        f"     Patients flagged total (a OR b): {diag.get('n_patients_financial_strain_need', '?'):,}",
        "",
        "  5. ipv_need",
        "     Domain  : Intimate Partner Violence",
        "     Trigger : AnswerText = 'Yes' to any IPV question.",
        "     Questions cover: humiliation/abuse, fear, physical harm, forced activity.",
        f"     Patients flagged: {diag.get('n_patients_ipv_need', '?'):,}",
        "",
        "  6. utilities_need",
        "     Domain  : Utilities",
        "     Question: 'How hard is it for you to pay for the very basics...'",
        "     Trigger : AnswerText in {'Somewhat hard', 'Hard', 'Very hard'}.",
        "     Excludes 'Not hard at all' and 'Not very hard' as non-need screens.",
        f"     Patients flagged: {diag.get('n_patients_utilities_need', '?'):,}",
        "",
        "  SENSITIVITY FLAG (not counted in core_need_count)",
        DASH,
        "",
        "  elevated_stress_flag",
        "     Domain  : Stress",
        "     Question: 'Do you feel stress - tense, restless, nervous, or anxious...'",
        "     Trigger : AnswerText in {'Rather much', 'Very much'}.",
        "     This flag is a separate context variable. It is NOT included in",
        "     core_need_count or has_any_core_need. Use it as an additional covariate.",
        f"     Patients flagged: {diag.get('n_patients_elevated_stress_flag', '?'):,}",
        "",
        "  FLAGS NOT BUILT IN THIS SCRIPT",
        DASH,
        "",
        "  The following domains require composite scoring or human review",
        "  and are intentionally deferred:",
        "",
        "  * Alcohol Use (AUDIT-C): requires summing Q1 + Q2 + Q3 into a",
        "    composite score and applying a threshold (typically >=3 women,",
        "    >=4 men). Not built here due to sex-specific thresholds.",
        "",
        "  * Depression (PHQ-2 / Edinburgh): PHQ-2 threshold = 3; Edinburgh",
        "    threshold varies by context. Both require total-score extraction",
        "    and clinical review of cutpoints.",
        "",
        "  * Physical Activity: requires combining days-per-week with",
        "    minutes-per-session to compute weekly volume.",
        "",
        "  * Social Connections: multi-dimensional; no single threshold",
        "    established for the Stormont Vail question set.",
        "",
        "  LIMITATIONS",
        DASH,
        "",
        "  1. FALSE NEGATIVE INTERPRETATION",
        "     A flag value of False does not mean the patient has no need.",
        "     They may not have been asked, may have declined to answer,",
        "     or may have incomplete SDOH screening coverage.",
        "",
        "  2. GRADUAL SDOH ROLLOUT",
        "     The SDOH survey was introduced gradually. Earlier encounters",
        "     have lower coverage than later ones. Need flags based on",
        "     pre-rollout patients will systematically under-count need.",
        "",
        "  3. PATIENT-LEVEL AGGREGATION",
        "     Flags aggregate any response across all encounters for a patient.",
        "     A patient who screened positive once in 2022 has the flag set",
        "     even if their situation improved by 2025. Frame results as",
        "     'ever recorded' rather than 'current'.",
        "",
        "  4. CAREFUL LANGUAGE",
        "     Use 'associated with', 'observed among', 'linked to', 'suggests'",
        "     — not 'causes', 'leads to', or 'results in'.",
        "     SDOH comparisons must always be framed as patterns among patients",
        "     with recorded responses, not the full population.",
        "",
        SEP,
    ]

    with open(AUDIT_DIR / "core_sdoh_need_flags_notes.txt", "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# 6. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(jdf: pd.DataFrame, features: pd.DataFrame) -> None:
    sep = "=" * 68
    n   = len(jdf)
    n_sdoh = len(features)

    print(f"\n{sep}")
    print("  CORE SDOH NEED FLAGS — AGGREGATE SUMMARY")
    print(sep)

    print(f"\n  Patients with clean SDOH responses : {n_sdoh:,}")
    n_any_pat = int(features["has_any_core_need"].sum())
    print(f"  Patients with any core need        : {n_any_pat:,}  "
          f"({round(n_any_pat/n_sdoh*100,1) if n_sdoh else 0}% of screened patients)")

    print(f"\n  Patient-level flag counts (among screened patients only):")
    print(f"  {'Flag':<32}  {'N patients':>12}  {'%':>6}")
    print(f"  {'-'*54}")
    for flag in ALL_FLAGS:
        n_flag = int(features[flag].sum())
        note   = " [sensitivity only]" if flag in SENSITIVITY_FLAGS else ""
        print(f"  {flag:<32}  {n_flag:>12,}  {round(n_flag/n_sdoh*100,1) if n_sdoh else 0:>5.1f}%{note}")

    print(f"\n  Core need group distribution (patient level):")
    grp_vc = features["core_need_group"].value_counts()
    for g in NEED_GROUP_ORDER:
        cnt = int(grp_vc.get(g, 0))
        print(f"  {g:<16}: {cnt:>10,}  ({round(cnt/n_sdoh*100,1) if n_sdoh else 0}%)")

    print(f"\n  Journey-level: core_need_group × primary_journey_type")
    print(f"  {'Type':<28}  {'0 needs':>10}  {'1 need':>8}  {'2-3':>6}  {'4+':>6}")
    print(f"  {'-'*62}")
    for ptype in PRIMARY_TYPE_ORDER:
        grp = jdf[jdf["primary_journey_type"] == ptype]
        if grp.empty:
            continue
        n_grp = len(grp)
        vals  = {g: int((grp["core_need_group"] == g).sum()) for g in NEED_GROUP_ORDER}
        print(
            f"  {ptype:<28}  "
            f"{vals['0 needs']:>10,}  "
            f"{vals['1 need']:>8,}  "
            f"{vals['2-3 needs']:>6,}  "
            f"{vals['4+ needs']:>6,}"
        )

    print(f"""
  NOTE: 'Patients with any core need' reflects positive screens only
  among patients with recorded SDOH responses. The majority of the
  journey cohort has no SDOH screening data and is not included in
  need comparisons. Results should be framed as patterns observed
  among screened patients.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 68)
    print("  BUILD CORE SDOH NEED FLAGS — DataFest 2026")
    print("=" * 68)

    print(f"\n  Loading and cleaning SDOH data …")
    sdoh_clean, sdoh_counts = load_clean_sdoh()
    print(f"  Retained clean SDOH rows : {sdoh_counts['n_rows_retained']:,}")
    print(f"  Unique patients          : {sdoh_counts['n_patients_in_clean_rows']:,}")

    print("  Computing need flags …")
    features, diagnostics = build_patient_features(sdoh_clean)
    del sdoh_clean
    print(f"  Patient-level features table: {len(features):,} rows")

    print(f"\n  Loading journey table from {IN_PARQUET} …")
    jdf = pd.read_parquet(IN_PARQUET)
    n_journey_patients = int(jdf["PatientDurableKey"].nunique())
    print(f"  Journeys: {len(jdf):,}  |  Unique patients: {n_journey_patients:,}")

    print("  Joining need flags to journey table …")
    jdf = join_to_journeys(jdf, features)
    n_any_j = int(jdf["has_any_core_need"].sum())
    print(f"  Journeys with any core need: {n_any_j:,}  "
          f"({round(n_any_j/len(jdf)*100,1)}%)")

    print(f"\n  Saving enriched journey table …")
    jdf.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")

    print("  Saving aggregate outputs …")
    save_patient_counts(features, n_journey_patients)
    save_journey_counts(jdf)
    save_type_by_need_group(jdf)
    save_type_by_any_need(jdf)
    save_type_by_specific_need(jdf)
    save_need_group_metrics(jdf)
    save_notes(features, jdf, diagnostics)

    for fname in [
        "core_sdoh_need_patient_counts.csv",
        "core_sdoh_need_journey_counts.csv",
        "journey_type_by_core_need_group.csv",
        "journey_type_by_any_core_need.csv",
        "journey_type_by_specific_core_need.csv",
        "core_need_group_metric_summary.csv",
        "core_sdoh_need_flags_notes.txt",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    print_summary(jdf, features)

    print(f"""
{'='*68}
  WARNING — RECORDED POSITIVE SCREENS ONLY
{'='*68}

These flags capture documented positive SDOH screens.  A False flag
does not indicate the absence of need — it may indicate that the
patient was not asked, declined, or lacked SDOH screening coverage.

Deferred flags not built here: Alcohol Use (AUDIT-C composite),
Depression (PHQ-2/Edinburgh), Physical Activity (weekly volume),
Social Connections (multi-dimensional).

Use careful language in all downstream analysis:
  "associated with", "observed among", "linked to", "suggests"
  — not "causes", "leads to", or "results in".
{'='*68}
""")


if __name__ == "__main__":
    main()
