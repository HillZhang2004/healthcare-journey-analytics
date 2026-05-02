"""
05_add_patient_sdoh_context.py
-------------------------------
Join patient background features and SDOH domain presence flags to the
classified journey table, then produce aggregate comparison outputs.

Inputs:
    outputs/tables/journeys_classified.parquet
    data/patients.csv
    data/social_determinants.csv

Outputs:
    outputs/tables/journeys_classified_with_patient_sdoh.parquet
    outputs/audit/patient_join_summary.csv
    outputs/audit/sdoh_domain_cleaning_summary.csv
    outputs/audit/journey_type_by_mychart.csv
    outputs/audit/journey_type_by_sdoh_recorded.csv
    outputs/audit/journey_type_by_sdoh_domain_presence.csv

Run from project root:
    python code/05_add_patient_sdoh_context.py

No raw patient-level or journey-level rows are printed or exported.
All printed output is aggregate only.

INTERPRETATION CAUTION
All SDOH patterns below describe observations among patients who have
recorded responses for a given domain. Patients without a recorded
response are NOT assumed to have no need — they may not have been
asked, may have declined to answer, or the survey may not yet have
been administered at their encounter. Use language such as
'associated with', 'observed among', or 'linked to' throughout.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True,  exist_ok=True)

IN_PARQUET  = TABLES_DIR / "journeys_classified.parquet"
OUT_PARQUET = TABLES_DIR / "journeys_classified_with_patient_sdoh.parquet"

PATIENT_FEATURES = [
    "PatientBirthYearBin",
    "MyChartStatus",
    "CensusBlockGroupFipsCode",
    "FirstRace",
    "OmbEthnicity",
    "OmbRace",
    "SmokingStatus",
    "VitalStatus",
]

# Canonical domain names after cleaning.
CANONICAL_DOMAINS = [
    "Alcohol Use",
    "Depression",
    "Financial Resource Strain",
    "Food Insecurity",
    "Housing Stability",
    "Intimate Partner Violence",
    "Physical Activity",
    "Social Connections",
    "Stress",
    "Transportation Needs",
    "Utilities",
]

# Map lowercase-stripped raw domain strings → canonical form.
# Covers both the confirmed misspelling and common casing variants.
DOMAIN_CORRECTION_MAP = {
    "alcohol use":                "Alcohol Use",
    "depression":                 "Depression",
    "financial resource strain":  "Financial Resource Strain",
    "food insecurity":            "Food Insecurity",
    "housing stability":          "Housing Stability",
    "intimate partner violence":  "Intimate Partner Violence",
    "intimate partner violance":  "Intimate Partner Violence",   # confirmed misspelling
    "physical activity":          "Physical Activity",
    "social connections":         "Social Connections",
    "stress":                     "Stress",
    "transportation needs":       "Transportation Needs",
    "utilities":                  "Utilities",
}


def _sdoh_col(domain: str) -> str:
    """Canonical domain → column name, e.g. 'Food Insecurity' → 'sdoh_food_insecurity'."""
    return "sdoh_" + domain.lower().replace(" ", "_")


SDOH_DOMAIN_COLS = [_sdoh_col(d) for d in CANONICAL_DOMAINS]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def safe_read(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV preserving special coded values ('NA', 'Unknown', etc.) as strings."""
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        usecols=usecols,
    )


def normalize_numeric_key(series: pd.Series) -> pd.Series:
    """Convert int64 / float64 / object key column to nullable Int64."""
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


# ---------------------------------------------------------------------------
# 1. Patient join
# ---------------------------------------------------------------------------

def load_and_join_patients(
    jdf: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Left-join patient background features to the journey table.
    Returns (enriched_df, summary_dict).
    """
    load_cols = ["DurableKey"] + PATIENT_FEATURES
    available = pd.read_csv(
        DATA_DIR / "patients.csv", nrows=0, keep_default_na=False
    ).columns.tolist()
    load_cols = [c for c in load_cols if c in available]
    missing_requested = [c for c in PATIENT_FEATURES if c not in available]
    if missing_requested:
        print(f"  NOTE: patient feature columns not in patients.csv: {missing_requested}")

    patients = safe_read(DATA_DIR / "patients.csv", usecols=load_cols)
    patients["DurableKey"] = normalize_numeric_key(patients["DurableKey"])
    patients = patients.rename(columns={"DurableKey": "PatientDurableKey"})

    n_journeys        = len(jdf)
    n_unique_patients = int(jdf["PatientDurableKey"].nunique())
    n_patient_rows    = len(patients)

    jdf = jdf.merge(patients, on="PatientDurableKey", how="left")

    # A matched journey is one where at least one patient feature is non-null.
    feat_present = [c for c in PATIENT_FEATURES if c in jdf.columns]
    if feat_present:
        matched_mask = jdf[feat_present[0]].notna() | False
        for c in feat_present[1:]:
            matched_mask = matched_mask | jdf[c].notna()
    else:
        matched_mask = pd.Series(False, index=jdf.index)

    n_matched   = int(matched_mask.sum())
    n_unmatched = n_journeys - n_matched
    pct_matched = round(n_matched / n_journeys * 100, 2) if n_journeys else 0.0

    summary = {
        "n_journeys":                 n_journeys,
        "n_unique_journey_patients":  n_unique_patients,
        "n_patient_rows_in_file":     n_patient_rows,
        "n_journeys_patient_matched": n_matched,
        "n_journeys_unmatched":       n_unmatched,
        "pct_patient_matched":        pct_matched,
    }

    # Per-feature null rate (after join) — included in same summary file.
    feat_null_rows = []
    for feat in feat_present:
        n_null = int(jdf[feat].isna().sum())
        n_blank = int((jdf[feat].astype(str).str.strip() == "").sum())
        feat_null_rows.append({
            "feature":      feat,
            "n_null":       n_null,
            "n_blank":      n_blank,
            "n_total":      n_journeys,
            "pct_missing":  round((n_null + n_blank) / n_journeys * 100, 2),
        })

    return jdf, summary, feat_null_rows


# ---------------------------------------------------------------------------
# 2. SDOH cleaning and patient-level feature construction
# ---------------------------------------------------------------------------

def clean_and_build_sdoh_features(
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Read social_determinants.csv, apply domain label corrections,
    exclude unclean rows, and build one row per PatientDurableKey
    with domain-presence flags.

    Returns (sdoh_features_df, cleaning_summary_dict, domain_counts_df).
    """
    sdoh = safe_read(DATA_DIR / "social_determinants.csv")

    # Normalize patient key — same logic as all other tables.
    sdoh["PatientDurableKey"] = normalize_numeric_key(sdoh["PatientDurableKey"])
    sdoh = sdoh[sdoh["PatientDurableKey"].notna()]

    n_total = len(sdoh)
    n_unique_patients_total = int(sdoh["PatientDurableKey"].nunique())

    # --- Domain cleaning ---
    # Preserve the raw domain for before/after comparison.
    sdoh["Domain_raw"]   = sdoh["Domain"].astype(str).str.strip()
    sdoh["Domain_lower"] = sdoh["Domain_raw"].str.lower()

    # Apply correction map; unknown lowercased values get title-cased as fallback.
    sdoh["Domain_clean"] = (
        sdoh["Domain_lower"]
        .map(DOMAIN_CORRECTION_MAP)
        .fillna(sdoh["Domain_raw"].str.title())
    )

    raw_domain_counts = sdoh["Domain_raw"].value_counts(dropna=False)
    n_raw_distinct    = int(sdoh["Domain_raw"].nunique(dropna=False))

    # --- Exclusion rules ---
    mask_unspecified = sdoh["DisplayName"] == "*Unspecified"
    mask_domain_na   = sdoh["Domain_raw"]  == "NA"
    n_excl_unspec    = int(mask_unspecified.sum())
    n_excl_na_domain = int(mask_domain_na.sum())
    # Rows excluded by either rule (may overlap)
    n_excl_either    = int((mask_unspecified | mask_domain_na).sum())

    sdoh_clean = sdoh[~mask_unspecified & ~mask_domain_na].copy()
    n_retained = len(sdoh_clean)

    clean_domain_counts = sdoh_clean["Domain_clean"].value_counts(dropna=False)
    n_clean_distinct    = int(sdoh_clean["Domain_clean"].nunique(dropna=False))
    n_patients_retained = int(sdoh_clean["PatientDurableKey"].nunique())

    # --- Domain counts table (before and after, for audit file) ---
    domain_audit = (
        raw_domain_counts.rename("n_rows_before")
        .reset_index()
        .rename(columns={"index": "Domain_raw"})
    )
    domain_audit.to_csv(AUDIT_DIR / "_sdoh_raw_domain_counts.csv", index=False)

    clean_domain_audit = (
        clean_domain_counts.rename("n_rows_after")
        .reset_index()
        .rename(columns={"index": "Domain_clean"})
    )
    clean_domain_audit.to_csv(AUDIT_DIR / "_sdoh_clean_domain_counts.csv", index=False)

    cleaning_summary = {
        "n_sdoh_rows_total":                      n_total,
        "n_patients_in_sdoh_file":                n_unique_patients_total,
        "n_distinct_domain_values_before":        n_raw_distinct,
        "n_rows_excluded_displayname_unspecified": n_excl_unspec,
        "n_rows_excluded_domain_na":               n_excl_na_domain,
        "n_rows_excluded_either":                  n_excl_either,
        "n_rows_retained":                         n_retained,
        "n_distinct_domain_values_after":          n_clean_distinct,
        "n_patients_with_clean_sdoh_response":     n_patients_retained,
    }

    # --- Build one row per patient with domain-presence flags ---
    # Use crosstab for efficiency: rows=patients, cols=domains.
    # Only patients with at least one retained row appear in sdoh_clean.
    # drop_duplicates first so a patient's repeated responses in the same
    # domain count as 1 (presence, not frequency).
    dedup = sdoh_clean.drop_duplicates(subset=["PatientDurableKey", "Domain_clean"])

    # Only use rows whose Domain_clean is a known canonical domain.
    dedup_known = dedup[dedup["Domain_clean"].isin(CANONICAL_DOMAINS)]

    if len(dedup_known) == 0:
        # Edge case: no clean domain data at all.
        features = pd.DataFrame(columns=["PatientDurableKey"] + SDOH_DOMAIN_COLS +
                                         ["has_any_sdoh_response", "n_sdoh_domains_recorded"])
    else:
        cross = (
            dedup_known
            .assign(_val=1)
            .pivot_table(
                index="PatientDurableKey",
                columns="Domain_clean",
                values="_val",
                aggfunc="first",
                fill_value=0,
            )
            .gt(0)
            .reindex(columns=CANONICAL_DOMAINS, fill_value=False)
            .reset_index()
        )
        # Rename domain columns to safe column names.
        cross.columns = (
            ["PatientDurableKey"] +
            [_sdoh_col(c) for c in cross.columns[1:]]
        )
        cross["n_sdoh_domains_recorded"] = cross[SDOH_DOMAIN_COLS].sum(axis=1).astype(int)
        cross["has_any_sdoh_response"]   = True
        features = cross

    # Domain counts table for the audit file.
    domain_counts_df = pd.concat([domain_audit, clean_domain_audit], axis=1)

    return features, cleaning_summary, clean_domain_audit


# ---------------------------------------------------------------------------
# 3. Join SDOH features
# ---------------------------------------------------------------------------

def join_sdoh_features(
    jdf:      pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join patient SDOH feature flags to the journey table."""
    jdf = jdf.merge(features, on="PatientDurableKey", how="left")

    # Fill NaN for patients with no retained SDOH data.
    jdf["has_any_sdoh_response"]   = jdf["has_any_sdoh_response"].fillna(False)
    jdf["n_sdoh_domains_recorded"] = jdf["n_sdoh_domains_recorded"].fillna(0).astype(int)
    for col in SDOH_DOMAIN_COLS:
        if col in jdf.columns:
            jdf[col] = jdf[col].fillna(False)

    return jdf


# ---------------------------------------------------------------------------
# 4. Aggregate comparison outputs
# ---------------------------------------------------------------------------

def save_journey_type_by_mychart(jdf: pd.DataFrame) -> None:
    """
    For each (primary_journey_type, MyChartStatus): n_journeys,
    pct within type, pct within MyChartStatus group.
    """
    if "MyChartStatus" not in jdf.columns:
        print("  NOTE: MyChartStatus not in journey table — skipping journey_type_by_mychart.csv.")
        return

    ct = (
        jdf.groupby(["primary_journey_type", "MyChartStatus"], dropna=False)
           .size()
           .reset_index(name="n_journeys")
    )
    # pct within primary_journey_type
    type_totals = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    ct["pct_of_type"] = (ct["n_journeys"] / type_totals * 100).round(2)
    # pct within MyChartStatus group
    mc_totals = ct.groupby("MyChartStatus")["n_journeys"].transform("sum")
    ct["pct_of_mychart_group"] = (ct["n_journeys"] / mc_totals * 100).round(2)

    ct.sort_values(["primary_journey_type", "n_journeys"], ascending=[True, False]).to_csv(
        AUDIT_DIR / "journey_type_by_mychart.csv", index=False
    )


def save_journey_type_by_sdoh_recorded(jdf: pd.DataFrame) -> None:
    """
    For each (primary_journey_type, has_any_sdoh_response): n_journeys, pct of type.
    """
    ct = (
        jdf.groupby(["primary_journey_type", "has_any_sdoh_response"], dropna=False)
           .size()
           .reset_index(name="n_journeys")
    )
    type_totals = ct.groupby("primary_journey_type")["n_journeys"].transform("sum")
    ct["pct_of_type"] = (ct["n_journeys"] / type_totals * 100).round(2)
    ct.sort_values(["primary_journey_type", "has_any_sdoh_response"]).to_csv(
        AUDIT_DIR / "journey_type_by_sdoh_recorded.csv", index=False
    )


def save_journey_type_by_sdoh_domain(jdf: pd.DataFrame) -> None:
    """
    For each primary_journey_type and each SDOH domain, report:
      - n_journeys linked to a patient with a response in that domain
      - pct of journeys in that type with the domain recorded

    This is a descriptive prevalence table, not a causal claim.
    """
    rows = []
    n_by_type = jdf.groupby("primary_journey_type").size().rename("n_type_total")

    for domain, col in zip(CANONICAL_DOMAINS, SDOH_DOMAIN_COLS):
        if col not in jdf.columns:
            continue
        grp = (
            jdf[jdf[col]]                          # journeys with domain present
               .groupby("primary_journey_type")
               .size()
               .reset_index(name="n_journeys_with_domain")
        )
        grp["sdoh_domain"] = domain
        grp = grp.merge(n_by_type.reset_index(), on="primary_journey_type", how="right")
        grp["n_journeys_with_domain"] = grp["n_journeys_with_domain"].fillna(0).astype(int)
        grp["pct_of_type_with_domain"] = (
            grp["n_journeys_with_domain"] / grp["n_type_total"] * 100
        ).round(2)
        grp["sdoh_domain"] = grp["sdoh_domain"].fillna(domain)
        rows.append(grp)

    if rows:
        result = pd.concat(rows, ignore_index=True)
        result = result[["primary_journey_type", "sdoh_domain",
                          "n_type_total", "n_journeys_with_domain",
                          "pct_of_type_with_domain"]]
        result.sort_values(["sdoh_domain", "primary_journey_type"]).to_csv(
            AUDIT_DIR / "journey_type_by_sdoh_domain_presence.csv", index=False
        )


# ---------------------------------------------------------------------------
# 5. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(jdf: pd.DataFrame, patient_summary: dict) -> None:
    sep = "=" * 64
    n   = len(jdf)

    print(f"\n{sep}")
    print("  PATIENT + SDOH CONTEXT — AGGREGATE SUMMARY")
    print(sep)

    print(f"\n  Patient join:")
    print(f"    Journeys                  : {n:,}")
    print(f"    Patient-matched           : {patient_summary['n_journeys_patient_matched']:,}  "
          f"({patient_summary['pct_patient_matched']}%)")
    print(f"    Unmatched                 : {patient_summary['n_journeys_unmatched']:,}")

    if "MyChartStatus" in jdf.columns:
        print(f"\n  MyChartStatus distribution across journeys:")
        mc_vc = jdf["MyChartStatus"].value_counts(dropna=False)
        for val, cnt in mc_vc.items():
            print(f"    {str(val):<30}: {cnt:>10,}  ({cnt/n*100:.1f}%)")

    n_any_sdoh = int(jdf["has_any_sdoh_response"].sum())
    print(f"\n  Journeys with any clean SDOH response   : {n_any_sdoh:,}  ({n_any_sdoh/n*100:.1f}%)")
    print(f"  Journeys without SDOH data              : {n - n_any_sdoh:,}  ({(n-n_any_sdoh)/n*100:.1f}%)")

    print(f"\n  Journey type × has_any_sdoh_response:")
    print(f"  {'Type':<28}  {'With SDOH':>10}  {'%':>6}  {'Without':>10}  {'%':>6}")
    print(f"  {'-'*66}")
    for ptype, grp in jdf.groupby("primary_journey_type", sort=False):
        n_grp  = len(grp)
        n_with = int(grp["has_any_sdoh_response"].sum())
        n_wo   = n_grp - n_with
        print(f"  {ptype:<28}  {n_with:>10,}  {n_with/n_grp*100:>5.1f}%"
              f"  {n_wo:>10,}  {n_wo/n_grp*100:>5.1f}%")

    print(f"\n  SDOH domain presence (% of all journeys, among those with patient match):")
    for domain, col in zip(CANONICAL_DOMAINS, SDOH_DOMAIN_COLS):
        if col in jdf.columns:
            n_domain = int(jdf[col].sum())
            print(f"    {domain:<36}: {n_domain:>10,}  ({n_domain/n*100:.1f}%)")

    print(f"""
  NOTE: SDOH percentages above reflect journeys linked to a patient
  who had any recorded response in that domain at any encounter in
  social_determinants.csv. Patients without a recorded response are
  NOT assumed to have no need; they may not have been asked.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 64)
    print("  ADD PATIENT + SDOH CONTEXT — DataFest 2026")
    print("=" * 64)

    # Load journey table
    print(f"\n  Loading {IN_PARQUET} …")
    jdf = pd.read_parquet(IN_PARQUET)
    print(f"  Journeys: {len(jdf):,}  |  Columns: {len(jdf.columns)}")

    # --- Patient join ---
    print("\n  Joining patient features …")
    jdf, patient_summary, feat_null_rows = load_and_join_patients(jdf)
    print(f"  Patient match rate: {patient_summary['pct_patient_matched']}%")

    # Save patient join summary
    summary_rows = [{"metric": k, "value": v} for k, v in patient_summary.items()]
    pd.DataFrame(summary_rows).to_csv(AUDIT_DIR / "patient_join_summary.csv", index=False)
    # Append per-feature null rates to same file.
    feat_df = pd.DataFrame(feat_null_rows)
    if not feat_df.empty:
        feat_df.to_csv(AUDIT_DIR / "patient_feature_null_rates.csv", index=False)
    print(f"  Saved: outputs/audit/patient_join_summary.csv")
    print(f"  Saved: outputs/audit/patient_feature_null_rates.csv")

    # --- SDOH features ---
    print("\n  Building SDOH domain features …")
    sdoh_features, cleaning_summary, domain_counts_df = clean_and_build_sdoh_features()
    n_sdoh_patients = len(sdoh_features)
    print(f"  Patients with clean SDOH responses: "
          f"{cleaning_summary['n_patients_with_clean_sdoh_response']:,}")
    print(f"  Rows excluded (DisplayName=*Unspecified): "
          f"{cleaning_summary['n_rows_excluded_displayname_unspecified']:,}")
    print(f"  Rows excluded (Domain=NA)               : "
          f"{cleaning_summary['n_rows_excluded_domain_na']:,}")

    # Save SDOH cleaning summary
    cleaning_rows = [{"metric": k, "value": v} for k, v in cleaning_summary.items()]
    pd.DataFrame(cleaning_rows).to_csv(
        AUDIT_DIR / "sdoh_domain_cleaning_summary.csv", index=False
    )
    print(f"  Saved: outputs/audit/sdoh_domain_cleaning_summary.csv")

    # --- Join SDOH features ---
    print("\n  Joining SDOH features to journey table …")
    jdf = join_sdoh_features(jdf, sdoh_features)
    n_any_sdoh = int(jdf["has_any_sdoh_response"].sum())
    print(f"  Journeys with any SDOH response: {n_any_sdoh:,}  ({n_any_sdoh/len(jdf)*100:.1f}%)")

    # --- Save enriched parquet ---
    jdf.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  Saved enriched journey table: {OUT_PARQUET}")

    # --- Aggregate comparison outputs ---
    print("\n  Saving aggregate comparison outputs …")
    save_journey_type_by_mychart(jdf)
    save_journey_type_by_sdoh_recorded(jdf)
    save_journey_type_by_sdoh_domain(jdf)
    for fname in [
        "journey_type_by_mychart.csv",
        "journey_type_by_sdoh_recorded.csv",
        "journey_type_by_sdoh_domain_presence.csv",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    # --- Console summary ---
    print_summary(jdf, patient_summary)

    print(f"""
{'='*64}
  WARNING — DESCRIPTIVE PATTERNS ONLY
{'='*64}

Results above describe structural differences in observed journey
patterns across patient groups. They do not establish causation.

Key limitations to state in any downstream analysis:
  - SDOH responses are not available for all patients. Comparisons
    are among patients with recorded responses only.
  - SDOH survey rollout was gradual; earlier encounters are less
    likely to have domain coverage.
  - MyChartStatus reflects enrollment status at the time of data
    extract, not necessarily at the time of each encounter.
  - PatientBirthYearBin is binned, not exact age.
  - SmokingStatus reflects last known status, not encounter-level.
  - CensusBlockGroupFipsCode may be suppressed for some patients.

Use careful language: 'associated with', 'observed among',
'linked to', 'suggests' — not 'causes' or 'leads to'.
{'='*64}
""")


if __name__ == "__main__":
    main()
