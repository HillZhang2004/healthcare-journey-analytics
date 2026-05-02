"""
03_build_journeys.py
--------------------
Build the primary patient journey table using clean one-to-one DiagnosisValue mappings.

Decision (from 02_diagnosis_mapping_audit.py):
  Strategy C — exclude all ambiguous DiagnosisKeys.
  Tiebreaking is not used because diagnosis_key_tiebreak_summary.csv shows
  0 clear plurality keys (100% of ambiguous keys are tied); any tiebreak
  would be arbitrary.

Journey definition:
  Journey = PatientDurableKey + DiagnosisValue over time.

Exclusion rules applied in order:
  1. PrimaryDiagnosisKey = -1 (no diagnosis assigned)
  2. PrimaryDiagnosisKey null / unparseable
  3. PrimaryDiagnosisKey maps to an ambiguous DiagnosisKey (2+ DiagnosisValues)
  4. PrimaryDiagnosisKey has no match in diagnosis.csv (unmatched)

Run from project root:
    python code/03_build_journeys.py

Outputs:
  outputs/tables/journeys_clean_diagnosisvalue.parquet   ← journey table (local only)
  outputs/audit/journey_build_counts.csv
  outputs/audit/journey_metric_summary.csv
  outputs/audit/journey_encounter_count_distribution.csv
  outputs/audit/journey_duration_distribution.csv
  outputs/audit/journey_one_visit_rate_by_groupname.csv
  outputs/audit/journey_top_diagnosis_groups.csv

No raw patient-level rows are printed or exported.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
AUDIT_DIR  = Path("outputs/audit")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Columns to load from encounters.csv (all confirmed present in the real file).
ENC_COLS = [
    "EncounterKey", "PatientDurableKey", "PrimaryDiagnosisKey",
    "Date", "Type", "VisitType", "VisitTypeDescription",
    "DepartmentKey", "ProviderDurableKey",
    "IsEdVisit", "IsHospitalAdmission", "IsInpatientAdmission",
    "IsObservation", "IsOutpatientFaceToFaceVisit",
]

# diagnosis.csv columns in the real file: DiagnosisKey, GroupName, GroupCode, DiagnosisName, DiagnosisValue
DIAG_VAL_COLS = ["DiagnosisValue", "DiagnosisName", "GroupCode", "GroupName"]

# Keyword sets for lab / imaging classification (heuristic, applied to
# combined Type + VisitTypeDescription text).
_LAB_PATTERN = re.compile(
    r"lab(?:oratory)?|patholog|specimen|blood\s+draw|urinalysis|"
    r"microbiol|hematol|serol|culture|venipuncture",
    re.IGNORECASE,
)
_IMAGING_PATTERN = re.compile(
    r"imaging|radiolog|x-?ray|mri|ultrasound|echocardio|"
    r"mammograph|nuclear\s+med|pet\s+scan|fluoroscop|angiograph|"
    r"\bct\b|bone\s+densit|dexa",
    re.IGNORECASE,
)

JOURNEY_COLS = ["PatientDurableKey", "DiagnosisValue"]


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
    """
    Normalize an ID column that may arrive as int64, float64, or object.
    Whole-number floats (86333.0) → integers (86333) via nullable Int64.
    Non-numeric tokens and blanks → pd.NA.
    """
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def flag_to_bool(series: pd.Series) -> pd.Series:
    """
    Convert an encounter flag column to boolean.
    Handles int (0/1), float (0.0/1.0), and string ('0','1','True','Yes').
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )


# ---------------------------------------------------------------------------
# 1. Build clean 1-to-1 diagnosis mapping
# ---------------------------------------------------------------------------

def build_clean_mapping(diag: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Return (mapping_df, counts_dict).

    mapping_df has one row per DiagnosisKey that maps uniquely to one
    DiagnosisValue.  Additional label columns (DiagnosisName, GroupCode,
    GroupName) are included only where they are also 1-to-1 for that key;
    otherwise they are left as NaN so no incorrect label is carried forward.
    """
    diag = diag.copy()
    diag["_key"] = normalize_numeric_key(diag["DiagnosisKey"])
    diag = diag[diag["_key"].notna()]

    n_total_rows = len(diag)
    n_total_keys = int(diag["_key"].nunique())

    # Count distinct values per key for each column of interest.
    fanout_agg = {"n_rows": ("_key", "count")}
    for col in DIAG_VAL_COLS:
        if col in diag.columns:
            fanout_agg[f"n_unique_{col}"] = (col, "nunique")

    fanout = diag.groupby("_key").agg(**fanout_agg).reset_index()

    if "n_unique_DiagnosisValue" not in fanout.columns:
        raise ValueError("DiagnosisValue column not found in diagnosis.csv")

    # Primary filter: keep only keys with exactly one DiagnosisValue.
    clean_mask  = fanout["n_unique_DiagnosisValue"] == 1
    clean_keys  = set(fanout.loc[clean_mask, "_key"])
    n_clean     = len(clean_keys)
    n_ambiguous = int((~clean_mask).sum())

    clean_rows = diag[diag["_key"].isin(clean_keys)]

    # DiagnosisValue is guaranteed unique per clean key — take first occurrence.
    mapping = (
        clean_rows.groupby("_key")["DiagnosisValue"]
                  .first()
                  .reset_index()
    )

    # Add each extra label column only where it is also 1-to-1 for that key.
    for col in ["DiagnosisName", "GroupCode", "GroupName"]:
        uc = f"n_unique_{col}"
        if col not in diag.columns or uc not in fanout.columns:
            continue
        col_1to1_keys = set(fanout.loc[fanout[uc] == 1, "_key"]) & clean_keys
        col_vals = (
            diag[diag["_key"].isin(col_1to1_keys)]
                .groupby("_key")[col]
                .first()
                .reset_index()
        )
        mapping = mapping.merge(col_vals, on="_key", how="left")

    mapping = mapping.rename(columns={"_key": "DiagnosisKey"})

    counts = {
        "n_total_diag_rows":  n_total_rows,
        "n_total_diag_keys":  n_total_keys,
        "n_clean_1to1_keys":  n_clean,
        "n_ambiguous_keys":   n_ambiguous,
        "pct_clean_keys":     round(n_clean / n_total_keys * 100, 2) if n_total_keys else 0.0,
    }
    return mapping, counts


# ---------------------------------------------------------------------------
# 2. Load and filter encounters
# ---------------------------------------------------------------------------

def load_encounters(path: Path) -> pd.DataFrame:
    """Load the subset of encounter columns needed for journey construction."""
    available = pd.read_csv(path, nrows=0, keep_default_na=False).columns.tolist()
    load_cols = [c for c in ENC_COLS if c in available]
    skipped   = [c for c in ENC_COLS if c not in available]
    if skipped:
        print(f"  NOTE: columns not in encounters.csv (skipped): {skipped}")
    return safe_read(path, usecols=load_cols)


def filter_encounters(
    enc: pd.DataFrame,
    clean_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the four exclusion rules and join the clean diagnosis mapping.
    Returns (filtered_df_with_diag_labels, counts_dict).
    """
    n_raw = len(enc)

    enc = enc.copy()
    enc["_dx_key"] = normalize_numeric_key(enc["PrimaryDiagnosisKey"])

    # Rule 1: PrimaryDiagnosisKey = -1
    mask_sentinel = enc["_dx_key"] == -1
    n_sentinel    = int(mask_sentinel.sum())
    enc = enc[~mask_sentinel]

    # Rule 2: null / unparseable key
    mask_null = enc["_dx_key"].isna()
    n_null    = int(mask_null.sum())
    enc = enc[~mask_null].copy()

    # Rules 3 & 4: ambiguous or unmatched keys — handled by inner join.
    mapping_keyed = clean_mapping.rename(columns={"DiagnosisKey": "_dx_key"})
    enc_joined    = enc.merge(mapping_keyed, on="_dx_key", how="inner")
    n_no_match    = len(enc) - len(enc_joined)
    n_retained    = len(enc_joined)

    counts = {
        "n_raw_encounters":              n_raw,
        "n_excluded_sentinel_minus1":    n_sentinel,
        "n_excluded_null_key":           n_null,
        "n_excluded_ambig_or_unmatched": n_no_match,
        "n_retained_for_journeys":       n_retained,
        "pct_retained":                  round(n_retained / n_raw * 100, 2) if n_raw else 0.0,
    }
    return enc_joined, counts


# ---------------------------------------------------------------------------
# 3. Parse dates
# ---------------------------------------------------------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the Date column.  Unparseable values become NaT."""
    df = df.copy()
    raw = df["Date"].replace("", np.nan) if "Date" in df.columns else pd.Series(
        dtype="object", index=df.index
    )
    df["date_parsed"] = pd.to_datetime(raw, errors="coerce")
    n_bad = int(df["date_parsed"].isna().sum())
    if n_bad:
        print(f"  NOTE: {n_bad:,} encounters have unparseable Date — excluded from gap metrics.")
    return df


# ---------------------------------------------------------------------------
# 4. Vectorized gap metrics
# ---------------------------------------------------------------------------

def compute_gap_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a frame with JOURNEY_COLS + date_parsed, return a journey-level
    DataFrame with first_to_second_gap_days, median_gap_days, max_gap_days.

    Uses vectorized shift() within each journey group — no groupby.apply().
    Journeys with only one datable encounter get NaN for all gap columns.
    """
    # Work on date-valid rows only.
    valid = (
        df[JOURNEY_COLS + ["EncounterKey", "date_parsed"]]
        .dropna(subset=["date_parsed"])
        .sort_values(JOURNEY_COLS + ["date_parsed"])
        .copy()
    )

    # Within-group previous date and ordinal rank.
    grp = valid.groupby(JOURNEY_COLS, sort=False)
    valid["_prev_date"] = grp["date_parsed"].shift(1)
    valid["_rank"]      = grp.cumcount()          # 0-indexed within journey
    valid["_gap_days"]  = (valid["date_parsed"] - valid["_prev_date"]).dt.days

    # first_to_second gap: gap at rank == 1 (i.e., the second encounter).
    second = (
        valid.loc[valid["_rank"] == 1, JOURNEY_COLS + ["_gap_days"]]
             .rename(columns={"_gap_days": "first_to_second_gap_days"})
    )

    # median and max gap: aggregate over all non-first-encounter gaps.
    gap_rows = valid.loc[valid["_rank"] > 0, JOURNEY_COLS + ["_gap_days"]]
    gap_agg  = (
        gap_rows.groupby(JOURNEY_COLS)["_gap_days"]
                .agg(median_gap_days="median", max_gap_days="max")
                .reset_index()
    )

    result = second.merge(gap_agg, on=JOURNEY_COLS, how="outer")
    return result


# ---------------------------------------------------------------------------
# 5. Build journey table
# ---------------------------------------------------------------------------

def build_journey_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate encounter-level data to one row per (PatientDurableKey, DiagnosisValue).
    """
    df = df.copy()

    # --- provider key: replace sentinel -1 with NaN so it is not counted ---
    df["_prov"] = normalize_numeric_key(df["ProviderDurableKey"]) if "ProviderDurableKey" in df.columns else pd.NA
    df.loc[df["_prov"] == -1, "_prov"] = pd.NA

    # --- lab / imaging flags from combined text ---
    text = pd.Series("", index=df.index)
    for col in ["Type", "VisitTypeDescription"]:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    df["_is_lab"]     = text.str.contains(_LAB_PATTERN)
    df["_is_imaging"] = text.str.contains(_IMAGING_PATTERN)

    # --- encounter-type flags ---
    flag_map = {
        "has_ed":                    "IsEdVisit",
        "has_hospital_admission":    "IsHospitalAdmission",
        "has_inpatient_admission":   "IsInpatientAdmission",
        "has_observation":           "IsObservation",
        "has_outpatient_facetoface": "IsOutpatientFaceToFaceVisit",
    }
    for dest, src in flag_map.items():
        df[f"_f_{dest}"] = flag_to_bool(df[src]) if src in df.columns else False

    # --- base aggregation ---
    agg_spec = {
        "first_encounter_date": ("date_parsed",  "min"),
        "last_encounter_date":  ("date_parsed",  "max"),
        "n_encounters":         ("EncounterKey",  "count"),
        "n_unique_departments": ("DepartmentKey", "nunique"),
        "n_unique_providers":   ("_prov",         "nunique"),
        "n_lab_encounters":     ("_is_lab",        "sum"),
        "n_imaging_encounters": ("_is_imaging",    "sum"),
    }

    if "Type" in df.columns:
        agg_spec["n_unique_types"] = ("Type", "nunique")

    for dest in flag_map:
        agg_spec[dest] = (f"_f_{dest}", "any")

    # Carry label columns through (same value per group — guaranteed by clean mapping).
    for lc in ["DiagnosisName", "GroupCode", "GroupName"]:
        if lc in df.columns:
            agg_spec[lc] = (lc, "first")

    journeys = df.groupby(JOURNEY_COLS).agg(**agg_spec).reset_index()

    # --- derived fields ---
    journeys["journey_duration_days"] = (
        (journeys["last_encounter_date"] - journeys["first_encounter_date"]).dt.days
    )
    journeys["has_follow_up"] = journeys["n_encounters"] > 1

    # --- gap metrics (separate vectorized pass) ---
    gap_df   = compute_gap_metrics(df[JOURNEY_COLS + ["EncounterKey", "date_parsed"]])
    journeys = journeys.merge(gap_df, on=JOURNEY_COLS, how="left")

    # Cast lab/imaging counts to int (sum of bool may be object on some pandas builds).
    for col in ["n_lab_encounters", "n_imaging_encounters"]:
        journeys[col] = journeys[col].fillna(0).astype(int)

    return journeys


# ---------------------------------------------------------------------------
# 6. Aggregate validation outputs
# ---------------------------------------------------------------------------

def save_validation_outputs(
    journeys:     pd.DataFrame,
    build_counts: dict,
    diag_counts:  dict,
) -> None:
    """Write six aggregate audit files.  No patient-level rows are saved."""

    n_journeys = len(journeys)

    # (a) Build counts
    all_counts = {**diag_counts, **build_counts, "n_journeys": n_journeys}
    pd.DataFrame([all_counts]).to_csv(AUDIT_DIR / "journey_build_counts.csv", index=False)

    # (b) Numeric metric summary (describe)
    num_cols = [
        c for c in journeys.select_dtypes(include=[np.number, "Int64"]).columns
        if c not in {"PatientDurableKey"}
    ]
    (
        journeys[num_cols]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
        .T.reset_index()
        .rename(columns={"index": "metric"})
        .to_csv(AUDIT_DIR / "journey_metric_summary.csv", index=False)
    )

    # (c) Encounter-count distribution (1, 2, …, 9, 10+)
    bucket_order = [str(i) for i in range(1, 10)] + ["10+"]
    enc_buckets  = journeys["n_encounters"].clip(upper=10).astype(int).astype(str)
    enc_buckets  = enc_buckets.replace("10", "10+")
    dist = (
        enc_buckets.value_counts()
                   .reindex(bucket_order, fill_value=0)
                   .reset_index()
    )
    dist.columns = ["n_encounters_bucket", "n_journeys"]
    dist["pct_journeys"] = (dist["n_journeys"] / n_journeys * 100).round(2)
    dist.to_csv(AUDIT_DIR / "journey_encounter_count_distribution.csv", index=False)

    # (d) Journey duration distribution (day buckets, multi-encounter only)
    dur_multi = journeys.loc[journeys["n_encounters"] > 1, "journey_duration_days"].dropna()
    dur_bins   = [0, 30, 90, 180, 365, 730, 1095, float("inf")]
    dur_labels = ["0–30d", "31–90d", "91–180d", "181–365d", "1–2y", "2–3y", "3y+"]
    dur_dist = (
        pd.cut(dur_multi, bins=dur_bins, labels=dur_labels,
               right=True, include_lowest=True)
          .value_counts()
          .reindex(dur_labels, fill_value=0)
          .reset_index()
    )
    dur_dist.columns = ["duration_bucket", "n_journeys_multi_enc"]
    dur_dist["pct_of_multi_enc_journeys"] = (
        dur_dist["n_journeys_multi_enc"] / len(dur_multi) * 100
    ).round(2) if len(dur_multi) else 0.0
    dur_dist.to_csv(AUDIT_DIR / "journey_duration_distribution.csv", index=False)

    # (e) Single-visit rate by GroupName
    if "GroupName" in journeys.columns:
        gn = (
            journeys.groupby("GroupName", dropna=False)
                    .agg(
                        n_journeys     =("n_encounters", "count"),
                        n_single_visit =("n_encounters", lambda x: (x == 1).sum()),
                        median_enc     =("n_encounters", "median"),
                    )
                    .reset_index()
        )
        gn["pct_single_visit"] = (gn["n_single_visit"] / gn["n_journeys"] * 100).round(2)
        gn.sort_values("n_journeys", ascending=False).to_csv(
            AUDIT_DIR / "journey_one_visit_rate_by_groupname.csv", index=False
        )

    # (f) Top diagnosis groups by journey count
    # Use the most specific available label (GroupName preferred for readability).
    for label_col in ["GroupName", "GroupCode", "DiagnosisValue"]:
        if label_col not in journeys.columns:
            continue
        top = (
            journeys[label_col]
            .value_counts(dropna=False)
            .head(30)
            .reset_index()
        )
        top.columns = [label_col, "n_journeys"]
        top["pct_journeys"] = (top["n_journeys"] / n_journeys * 100).round(2)
        top.to_csv(AUDIT_DIR / "journey_top_diagnosis_groups.csv", index=False)
        break


# ---------------------------------------------------------------------------
# 7. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(
    journeys:     pd.DataFrame,
    build_counts: dict,
    diag_counts:  dict,
) -> None:

    n_j = len(journeys)
    sep = "=" * 60

    print(f"\n{sep}")
    print("  JOURNEY TABLE — AGGREGATE SUMMARY")
    print(sep)

    print("\n  Diagnosis mapping:")
    print(f"    Total DiagnosisKey values     : {diag_counts['n_total_diag_keys']:>10,}")
    print(f"    Clean 1-to-1 keys retained    : {diag_counts['n_clean_1to1_keys']:>10,}  ({diag_counts['pct_clean_keys']}%)")
    print(f"    Ambiguous keys excluded       : {diag_counts['n_ambiguous_keys']:>10,}")

    print("\n  Encounter filtering:")
    print(f"    Raw encounters                : {build_counts['n_raw_encounters']:>10,}")
    print(f"    Excluded (key = -1)           : {build_counts['n_excluded_sentinel_minus1']:>10,}")
    print(f"    Excluded (null/unparseable)   : {build_counts['n_excluded_null_key']:>10,}")
    print(f"    Excluded (ambig/unmatched)    : {build_counts['n_excluded_ambig_or_unmatched']:>10,}")
    print(f"    Retained                      : {build_counts['n_retained_for_journeys']:>10,}  ({build_counts['pct_retained']}%)")

    print(f"\n  Journey table:")
    print(f"    Total journeys                : {n_j:>10,}")

    n_single = int((journeys["n_encounters"] == 1).sum())
    n_multi  = n_j - n_single
    print(f"    Single-encounter journeys     : {n_single:>10,}  ({round(n_single/n_j*100,1) if n_j else 0}%)")
    print(f"    Multi-encounter journeys      : {n_multi:>10,}  ({round(n_multi/n_j*100,1)  if n_j else 0}%)")

    if "journey_duration_days" in journeys.columns and n_multi > 0:
        dur = journeys.loc[journeys["n_encounters"] > 1, "journey_duration_days"].dropna()
        print(f"\n  Journey duration (multi-encounter, days):")
        print(f"    Median                        : {dur.median():>10.0f}")
        print(f"    Mean                          : {dur.mean():>10.0f}")
        print(f"    90th pct                      : {dur.quantile(0.90):>10.0f}")
        print(f"    Max                           : {dur.max():>10.0f}")

    if "first_to_second_gap_days" in journeys.columns:
        gap = journeys["first_to_second_gap_days"].dropna()
        if len(gap):
            print(f"\n  First-to-second encounter gap (days):")
            print(f"    Median                        : {gap.median():>10.0f}")
            print(f"    Mean                          : {gap.mean():>10.0f}")
            print(f"    90th pct                      : {gap.quantile(0.90):>10.0f}")

    for flag in ["has_ed", "has_hospital_admission", "has_inpatient_admission",
                 "has_observation", "has_outpatient_facetoface"]:
        if flag in journeys.columns:
            n_flag = int(journeys[flag].sum())
            print(f"    {flag:<36}: {n_flag:>8,}  ({round(n_flag/n_j*100,1) if n_j else 0}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  BUILD JOURNEYS — DataFest 2026")
    print("="*60)

    # -- Load diagnosis and build clean mapping --
    print("\n  Loading diagnosis.csv …")
    diag = safe_read(DATA_DIR / "diagnosis.csv")
    print(f"  diagnosis.csv: {diag.shape[0]:,} rows")

    print("  Building clean 1-to-1 diagnosis mapping …")
    clean_mapping, diag_counts = build_clean_mapping(diag)
    print(f"  Clean mapping: {len(clean_mapping):,} keys  "
          f"({diag_counts['pct_clean_keys']}% of total keys)")
    print(f"  Ambiguous keys excluded: {diag_counts['n_ambiguous_keys']:,}")
    del diag   # free memory

    # -- Load encounters --
    print("\n  Loading encounters.csv …")
    enc_raw = load_encounters(DATA_DIR / "encounters.csv")
    print(f"  encounters.csv: {enc_raw.shape[0]:,} rows")

    # -- Filter and join --
    print("  Filtering encounters and joining diagnosis labels …")
    enc_filtered, build_counts = filter_encounters(enc_raw, clean_mapping)
    del enc_raw
    print(f"  Retained: {build_counts['n_retained_for_journeys']:,} encounters "
          f"({build_counts['pct_retained']}%)")

    # -- Parse dates --
    print("  Parsing Date column …")
    enc_filtered = parse_dates(enc_filtered)

    # -- Build journey table --
    print("  Building journey-level metrics …")
    journeys = build_journey_table(enc_filtered)
    del enc_filtered
    print(f"  Journey table: {len(journeys):,} journeys across "
          f"{journeys['PatientDurableKey'].nunique():,} patients")

    # -- Save journey table (local parquet only, not printed) --
    out_path = TABLES_DIR / "journeys_clean_diagnosisvalue.parquet"
    journeys.to_parquet(out_path, index=False)
    print(f"  Saved journey table: {out_path}")

    # -- Save aggregate validation outputs --
    print("\n  Saving aggregate validation outputs …")
    save_validation_outputs(journeys, build_counts, diag_counts)
    for fname in [
        "journey_build_counts.csv",
        "journey_metric_summary.csv",
        "journey_encounter_count_distribution.csv",
        "journey_duration_distribution.csv",
        "journey_one_visit_rate_by_groupname.csv",
        "journey_top_diagnosis_groups.csv",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    # -- Console summary --
    print_summary(journeys, build_counts, diag_counts)

    print(f"""
{('='*60)}
  WARNING — VALIDATION SUMMARIES ONLY, NOT FINAL FINDINGS
{('='*60)}

The numbers above describe the journey table construction process.
They are not final project findings.

Before proceeding to analysis:
  [ ] Verify excluded encounter share does not bias the journey
      population (see journey_build_counts.csv).
  [ ] Confirm that single-encounter journey rates are plausible for
      observed GroupName categories (acute, screening, admin episodes
      can naturally have only one observed encounter).
  [ ] Check journey_metric_summary.csv for implausible values
      (negative durations, extreme gaps).
  [ ] Note left/right censoring: some journeys may have started before
      Jan 2022 or continue after Dec 2025.
  [ ] Use careful language throughout:
        "associated with" / "observed among" / "linked to" / "suggests"
        — never "causes" or "leads to".
  [ ] SDOH patterns must always be framed as findings among patients
      with recorded responses, not the full population.
{('='*60)}
""")


if __name__ == "__main__":
    main()
