"""
01_join_validation.py
---------------------
Cross-table join checks and diagnosis inflation check.

Run from project root AFTER 00_data_audit.py:
    python code/01_join_validation.py

All outputs go to outputs/audit/.
No raw patient-level rows are printed or exported.

KEY NOTE — why numeric normalization is needed
----------------------------------------------
keep_default_na=False preserves special strings but does NOT fix the int/float
dtype mismatch that arises when one CSV stores a key as int64 (e.g. 86333) and
another stores it as float64 (e.g. 86333.0).  A naive .astype(str) comparison
will see "86333" != "86333.0" and falsely report 0% match rates.

normalize_numeric_key() converts both sides of every join to pandas nullable
Int64 so that 86333 == 86333.0 after normalization.  Non-numeric tokens
(sentinel strings like "*Not Applicable") are kept as-is so they can be
counted separately but will naturally fail to match integer keys.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR  = Path("data")
AUDIT_DIR = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Same safe-read settings as 00_data_audit.py — preserves "NA" as a string.
def safe_read(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        usecols=usecols,
    )


# ---------------------------------------------------------------------------
# Numeric key normalization
# ---------------------------------------------------------------------------

def normalize_numeric_key(series: pd.Series) -> pd.Series:
    """
    Normalize a key column that may be int64, float64, or object (string).

    Steps:
      1. Parse with pd.to_numeric(errors="coerce") — non-numeric tokens become NaN.
      2. Convert whole-number floats (e.g. 86333.0) to integers.
      3. Return as pandas nullable Int64 so comparisons between formerly-int
         and formerly-float columns resolve correctly.

    Values that cannot be parsed (true sentinel strings, blank cells) become
    pd.NA and will not match any integer key — they are excluded from match
    counts automatically.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    # Round to remove floating-point noise before casting, then convert to Int64.
    # Only finite values survive; NaN/Inf stay as pd.NA in Int64.
    return numeric.round().astype("Int64")


# ---------------------------------------------------------------------------
# Generic 1-sided match-rate check
# ---------------------------------------------------------------------------

def join_match_rate(
    left_df:      pd.DataFrame,
    left_col:     str,
    right_df:     pd.DataFrame,
    right_col:    str,
    label:        str,
    exclude_vals: list[int] | None = None,
) -> dict:
    """
    Report what fraction of left_col values have a match in right_col.

    Both columns are normalized to Int64 before comparison so that int64 keys
    in one table and float64 keys in another resolve to the same value.

    exclude_vals: integer sentinel values to exclude BEFORE counting
                  (e.g. [-1] for PrimaryDiagnosisKey / provider keys).
    """
    left_keys  = normalize_numeric_key(left_df[left_col])
    right_keys = set(normalize_numeric_key(right_df[right_col]).dropna())

    # Exclude sentinels and NA
    non_null_mask = left_keys.notna()
    if exclude_vals:
        excl_set = set(exclude_vals)
        excl_mask = left_keys.isin(excl_set)
        eligible_mask = non_null_mask & ~excl_mask
        n_excluded = int(excl_mask.sum())
    else:
        eligible_mask = non_null_mask
        n_excluded    = 0

    n_null     = int((~non_null_mask).sum())
    eligible   = left_keys[eligible_mask]
    n_eligible = len(eligible)

    n_matched   = int(eligible.isin(right_keys).sum())
    n_unmatched = n_eligible - n_matched
    pct_matched   = round(n_matched   / n_eligible * 100, 2) if n_eligible > 0 else 0.0
    pct_unmatched = round(n_unmatched / n_eligible * 100, 2) if n_eligible > 0 else 0.0

    result = {
        "join":             label,
        "left_col":         left_col,
        "right_col":        right_col,
        "n_left_total":     len(left_keys),
        "n_null_or_unparseable": n_null,
        "n_excluded":       n_excluded,
        "n_eligible":       n_eligible,
        "n_matched":        n_matched,
        "pct_matched":      pct_matched,
        "n_unmatched":      n_unmatched,
        "pct_unmatched":    pct_unmatched,
        "n_right_distinct": len(right_keys),
    }

    flag = "OK" if pct_matched >= 95 else ("WARN" if pct_matched >= 80 else "LOW")
    print(f"\n  [{flag}] {label}")
    print(f"        left  : {left_col} ({len(left_keys):,} rows, "
          f"{n_excluded:,} sentinel-excluded, {n_null:,} null/unparseable)")
    print(f"        right : {right_col} ({len(right_keys):,} distinct values)")
    print(f"        match : {n_matched:,} / {n_eligible:,} eligible ({pct_matched}%)")
    if n_unmatched > 0:
        print(f"        UNMATCHED: {n_unmatched:,} rows ({pct_unmatched}%)")

    return result


# ---------------------------------------------------------------------------
# Diagnosis join inflation check
# ---------------------------------------------------------------------------

def check_diagnosis_inflation(enc: pd.DataFrame, diag: pd.DataFrame) -> dict:
    """
    Measure whether joining encounters → diagnosis on PrimaryDiagnosisKey
    inflates the row count (1-to-many join due to non-unique DiagnosisKey).

    Uses normalized Int64 keys so int/float mismatch does not mask real issues.
    """
    print(f"\n{'='*60}")
    print("  DIAGNOSIS JOIN INFLATION CHECK")
    print(f"{'='*60}")

    enc_key  = normalize_numeric_key(enc["PrimaryDiagnosisKey"])
    diag_key = normalize_numeric_key(diag["DiagnosisKey"])

    n_enc_total    = len(enc_key)
    n_enc_null     = int(enc_key.isna().sum())
    n_enc_minus1   = int((enc_key == -1).sum())
    # Eligible = not null, not sentinel -1
    eligible_mask  = enc_key.notna() & (enc_key != -1)
    n_enc_eligible = int(eligible_mask.sum())

    diag_key_counts    = diag_key.dropna().value_counts()
    n_diag_total_rows  = len(diag_key)
    n_diag_unique_keys = int(diag_key.dropna().nunique())
    n_diag_dup_keys    = int((diag_key_counts > 1).sum())

    print(f"\n  diagnosis.csv:")
    print(f"    Total rows            : {n_diag_total_rows:,}")
    print(f"    Unique DiagnosisKey   : {n_diag_unique_keys:,}")
    print(f"    Keys with > 1 row     : {n_diag_dup_keys:,}  "
          f"({'ISSUE — non-unique key' if n_diag_dup_keys > 0 else 'OK'})")

    # For each eligible encounter key, look up how many diagnosis rows it joins to.
    eligible_keys = enc_key[eligible_mask]
    merge_counts  = eligible_keys.map(diag_key_counts.to_dict()).fillna(0).astype(int)

    n_no_match    = int((merge_counts == 0).sum())
    n_one_match   = int((merge_counts == 1).sum())
    n_multi_match = int((merge_counts > 1).sum())
    n_rows_naive  = int(merge_counts.sum())
    row_inflation = n_rows_naive - n_enc_eligible

    print(f"\n  encounters.csv:")
    print(f"    Total rows                        : {n_enc_total:,}")
    print(f"    PrimaryDiagnosisKey = -1 (sentinel): {n_enc_minus1:,}")
    print(f"    PrimaryDiagnosisKey null/unparseable: {n_enc_null:,}")
    print(f"    Eligible for diagnosis join        : {n_enc_eligible:,}")
    print(f"\n  Join outcomes for eligible encounters:")
    print(f"    → no diagnosis match      : {n_no_match:,}")
    print(f"    → exactly 1 diagnosis row : {n_one_match:,}")
    print(f"    → 2+ diagnosis rows       : {n_multi_match:,}  "
          f"({'INFLATION RISK' if n_multi_match > 0 else 'OK'})")
    print(f"\n  Naive merge row count   : {n_rows_naive:,}")
    print(f"  Row inflation           : {row_inflation:,}  "
          f"({'ISSUE — naive merge inflates rows' if row_inflation > 0 else 'OK'})")

    fanout_dist = diag_key_counts.value_counts().sort_index()
    print(f"\n  Fanout distribution (n_diag_rows_per_key → n_keys_with_that_count):")
    print(fanout_dist.head(10).to_string())
    fanout_dist.to_frame("n_keys").to_csv(AUDIT_DIR / "join_dx_fanout_dist.csv")

    if n_diag_dup_keys > 0:
        print("\n  RECOMMENDATION: Do NOT use a naive merge on DiagnosisKey.")
        print("  Options:")
        print("    a) Deduplicate diagnosis.csv by DiagnosisKey before joining.")
        print("    b) Use PatientDurableKey + DiagnosisValue as the journey grouping key.")
        print("  Validate chosen strategy in 02_build_journeys.py.")

    summary = {
        "n_enc_total":         n_enc_total,
        "n_enc_null_key":      n_enc_null,
        "n_enc_minus1":        n_enc_minus1,
        "n_enc_eligible":      n_enc_eligible,
        "n_diag_total_rows":   n_diag_total_rows,
        "n_diag_unique_keys":  n_diag_unique_keys,
        "n_diag_dup_keys":     n_diag_dup_keys,
        "n_no_dx_match":       n_no_match,
        "n_one_dx_match":      n_one_match,
        "n_multi_dx_match":    n_multi_match,
        "n_rows_naive_merge":  n_rows_naive,
        "row_inflation":       row_inflation,
    }
    pd.DataFrame([summary]).to_csv(AUDIT_DIR / "join_dx_inflation.csv", index=False)
    print(f"\n  Saved: outputs/audit/join_dx_inflation.csv")
    return summary


# ---------------------------------------------------------------------------
# DiagnosisKey → value mapping uniqueness
# ---------------------------------------------------------------------------

def check_diagnosis_key_mapping(diag: pd.DataFrame):
    """
    For each DiagnosisKey, count distinct DiagnosisValue / GroupCode values.
    A key mapping to multiple values means it cannot safely anchor a journey.
    """
    print(f"\n{'='*60}")
    print("  DIAGNOSIS KEY → VALUE MAPPING CHECK")
    print(f"{'='*60}")

    for val_col in ["DiagnosisValue", "DiagnosisName", "GroupCode", "GroupName"]:
        if val_col not in diag.columns:
            continue
        mapping = (
            diag.groupby("DiagnosisKey")[val_col]
                .nunique()
                .rename("n_distinct_values")
                .reset_index()
        )
        n_ambiguous = int((mapping["n_distinct_values"] > 1).sum())
        print(f"\n  DiagnosisKey → {val_col}: "
              f"{n_ambiguous:,} keys map to multiple values "
              f"({'AMBIGUOUS' if n_ambiguous > 0 else 'OK'})")
        mapping.sort_values("n_distinct_values", ascending=False).head(20).to_csv(
            AUDIT_DIR / f"join_diag_key_to_{val_col.lower()}.csv", index=False
        )


# ---------------------------------------------------------------------------
# Social determinants join checks
# ---------------------------------------------------------------------------

def check_sdoh_joins(sdoh: pd.DataFrame, enc: pd.DataFrame):
    """
    Check match rates for both SDOH foreign keys using normalized Int64 keys.
    Also reports what fraction of encounter patients have any SDOH row.
    """
    print(f"\n{'='*60}")
    print("  SDOH JOIN CHECKS")
    print(f"{'='*60}")

    enc_enc_set = set(normalize_numeric_key(enc["EncounterKey"]).dropna())
    enc_pat_set = set(normalize_numeric_key(enc["PatientDurableKey"]).dropna())

    sdoh_enc_keys = normalize_numeric_key(sdoh["EncounterKey"])
    sdoh_pat_keys = normalize_numeric_key(sdoh["PatientDurableKey"])

    for label, sdoh_col, ref_set in [
        ("SDOH.EncounterKey → encounters.EncounterKey",           sdoh_enc_keys, enc_enc_set),
        ("SDOH.PatientDurableKey → encounters.PatientDurableKey", sdoh_pat_keys, enc_pat_set),
    ]:
        eligible  = sdoh_col.dropna()
        n_total   = len(sdoh_col)
        n_null    = int(sdoh_col.isna().sum())
        n_matched = int(eligible.isin(ref_set).sum())
        pct       = round(n_matched / len(eligible) * 100, 2) if len(eligible) > 0 else 0.0
        flag      = "OK" if pct >= 95 else ("WARN" if pct >= 80 else "LOW")
        print(f"\n  [{flag}] {label}")
        print(f"        {n_matched:,} / {len(eligible):,} non-null SDOH rows matched ({pct}%)")
        if n_null > 0:
            print(f"        {n_null:,} SDOH rows had null/unparseable key (excluded from rate)")

    # Fraction of encounter patients with any SDOH response
    sdoh_patient_set  = set(sdoh_pat_keys.dropna())
    n_enc_patients    = len(enc_pat_set)
    n_with_sdoh       = len(enc_pat_set & sdoh_patient_set)
    pct_with_sdoh     = round(n_with_sdoh / n_enc_patients * 100, 2) if n_enc_patients > 0 else 0.0
    print(f"\n  Encounter patients with ≥1 SDOH response: "
          f"{n_with_sdoh:,} / {n_enc_patients:,} ({pct_with_sdoh}%)")

    coverage = pd.DataFrame([{
        "n_encounter_patients":  n_enc_patients,
        "n_with_any_sdoh":       n_with_sdoh,
        "pct_with_any_sdoh":     pct_with_sdoh,
        "n_without_sdoh":        n_enc_patients - n_with_sdoh,
    }])
    coverage.to_csv(AUDIT_DIR / "sdoh_patient_coverage.csv", index=False)
    print(f"  Saved: outputs/audit/sdoh_patient_coverage.csv")


# ---------------------------------------------------------------------------
# Provider key join checks
# ---------------------------------------------------------------------------

def check_provider_joins(enc: pd.DataFrame, prov: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  PROVIDER KEY JOIN CHECKS")
    print(f"{'='*60}")

    results = []
    for enc_col, label in [
        ("ProviderDurableKey",         "encounters.ProviderDurableKey → providers.DurableKey"),
        ("AttendingProviderDurableKey", "encounters.AttendingProviderDurableKey → providers.DurableKey"),
        ("DischargeProviderDurableKey", "encounters.DischargeProviderDurableKey → providers.DurableKey"),
    ]:
        if enc_col not in enc.columns:
            print(f"\n  Column '{enc_col}' not found in encounters — skipping.")
            continue
        r = join_match_rate(enc, enc_col, prov, "DurableKey", label, exclude_vals=[-1])
        results.append(r)

    pd.DataFrame(results).to_csv(AUDIT_DIR / "join_provider_match_rates.csv", index=False)
    print(f"\n  Saved: outputs/audit/join_provider_match_rates.csv")


# ---------------------------------------------------------------------------
# Report raw dtype of key columns (diagnostic aid)
# ---------------------------------------------------------------------------

def report_key_dtypes(tables: dict[str, tuple[pd.DataFrame, list[str]]]):
    """Print the raw pandas dtype for every key column so dtype mismatches are visible."""
    print(f"\n{'='*60}")
    print("  KEY COLUMN DTYPES (pre-normalization)")
    print(f"{'='*60}")
    rows = []
    for tbl, (df, cols) in tables.items():
        for col in cols:
            if col in df.columns:
                rows.append({"table": tbl, "column": col, "dtype": str(df[col].dtype)})
                print(f"  {tbl}.{col}: {df[col].dtype}")
    pd.DataFrame(rows).to_csv(AUDIT_DIR / "join_key_dtypes.csv", index=False)
    print(f"\n  Saved: outputs/audit/join_key_dtypes.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  JOIN VALIDATION — DataFest 2026")
    print("="*60)

    print("\n  Loading tables (selected columns for joins) …")

    enc = safe_read(
        DATA_DIR / "encounters.csv",
        usecols=[
            "EncounterKey", "PatientDurableKey", "PrimaryDiagnosisKey",
            "DepartmentKey", "ProviderDurableKey",
            "AttendingProviderDurableKey", "DischargeProviderDurableKey",
        ],
    )
    patients = safe_read(DATA_DIR / "patients.csv",    usecols=["DurableKey"])
    diag     = safe_read(DATA_DIR / "diagnosis.csv")   # full table for mapping checks
    depts    = safe_read(DATA_DIR / "departments.csv", usecols=["DepartmentKey"])
    prov     = safe_read(DATA_DIR / "providers.csv",   usecols=["DurableKey"])
    sdoh     = safe_read(DATA_DIR / "social_determinants.csv",
                         usecols=["PatientDurableKey", "EncounterKey"])

    print(f"\n  encounters         : {enc.shape[0]:,} rows")
    print(f"  patients           : {patients.shape[0]:,} rows")
    print(f"  diagnosis          : {diag.shape[0]:,} rows")
    print(f"  departments        : {depts.shape[0]:,} rows")
    print(f"  providers          : {prov.shape[0]:,} rows")
    print(f"  social_determinants: {sdoh.shape[0]:,} rows")

    # Show raw key dtypes so any future int/float issue is immediately visible.
    report_key_dtypes({
        "encounters":          (enc,      ["EncounterKey", "PatientDurableKey",
                                            "PrimaryDiagnosisKey", "DepartmentKey",
                                            "ProviderDurableKey",
                                            "AttendingProviderDurableKey",
                                            "DischargeProviderDurableKey"]),
        "patients":            (patients, ["DurableKey"]),
        "diagnosis":           (diag,     ["DiagnosisKey"]),
        "departments":         (depts,    ["DepartmentKey"]),
        "providers":           (prov,     ["DurableKey"]),
        "social_determinants": (sdoh,     ["PatientDurableKey", "EncounterKey"]),
    })

    # ------------------------------------------------------------------
    # Core join match rates
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  CORE JOIN MATCH RATES")
    print(f"{'='*60}")

    core_results = []
    core_results.append(join_match_rate(
        enc, "PatientDurableKey",
        patients, "DurableKey",
        "encounters.PatientDurableKey → patients.DurableKey",
    ))
    core_results.append(join_match_rate(
        enc, "PrimaryDiagnosisKey",
        diag, "DiagnosisKey",
        "encounters.PrimaryDiagnosisKey → diagnosis.DiagnosisKey",
        exclude_vals=[-1],
    ))
    core_results.append(join_match_rate(
        enc, "DepartmentKey",
        depts, "DepartmentKey",
        "encounters.DepartmentKey → departments.DepartmentKey",
    ))
    pd.DataFrame(core_results).to_csv(AUDIT_DIR / "join_core_match_rates.csv", index=False)
    print(f"\n  Saved: outputs/audit/join_core_match_rates.csv")

    # ------------------------------------------------------------------
    # Provider joins (sentinel = -1)
    # ------------------------------------------------------------------
    check_provider_joins(enc, prov)

    # ------------------------------------------------------------------
    # Diagnosis inflation (uses normalized keys internally)
    # ------------------------------------------------------------------
    check_diagnosis_inflation(enc, diag)

    # ------------------------------------------------------------------
    # Diagnosis key → value ambiguity
    # ------------------------------------------------------------------
    check_diagnosis_key_mapping(diag)

    # ------------------------------------------------------------------
    # SDOH joins
    # ------------------------------------------------------------------
    check_sdoh_joins(sdoh, enc)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("""
============================================================
  JOIN VALIDATION COMPLETE
============================================================

Key outputs in outputs/audit/:
  join_key_dtypes.csv            — raw dtype of every key column (pre-normalization)
  join_core_match_rates.csv      — patient / diagnosis / department match rates
  join_provider_match_rates.csv  — three provider key match rates
  join_dx_inflation.csv          — row-inflation summary for diagnosis join
  join_dx_fanout_dist.csv        — distribution of diagnosis rows per key
  join_diag_key_to_*.csv         — DiagnosisKey → value ambiguity checks
  sdoh_patient_coverage.csv      — fraction of patients with SDOH responses

Next step:
  If join_dx_inflation.csv shows row_inflation > 0, decide dedup strategy
  before running 02_build_journeys.py.
============================================================
""")


if __name__ == "__main__":
    main()
