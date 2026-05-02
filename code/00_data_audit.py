"""
00_data_audit.py
----------------
Per-table shape, dtype, missing-value, and special-coded-value audit.

Run from project root:
    python code/00_data_audit.py

All outputs go to outputs/audit/.
No raw patient-level rows are printed or exported.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths — keep relative so the script works from the project root.
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
AUDIT_DIR  = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Special coded values we must NOT silently convert to NaN.
# We use keep_default_na=False, na_values=[] when reading every CSV so that
# strings like "NA", "Unknown", "*Not Applicable", and empty strings are
# preserved as-is and counted separately here.
# ---------------------------------------------------------------------------
SPECIAL_CODES = ["Unknown", "*Unknown", "Unspecified", "*Unspecified",
                 "NA", "*Not Applicable", ""]

# Columns we will treat as date strings (parsed after initial read).
DATE_COLS = {
    "encounters":  ["Date", "AdmissionInstant", "DischargeInstant"],
    "patients":    [],
    "diagnosis":   [],
    "departments": [],
    "providers":   [],
    "social_determinants": [],
    "tigercensuscodes": [],
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_read(path: Path) -> pd.DataFrame:
    """Read a CSV preserving ALL special coded values as strings."""
    return pd.read_csv(
        path,
        keep_default_na=False,   # do not treat "NA", "NaN", etc. as missing
        na_values=[],            # no additional strings mapped to NaN
        low_memory=False,
    )


def count_special_codes(series: pd.Series) -> dict:
    """Return counts of each special coded value in a Series."""
    counts = {}
    for code in SPECIAL_CODES:
        n = (series == code).sum()
        if n > 0:
            counts[code if code != "" else "<blank>"] = int(n)
    return counts


def top_value_counts(series: pd.Series, n: int = 10) -> pd.Series:
    """Return top-n value counts for a Series (safe — no raw rows)."""
    return series.value_counts(dropna=False).head(n)


def is_likely_date_col(col_name: str) -> bool:
    """Heuristic: column name contains a date-related keyword."""
    keywords = ["date", "instant", "time", "year", "month", "day", "hour", "minute"]
    return any(k in col_name.lower() for k in keywords)


def date_range_summary(df: pd.DataFrame, col: str) -> dict:
    """Parse a date column and return min/max/n_valid/n_unparsed."""
    raw = df[col].replace("", np.nan)
    raw = raw.replace(SPECIAL_CODES, np.nan)
    parsed = pd.to_datetime(raw, errors="coerce")
    valid  = parsed.dropna()
    return {
        "col":        col,
        "n_valid":    int(valid.shape[0]),
        "n_unparsed": int(parsed.isna().sum()),
        "min":        str(valid.min()) if not valid.empty else "n/a",
        "max":        str(valid.max()) if not valid.empty else "n/a",
    }


def audit_table(name: str, df: pd.DataFrame, key_cols: list[str]) -> dict:
    """
    Run the full per-table audit. Returns a summary dict and writes
    detailed CSVs to AUDIT_DIR.
    """
    print(f"\n{'='*60}")
    print(f"  TABLE: {name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Shape
    # ------------------------------------------------------------------
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Column names + dtypes
    # ------------------------------------------------------------------
    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype":  df.dtypes.astype(str).values,
    })
    print("\n  Column dtypes:")
    print(dtype_df.to_string(index=False))
    dtype_df.to_csv(AUDIT_DIR / f"{name}_dtypes.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Missing-value summary
    #    We use pd.isna() which catches only genuine NaN/None after our
    #    safe_read, because special-coded strings are still strings here.
    # ------------------------------------------------------------------
    missing = (
        df.isna()
          .sum()
          .rename("n_null")
          .reset_index()
          .rename(columns={"index": "column"})
    )
    missing["pct_null"] = (missing["n_null"] / df.shape[0] * 100).round(2)
    missing = missing[missing["n_null"] > 0]
    if missing.empty:
        print("\n  Missing (NaN/None): none detected after safe read.")
    else:
        print(f"\n  Missing (NaN/None): {missing.shape[0]} column(s) have nulls.")
        print(missing.to_string(index=False))
    missing.to_csv(AUDIT_DIR / f"{name}_missing.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Special coded-value counts across all columns
    # ------------------------------------------------------------------
    special_rows = []
    for col in df.columns:
        col_counts = count_special_codes(df[col].astype(str))
        for code, n in col_counts.items():
            special_rows.append({"column": col, "coded_value": code, "count": n})
    special_df = pd.DataFrame(special_rows)
    if special_df.empty:
        print("\n  Special coded values: none found.")
    else:
        print(f"\n  Special coded values ({special_df.shape[0]} column-code combinations):")
        print(special_df.sort_values("count", ascending=False).to_string(index=False))
    special_df.to_csv(AUDIT_DIR / f"{name}_special_codes.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Duplicate key check
    # ------------------------------------------------------------------
    dup_summary = {}
    for kc in key_cols:
        if kc not in df.columns:
            print(f"\n  Key column '{kc}' not found in {name}.")
            continue
        n_total = df.shape[0]
        n_unique = df[kc].nunique(dropna=False)
        n_dup    = int(df.duplicated(subset=[kc]).sum())
        dup_summary[kc] = {"total": n_total, "unique": n_unique, "duplicated_rows": n_dup}
        status = "DUPLICATE KEYS DETECTED" if n_dup > 0 else "unique"
        print(f"\n  Key '{kc}': {n_unique:,} unique / {n_total:,} rows — {status}")

    if len(key_cols) > 1:
        n_dup_composite = int(df.duplicated(subset=key_cols, keep=False).sum())
        status = "DUPLICATE COMPOSITE KEYS" if n_dup_composite > 0 else "unique"
        print(f"\n  Composite key {key_cols}: {n_dup_composite:,} duplicated rows — {status}")

    pd.DataFrame(dup_summary).T.to_csv(AUDIT_DIR / f"{name}_key_check.csv")

    # ------------------------------------------------------------------
    # 6. Date column ranges
    # ------------------------------------------------------------------
    date_cols_to_check = DATE_COLS.get(name, [])
    # Also auto-detect by column name heuristic
    auto_detected = [c for c in df.columns
                     if is_likely_date_col(c) and c not in date_cols_to_check]
    all_date_cols = date_cols_to_check + auto_detected

    date_rows = []
    for col in all_date_cols:
        if col not in df.columns:
            continue
        dr = date_range_summary(df, col)
        date_rows.append(dr)
        if dr["n_valid"] > 0:
            print(f"\n  Date '{col}': {dr['min']} → {dr['max']}"
                  f"  ({dr['n_valid']:,} parseable, {dr['n_unparsed']:,} unparseable)")

    if date_rows:
        pd.DataFrame(date_rows).to_csv(AUDIT_DIR / f"{name}_date_ranges.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Top value counts for object columns (safe summary only)
    # ------------------------------------------------------------------
    cat_rows = []
    for col in df.select_dtypes(include="object").columns:
        vc = top_value_counts(df[col], n=10)
        for val, cnt in vc.items():
            cat_rows.append({"column": col, "value": val, "count": cnt})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(AUDIT_DIR / f"{name}_top_values.csv", index=False)
    print(f"\n  Top-value counts saved to outputs/audit/{name}_top_values.csv")

    summary = {
        "table": name,
        "rows":  df.shape[0],
        "cols":  df.shape[1],
    }
    return summary


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------

def run_audit():
    print("\n" + "="*60)
    print("  DATA AUDIT — DataFest 2026")
    print("="*60)

    # Check all files exist before loading anything.
    tables = {
        "encounters":          {"file": "encounters.csv",          "keys": ["EncounterKey"]},
        "patients":            {"file": "patients.csv",            "keys": ["DurableKey"]},
        "diagnosis":           {"file": "diagnosis.csv",           "keys": ["DiagnosisKey"]},
        "departments":         {"file": "departments.csv",         "keys": ["DepartmentKey"]},
        "providers":           {"file": "providers.csv",           "keys": ["DurableKey"]},
        "social_determinants": {"file": "social_determinants.csv", "keys": ["PatientDurableKey", "EncounterKey"]},
        "tigercensuscodes":    {"file": "tigercensuscodes.csv",    "keys": ["GEOID"]},
    }

    missing_files = [cfg["file"] for cfg in tables.values()
                     if not (DATA_DIR / cfg["file"]).exists()]
    if missing_files:
        print(f"\n  ERROR: Missing files in {DATA_DIR}/:")
        for f in missing_files:
            print(f"    {f}")
        sys.exit(1)

    # Load all tables (encounters is large — we load it in chunks for the
    # special-code scan but keep the full frame for join checks).
    dfs = {}
    for name, cfg in tables.items():
        path = DATA_DIR / cfg["file"]
        print(f"\n  Loading {cfg['file']} …")
        dfs[name] = safe_read(path)

    # Per-table audits
    summaries = []
    for name, cfg in tables.items():
        s = audit_table(name, dfs[name], cfg["keys"])
        summaries.append(s)

    # Save cross-table shape summary
    pd.DataFrame(summaries).to_csv(AUDIT_DIR / "00_shape_summary.csv", index=False)
    print(f"\n  Shape summary saved to outputs/audit/00_shape_summary.csv")

    return dfs


# ---------------------------------------------------------------------------
# Encounter binary-flag audit
# ---------------------------------------------------------------------------

def audit_encounter_flags(enc: pd.DataFrame):
    """Count and cross-tab the binary encounter-type flags."""
    flag_cols = [
        "IsEdVisit", "IsHospitalAdmission", "IsHospitalOutpatientVisit",
        "IsInpatientAdmission", "IsObservation", "IsOutpatientFaceToFaceVisit",
    ]
    present = [c for c in flag_cols if c in enc.columns]
    if not present:
        print("\n  No binary encounter flags found.")
        return

    print("\n  Encounter flag counts:")
    flag_counts = {}
    for col in present:
        vc = enc[col].value_counts(dropna=False)
        flag_counts[col] = vc.to_dict()
        print(f"    {col}: {vc.to_dict()}")

    pd.DataFrame(flag_counts).T.to_csv(AUDIT_DIR / "encounters_flag_counts.csv")

    # Check mutual exclusivity: how many encounters have more than one flag = 1/Yes/True?
    def is_true(series):
        return series.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])

    flag_numeric = pd.DataFrame({c: is_true(enc[c]).astype(int) for c in present})
    flag_sum = flag_numeric.sum(axis=1)
    overlap_counts = flag_sum.value_counts().sort_index()
    print("\n  Encounters with N flags active simultaneously:")
    print(overlap_counts.to_string())
    overlap_counts.to_frame("n_encounters").to_csv(
        AUDIT_DIR / "encounters_flag_overlap.csv"
    )


# ---------------------------------------------------------------------------
# PrimaryDiagnosisKey = -1 summary
# ---------------------------------------------------------------------------

def audit_diagnosis_key_minus1(enc: pd.DataFrame):
    """Report how many encounters have PrimaryDiagnosisKey = -1."""
    if "PrimaryDiagnosisKey" not in enc.columns:
        print("\n  PrimaryDiagnosisKey column not found in encounters.")
        return

    col = enc["PrimaryDiagnosisKey"].astype(str).str.strip()
    n_minus1   = int((col == "-1").sum())
    n_total    = enc.shape[0]
    n_other    = n_total - n_minus1
    pct_minus1 = round(n_minus1 / n_total * 100, 2)
    print(f"\n  PrimaryDiagnosisKey = -1: {n_minus1:,} / {n_total:,} encounters ({pct_minus1}%)")
    print(f"  PrimaryDiagnosisKey ≠ -1: {n_other:,} encounters eligible for diagnosis join")

    summary = pd.DataFrame([{
        "total_encounters":      n_total,
        "primary_dx_key_minus1": n_minus1,
        "pct_minus1":            pct_minus1,
        "eligible_for_dx_join":  n_other,
    }])
    summary.to_csv(AUDIT_DIR / "encounters_dx_minus1.csv", index=False)


# ---------------------------------------------------------------------------
# SDOH domain normalization audit
# ---------------------------------------------------------------------------

def audit_sdoh_domains(sdoh: pd.DataFrame):
    """Check Domain and DisplayName for casing inconsistencies and known issues."""
    print(f"\n{'='*60}")
    print("  SDOH DOMAIN AUDIT")
    print(f"{'='*60}")

    if "Domain" in sdoh.columns:
        domain_vc = sdoh["Domain"].value_counts(dropna=False)
        print(f"\n  Domain value counts ({len(domain_vc)} distinct values):")
        print(domain_vc.to_string())
        domain_vc.to_frame("count").to_csv(AUDIT_DIR / "sdoh_domain_counts.csv")

    if "DisplayName" in sdoh.columns:
        dn_vc = sdoh["DisplayName"].value_counts(dropna=False).head(30)
        print(f"\n  DisplayName top-30 value counts:")
        print(dn_vc.to_string())
        dn_vc.to_frame("count").to_csv(AUDIT_DIR / "sdoh_displayname_top30.csv")

    # Flag known misspelling
    if "Domain" in sdoh.columns:
        n_misspelled = int(sdoh["Domain"].astype(str)
                          .str.lower()
                          .str.contains("violance").sum())
        if n_misspelled > 0:
            print(f"\n  WARNING: {n_misspelled} rows contain 'violance' (misspelling) in Domain.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dfs = run_audit()

    enc  = dfs["encounters"]
    sdoh = dfs["social_determinants"]

    audit_encounter_flags(enc)
    audit_diagnosis_key_minus1(enc)
    audit_sdoh_domains(sdoh)

    # Final checklist
    checklist = """
============================================================
  AUDIT COMPLETE — Pre-journey validation checklist
============================================================

Before building journeys, manually verify:

  [ ] 1. DiagnosisKey uniqueness
          Is diagnosis.DiagnosisKey a 1:1 key, or does one key
          map to multiple DiagnosisValue / GroupCode values?
          → See outputs/audit/diagnosis_key_check.csv
          → See outputs/audit/01_join_validation.py output

  [ ] 2. Diagnosis join inflation
          Does merging encounters → diagnosis on PrimaryDiagnosisKey
          increase row count (1:many join)?
          → See outputs/audit/join_dx_inflation.csv (from 01_join_validation.py)

  [ ] 3. PrimaryDiagnosisKey = -1 handling
          How many encounters have no diagnosis key?
          → See outputs/audit/encounters_dx_minus1.csv

  [ ] 4. Encounter binary-flag overlap
          Are encounter-type flags mutually exclusive?
          → See outputs/audit/encounters_flag_overlap.csv

  [ ] 5. Special coded values in key columns
          Are 'NA', 'Unknown', '-1', blank strings present in
          PatientDurableKey, EncounterKey, DiagnosisKey, DepartmentKey?
          → See outputs/audit/*_special_codes.csv

  [ ] 6. Date column coverage
          Does 'Date' have full coverage in encounters?
          Are AdmissionInstant / DischargeInstant sparse for office visits?
          → See outputs/audit/encounters_date_ranges.csv

  [ ] 7. SDOH domain normalization
          Are Domain values consistently cased?
          Is 'intimate partner violance' misspelling present?
          → See outputs/audit/sdoh_domain_counts.csv

  [ ] 8. Join match rates for all six foreign keys
          (covered fully in 01_join_validation.py)

  [ ] 9. PatientDurableKey coverage across tables
          What fraction of encounter patients appear in patients.csv?

  [ ] 10. Social determinants EncounterKey match rate
           What fraction of SDOH rows link to a known EncounterKey?

============================================================
"""
    print(checklist)

    with open(AUDIT_DIR / "00_checklist.txt", "w") as f:
        f.write(checklist)

    print(f"  All audit outputs saved to: {AUDIT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
