"""
02_diagnosis_mapping_audit.py
------------------------------
Audit the diagnosis.csv mapping structure before building patient journeys.

Problem identified in 01_join_validation.py:
  - DiagnosisKey is not unique in diagnosis.csv.
  - A naive merge on PrimaryDiagnosisKey → DiagnosisKey inflates 6,096,451
    eligible encounters into ~6,519,673 rows (+423,222 extra rows).
  - DiagnosisKey → DiagnosisValue is ambiguous for 129,174 keys.
  - DiagnosisKey → GroupCode is ambiguous for 103,332 keys.

This script characterises the ambiguity in detail, measures its encounter-level
impact, and recommends a safe mapping strategy for journey construction.

Run from project root:
    python code/02_diagnosis_mapping_audit.py

Outputs go to outputs/audit/.
No raw patient-level rows are printed or exported.
"""

from pathlib import Path
import textwrap
import pandas as pd
import numpy as np

DATA_DIR  = Path("data")
AUDIT_DIR = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

VAL_COLS = ["DiagnosisValue", "DiagnosisName", "GroupCode", "GroupName"]


# ---------------------------------------------------------------------------
# Shared utilities (mirrors 01_join_validation.py)
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

    Converts whole-number floats (86333.0) to integers (86333) via nullable
    Int64 so that comparisons across tables with mismatched dtypes are correct.
    Non-numeric tokens and blanks become pd.NA.
    """
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


# ---------------------------------------------------------------------------
# 1. Per-key ambiguity profile for diagnosis.csv
# ---------------------------------------------------------------------------

def build_key_ambiguity_table(diag: pd.DataFrame) -> pd.DataFrame:
    """
    For each DiagnosisKey, count:
      - n_rows           : how many rows in diagnosis.csv share this key
      - n_unique_*       : distinct values of each mapping column
      - is_1to1_*        : True when the key maps to exactly one value
    """
    diag = diag.copy()
    diag["_key"] = normalize_numeric_key(diag["DiagnosisKey"])
    diag = diag[diag["_key"].notna()]

    agg = diag.groupby("_key").agg(
        n_rows=("_key", "count"),
        **{
            f"n_unique_{col}": (col, "nunique")
            for col in VAL_COLS if col in diag.columns
        },
    ).reset_index().rename(columns={"_key": "DiagnosisKey"})

    for col in VAL_COLS:
        uc = f"n_unique_{col}"
        if uc in agg.columns:
            agg[f"is_1to1_{col}"] = agg[uc] == 1

    return agg


def summarise_key_ambiguity(amb: pd.DataFrame) -> dict:
    """Print and return high-level ambiguity counts for diagnosis.csv."""
    print(f"\n{'='*60}")
    print("  DIAGNOSIS.CSV — KEY AMBIGUITY PROFILE")
    print(f"{'='*60}")

    n_keys_total = len(amb)
    print(f"\n  Total unique DiagnosisKey values : {n_keys_total:,}")
    print(f"  Total rows in diagnosis.csv      : {amb['n_rows'].sum():,}")

    summary = {"n_keys_total": n_keys_total}

    for col in VAL_COLS:
        uc = f"n_unique_{col}"
        flag_col = f"is_1to1_{col}"
        if uc not in amb.columns:
            continue
        n_1to1   = int(amb[flag_col].sum())
        n_ambig  = n_keys_total - n_1to1
        pct_1to1 = round(n_1to1 / n_keys_total * 100, 2) if n_keys_total else 0.0
        max_fan  = int(amb[uc].max())

        print(f"\n  DiagnosisKey → {col}:")
        print(f"    1-to-1 keys  : {n_1to1:,}  ({pct_1to1}%)")
        print(f"    Ambiguous    : {n_ambig:,}  ({100 - pct_1to1}%)")
        print(f"    Max fanout   : {max_fan} distinct values per key")

        # Distribution of fanout (how many keys have 1, 2, 3, … distinct values)
        fanout_dist = amb[uc].value_counts().sort_index()
        print(f"    Fanout distribution (top 8):")
        for n_vals, n_keys in fanout_dist.head(8).items():
            print(f"      {n_vals} distinct value(s) → {n_keys:,} keys")

        summary[f"n_1to1_{col}"]  = n_1to1
        summary[f"n_ambig_{col}"] = n_ambig

    return summary


# ---------------------------------------------------------------------------
# 2. Encounter-weighted ambiguity
# ---------------------------------------------------------------------------

def build_encounter_weighted_ambiguity(
    enc: pd.DataFrame,
    amb: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each eligible encounter (PrimaryDiagnosisKey ≠ -1 and not null),
    look up how many distinct values its key maps to for each VAL_COL.

    Returns a tidy DataFrame: one row per (mapping_col, n_distinct_values_bucket)
    with encounter counts and percentages.
    """
    enc = enc.copy()
    enc["_key"] = normalize_numeric_key(enc["PrimaryDiagnosisKey"])

    # Restrict to eligible encounters
    eligible = enc[enc["_key"].notna() & (enc["_key"] != -1)].copy()
    n_eligible = len(eligible)

    # Build a lookup dict: DiagnosisKey → n_unique_<col> for each val col
    rows = []
    for col in VAL_COLS:
        uc = f"n_unique_{col}"
        if uc not in amb.columns:
            continue

        lookup = amb.set_index("DiagnosisKey")[uc].to_dict()
        fanout = eligible["_key"].map(lookup)

        # Bucket: 0 = no match, 1 = clean 1-to-1, 2+ = ambiguous
        n_no_match  = int(fanout.isna().sum())
        n_1to1      = int((fanout == 1).sum())
        n_ambiguous = int((fanout > 1).sum())

        pct_1to1     = round(n_1to1      / n_eligible * 100, 2) if n_eligible else 0.0
        pct_ambig    = round(n_ambiguous  / n_eligible * 100, 2) if n_eligible else 0.0
        pct_no_match = round(n_no_match   / n_eligible * 100, 2) if n_eligible else 0.0

        rows.append({
            "mapping_col":            col,
            "n_eligible_encounters":  n_eligible,
            "n_no_match":             n_no_match,
            "pct_no_match":           pct_no_match,
            "n_clean_1to1":           n_1to1,
            "pct_clean_1to1":         pct_1to1,
            "n_ambiguous_2plus":      n_ambiguous,
            "pct_ambiguous_2plus":    pct_ambig,
        })

        print(f"\n  Encounter-weighted ambiguity — DiagnosisKey → {col}:")
        print(f"    Eligible encounters         : {n_eligible:,}")
        print(f"    No diagnosis match          : {n_no_match:,}  ({pct_no_match}%)")
        print(f"    Clean 1-to-1 mapping        : {n_1to1:,}  ({pct_1to1}%)")
        print(f"    Ambiguous (2+ values)       : {n_ambiguous:,}  ({pct_ambig}%)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Strategy evaluation
# ---------------------------------------------------------------------------

def evaluate_strategies(
    amb:       pd.DataFrame,
    enc_weighted: pd.DataFrame,
    n_enc_total: int,
    n_enc_minus1: int,
) -> str:
    """
    Evaluate three deduplication / join strategies and return a text report.
    All reasoning is based on aggregate counts only.
    """

    n_eligible = int(
        enc_weighted.loc[enc_weighted["mapping_col"] == "DiagnosisValue",
                         "n_eligible_encounters"].iloc[0]
    ) if "DiagnosisValue" in enc_weighted["mapping_col"].values else None

    def _row(col):
        r = enc_weighted[enc_weighted["mapping_col"] == col]
        return r.iloc[0].to_dict() if len(r) else {}

    dv = _row("DiagnosisValue")
    gc = _row("GroupCode")
    gn = _row("GroupName")

    lines = []
    sep   = "=" * 60

    lines += [
        sep,
        "  DIAGNOSIS MAPPING — STRATEGY EVALUATION",
        sep,
        "",
        f"  Total encounters in file          : {n_enc_total:,}",
        f"  PrimaryDiagnosisKey = -1 (no Dx)  : {n_enc_minus1:,}",
        f"  Eligible for diagnosis join        : {n_eligible:,}" if n_eligible else "",
        "",
    ]

    # ---- Strategy A ----
    a_clean   = dv.get("n_clean_1to1", "?")
    a_ambig   = dv.get("n_ambiguous_2plus", "?")
    a_pct_c   = dv.get("pct_clean_1to1", "?")
    a_pct_a   = dv.get("pct_ambiguous_2plus", "?")
    lines += [
        "  STRATEGY A — use DiagnosisValue, flag ambiguous keys",
        "  " + "-"*56,
        "  For DiagnosisKeys that map to exactly one DiagnosisValue,",
        "  take that value directly.  For keys with 2+ DiagnosisValue",
        "  values, flag those encounters as 'ambiguous_dx_value' and",
        "  exclude them from the primary journey analysis.",
        "",
        f"  Encounters with clean 1-to-1 DiagnosisValue : {a_clean:,}  ({a_pct_c}%)",
        f"  Encounters flagged as ambiguous              : {a_ambig:,}  ({a_pct_a}%)",
        "",
        "  PRO : Preserves the most specific diagnosis label.",
        "  CON : Ambiguous encounters are excluded or need a tiebreak rule.",
        "  WHEN TO USE : If the ambiguous share is small enough that",
        "  excluding it does not bias the journey population materially.",
        "",
    ]

    # ---- Strategy B ----
    b_clean  = gc.get("n_clean_1to1", "?")
    b_ambig  = gc.get("n_ambiguous_2plus", "?")
    b_pct_c  = gc.get("pct_clean_1to1", "?")
    b_pct_a  = gc.get("pct_ambiguous_2plus", "?")
    bn_clean = gn.get("n_clean_1to1", "?")
    bn_ambig = gn.get("n_ambiguous_2plus", "?")
    bn_pct_c = gn.get("pct_clean_1to1", "?")
    bn_pct_a = gn.get("pct_ambiguous_2plus", "?")
    lines += [
        "  STRATEGY B — use GroupCode / GroupName as grouping key",
        "  " + "-"*56,
        "  Instead of DiagnosisValue, group journeys by the broader",
        "  GroupCode or GroupName category, which may have lower",
        "  encounter-weighted ambiguity.",
        "",
        f"  Encounters with clean 1-to-1 GroupCode  : {b_clean:,}  ({b_pct_c}%)",
        f"  Encounters flagged as ambiguous (GroupCode): {b_ambig:,}  ({b_pct_a}%)",
        f"  Encounters with clean 1-to-1 GroupName  : {bn_clean:,}  ({bn_pct_c}%)",
        f"  Encounters flagged as ambiguous (GroupName): {bn_ambig:,}  ({bn_pct_a}%)",
        "",
        "  PRO : Broader grouping may reduce ambiguity and produce more",
        "  populated journey groups suitable for aggregate comparisons.",
        "  CON : Loses specificity; two clinically distinct diagnoses may",
        "  be merged into the same journey.",
        "  WHEN TO USE : If GroupCode ambiguity is substantially lower than",
        "  DiagnosisValue ambiguity, or when the research question focuses",
        "  on care-setting patterns rather than specific diagnoses.",
        "",
    ]

    # ---- Strategy C ----
    c_excl     = a_ambig  # same excluded set as A
    c_pct_excl = a_pct_a
    lines += [
        "  STRATEGY C — exclude all ambiguous DiagnosisKeys entirely",
        "  " + "-"*56,
        "  Build a 'clean journey subset' using only encounters whose",
        "  PrimaryDiagnosisKey maps to exactly one DiagnosisValue.",
        "  Report the excluded share explicitly as a limitation.",
        "",
        f"  Encounters excluded (ambiguous key)    : {c_excl:,}  ({c_pct_excl}%)",
        f"  Encounters retained for clean analysis : {a_clean:,}  ({a_pct_c}%)",
        "",
        "  PRO : Guarantees no row inflation and unambiguous diagnosis labels.",
        "  CON : If the excluded share is large or systematically different",
        "  from the included population, results may not generalise.",
        "  WHEN TO USE : As the primary strategy when a clean, conservative",
        "  journey definition is preferred over completeness.",
        "",
    ]

    # ---- Recommendation ----
    # Base recommendation on encounter-weighted ambiguity of DiagnosisValue vs GroupCode.
    dv_pct_ambig = dv.get("pct_ambiguous_2plus", 100.0)
    gc_pct_ambig = gc.get("pct_ambiguous_2plus", 100.0)

    if isinstance(dv_pct_ambig, float) and isinstance(gc_pct_ambig, float):
        if dv_pct_ambig <= 5.0:
            rec = (
                "RECOMMENDED: Strategy C (exclude ambiguous keys).\n"
                f"  Only {dv_pct_ambig}% of eligible encounters use ambiguous DiagnosisKeys,\n"
                "  so exclusion has minimal impact on the journey population.\n"
                "  Use DiagnosisValue as the journey label for clean encounters."
            )
        elif gc_pct_ambig < dv_pct_ambig * 0.5:
            rec = (
                "RECOMMENDED: Strategy B (GroupCode as grouping key).\n"
                f"  DiagnosisValue ambiguity ({dv_pct_ambig}%) is high, but GroupCode\n"
                f"  ambiguity ({gc_pct_ambig}%) is substantially lower.\n"
                "  Use GroupCode for journey grouping and DiagnosisValue as a label\n"
                "  where it is unambiguous."
            )
        else:
            rec = (
                "RECOMMENDED: Strategy A + Strategy C hybrid.\n"
                f"  DiagnosisValue ambiguity is {dv_pct_ambig}% and GroupCode is\n"
                f"  {gc_pct_ambig}% — neither offers a clearly clean path.\n"
                "  Build the primary journey table using Strategy C (clean keys only),\n"
                "  then build a secondary table with Strategy A tiebreaking\n"
                "  (pick the DiagnosisValue with the most rows for that key)\n"
                "  and compare results as a sensitivity check."
            )
    else:
        rec = "  Could not compute recommendation — check that DiagnosisValue and GroupCode columns exist."

    lines += [
        sep,
        "  RECOMMENDATION",
        sep,
        "",
        f"  {rec}",
        "",
        "  In all cases:",
        "  - Report the number and percentage of encounters excluded or flagged.",
        "  - Do not treat a one-encounter journey as inherently incomplete.",
        "  - Use 'associated with' / 'observed among' language, not causal framing.",
        "",
        sep,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. DeduplicatingDiagnosisKey: show what a tiebreak lookup would look like
# ---------------------------------------------------------------------------

def build_dedup_preview(diag: pd.DataFrame, amb: pd.DataFrame) -> pd.DataFrame:
    """
    For ambiguous DiagnosisKeys, identify the most frequent DiagnosisValue
    per key (a simple majority-vote tiebreak).  Returns a summary table —
    no raw patient data, only key-level counts.
    """
    diag = diag.copy()
    diag["_key"] = normalize_numeric_key(diag["DiagnosisKey"])
    diag = diag[diag["_key"].notna()]

    if "DiagnosisValue" not in diag.columns:
        return pd.DataFrame()

    # Identify ambiguous keys
    ambig_keys = set(
        amb.loc[amb.get("n_unique_DiagnosisValue", pd.Series(dtype=int)) > 1,
                "DiagnosisKey"]
    ) if "n_unique_DiagnosisValue" in amb.columns else set()

    if not ambig_keys:
        return pd.DataFrame()

    ambig_rows = diag[diag["_key"].isin(ambig_keys)]

    # For each ambiguous key: count rows per DiagnosisValue, pick the plurality
    tiebreak = (
        ambig_rows.groupby(["_key", "DiagnosisValue"])
        .size()
        .reset_index(name="n_rows_with_this_value")
    )
    # Rank within each key
    tiebreak["rank"] = (
        tiebreak.groupby("_key")["n_rows_with_this_value"]
                .rank(method="first", ascending=False)
                .astype(int)
    )
    top1 = tiebreak[tiebreak["rank"] == 1].copy()

    # Flag whether the tiebreak was unambiguous (unique top count)
    max_counts = tiebreak.groupby("_key")["n_rows_with_this_value"].max()
    count_of_max = tiebreak.groupby("_key").apply(
        lambda g: (g["n_rows_with_this_value"] == g["n_rows_with_this_value"].max()).sum(),
        include_groups=False,
    )
    top1["tiebreak_is_unique"] = top1["_key"].map(count_of_max == 1)

    # Summary counts only — no individual rows
    n_ambig_keys        = len(ambig_keys)
    n_clear_plurality   = int(top1["tiebreak_is_unique"].sum())
    n_tied_plurality    = n_ambig_keys - n_clear_plurality
    pct_clear           = round(n_clear_plurality / n_ambig_keys * 100, 2) if n_ambig_keys else 0.0

    print(f"\n  Tiebreak preview (most-frequent DiagnosisValue per ambiguous key):")
    print(f"    Ambiguous DiagnosisKey count         : {n_ambig_keys:,}")
    print(f"    Keys with a clear plurality value    : {n_clear_plurality:,}  ({pct_clear}%)")
    print(f"    Keys with tied plurality (still ambig): {n_tied_plurality:,}  ({100-pct_clear}%)")

    tiebreak_summary = pd.DataFrame([{
        "n_ambiguous_keys":            n_ambig_keys,
        "n_clear_plurality":           n_clear_plurality,
        "pct_clear_plurality":         pct_clear,
        "n_still_tied_after_tiebreak": n_tied_plurality,
    }])
    return tiebreak_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  DIAGNOSIS MAPPING AUDIT — DataFest 2026")
    print("="*60)

    # Load tables
    print("\n  Loading diagnosis.csv and encounters.csv …")
    diag = safe_read(DATA_DIR / "diagnosis.csv")
    enc  = safe_read(DATA_DIR / "encounters.csv",
                     usecols=["EncounterKey", "PrimaryDiagnosisKey"])

    print(f"  diagnosis.csv  : {diag.shape[0]:,} rows, columns: {list(diag.columns)}")
    print(f"  encounters.csv : {enc.shape[0]:,} rows (PrimaryDiagnosisKey only)")

    # Encounter counts for strategy evaluation
    enc_key_norm = normalize_numeric_key(enc["PrimaryDiagnosisKey"])
    n_enc_total  = len(enc_key_norm)
    n_enc_minus1 = int((enc_key_norm == -1).sum())
    n_enc_null   = int(enc_key_norm.isna().sum())
    print(f"\n  PrimaryDiagnosisKey = -1 (sentinel) : {n_enc_minus1:,}")
    print(f"  PrimaryDiagnosisKey null/unparseable: {n_enc_null:,}")
    print(f"  Eligible for diagnosis join         : {n_enc_total - n_enc_minus1 - n_enc_null:,}")

    # ------------------------------------------------------------------
    # 1. Per-key ambiguity table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  BUILDING PER-KEY AMBIGUITY TABLE …")
    print(f"{'='*60}")
    amb = build_key_ambiguity_table(diag)
    amb.to_csv(AUDIT_DIR / "diagnosis_key_ambiguity_summary.csv", index=False)
    print(f"  Saved: outputs/audit/diagnosis_key_ambiguity_summary.csv")
    print(f"  Rows in ambiguity table: {len(amb):,}  (one row per unique DiagnosisKey)")

    key_summary = summarise_key_ambiguity(amb)

    # ------------------------------------------------------------------
    # 2. Encounter-weighted ambiguity
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ENCOUNTER-WEIGHTED AMBIGUITY")
    print(f"{'='*60}")
    enc_weighted = build_encounter_weighted_ambiguity(enc, amb)
    enc_weighted.to_csv(
        AUDIT_DIR / "diagnosis_key_fanout_by_encounter_weight.csv", index=False
    )
    print(f"\n  Saved: outputs/audit/diagnosis_key_fanout_by_encounter_weight.csv")

    # ------------------------------------------------------------------
    # 3. Tiebreak preview for ambiguous keys
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  TIEBREAK PREVIEW (ambiguous DiagnosisKeys)")
    print(f"{'='*60}")
    tiebreak_summary = build_dedup_preview(diag, amb)
    if not tiebreak_summary.empty:
        tiebreak_summary.to_csv(
            AUDIT_DIR / "diagnosis_key_tiebreak_summary.csv", index=False
        )
        print(f"  Saved: outputs/audit/diagnosis_key_tiebreak_summary.csv")

    # ------------------------------------------------------------------
    # 4. Strategy evaluation and recommendation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  STRATEGY EVALUATION")
    print(f"{'='*60}")
    strategy_text = evaluate_strategies(amb, enc_weighted, n_enc_total, n_enc_minus1)
    print("\n" + strategy_text)

    notes_path = AUDIT_DIR / "diagnosis_mapping_strategy_notes.txt"
    with open(notes_path, "w") as f:
        f.write("DIAGNOSIS MAPPING STRATEGY NOTES\n")
        f.write("Generated by: code/02_diagnosis_mapping_audit.py\n")
        f.write(f"diagnosis.csv rows       : {diag.shape[0]:,}\n")
        f.write(f"encounters eligible      : {n_enc_total - n_enc_minus1 - n_enc_null:,}\n")
        f.write("\n")
        f.write(strategy_text)
        f.write("\n\nCOLUMN PRESENCE IN diagnosis.csv\n")
        for col in VAL_COLS:
            present = col in diag.columns
            f.write(f"  {col}: {'PRESENT' if present else 'MISSING'}\n")
        f.write("\nOUTPUT FILES\n")
        f.write("  diagnosis_key_ambiguity_summary.csv\n")
        f.write("      One row per unique DiagnosisKey.\n")
        f.write("      Columns: DiagnosisKey, n_rows, n_unique_{col}, is_1to1_{col}\n")
        f.write("      for DiagnosisValue, DiagnosisName, GroupCode, GroupName.\n\n")
        f.write("  diagnosis_key_fanout_by_encounter_weight.csv\n")
        f.write("      One row per mapping column.\n")
        f.write("      Counts of eligible encounters by clean/ambiguous/no-match bucket.\n\n")
        f.write("  diagnosis_key_tiebreak_summary.csv\n")
        f.write("      Summary of plurality-tiebreak feasibility for ambiguous keys.\n\n")
        f.write("NEXT STEP\n")
        f.write("  Choose a strategy above and implement it in 03_build_journeys.py.\n")
        f.write("  Always report the number of encounters excluded or flagged as a\n")
        f.write("  limitation in any analysis that uses journey-level data.\n")

    print(f"\n  Saved: outputs/audit/diagnosis_mapping_strategy_notes.txt")

    print(f"""
============================================================
  AUDIT COMPLETE
============================================================

Outputs in outputs/audit/:
  diagnosis_key_ambiguity_summary.csv          — per-key fanout profile
  diagnosis_key_fanout_by_encounter_weight.csv — encounter-level impact
  diagnosis_key_tiebreak_summary.csv           — plurality-tiebreak feasibility
  diagnosis_mapping_strategy_notes.txt         — full strategy evaluation

Read diagnosis_mapping_strategy_notes.txt to choose a strategy,
then implement it in 03_build_journeys.py.
============================================================
""")


if __name__ == "__main__":
    main()
