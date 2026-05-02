"""
06_sdoh_answer_coding_audit.py
-------------------------------
Audit AnswerText values by SDOH domain and question to inform
the decision of which answers indicate a recorded need or risk.

Does NOT assign final need/risk flags — only characterises the
distributions and recommends a coding approach per question.

Inputs:
    data/social_determinants.csv
    outputs/tables/journeys_classified_with_patient_sdoh.parquet
        (used only to get the journey-cohort patient key set)

Outputs:
    outputs/audit/sdoh_answer_counts_by_domain_question.csv
    outputs/audit/sdoh_question_counts_by_domain.csv
    outputs/audit/sdoh_domain_answer_top_values.csv
    outputs/audit/sdoh_need_coding_notes.txt

Run from project root:
    python code/06_sdoh_answer_coding_audit.py

No raw patient-level rows are printed or exported.
All printed output is aggregate only.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SDOH_PATH    = DATA_DIR   / "social_determinants.csv"
JOURNEY_PATH = TABLES_DIR / "journeys_classified_with_patient_sdoh.parquet"

# ---------------------------------------------------------------------------
# Domain normalization — identical to 05_add_patient_sdoh_context.py
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Answer-set classifiers (based on observed real values in this dataset)
# ---------------------------------------------------------------------------

# Values treated as non-substantive (refusal / system-generated / unknown).
SPECIAL_ANSWER_LOWER = {
    "patient declined",
    "patient unable to answer",
    "patient does not drink",
    "unknown",
    "*unknown",
    "na",
    "*not applicable",
    "",
    "never assessed",
}

# Known answer-set patterns with their coding implications.
# Each entry: (frozenset of expected clean lowercased values, type_label, coding_note)
_ANSWER_PATTERNS = [
    (
        frozenset({"yes", "no"}),
        "binary_yes_no",
        "YES = positive screen / need indicator.  NO = not screened positive.",
    ),
    (
        frozenset({"never true", "sometimes true", "often true"}),
        "food_insecurity_scale",
        "SOMETIMES TRUE or OFTEN TRUE = food insecurity need per standard HRSN protocol.",
    ),
    (
        frozenset({"not hard at all", "not very hard", "somewhat hard", "hard", "very hard"}),
        "hardness_likert",
        "SOMEWHAT HARD, HARD, or VERY HARD may indicate financial/utility strain. "
        "Threshold requires human review — consider 'Somewhat hard' and above vs. only 'Hard'.",
    ),
    (
        frozenset({"not at all", "only a little", "to some extent", "rather much", "very much"}),
        "stress_likert",
        "RATHER MUCH or VERY MUCH likely indicates elevated stress.  "
        "TO SOME EXTENT is borderline — threshold requires human review.",
    ),
    (
        frozenset({"never", "monthly or less", "2-4 times a month",
                   "2-3 times a week", "4 or more times a week"}),
        "alcohol_frequency",
        "AUDIT-C Q1 frequency.  Any response above NEVER may warrant attention; "
        "standard AUDIT-C threshold (score ≥ 3 for women, ≥ 4 for men) requires combining Q1–Q3.",
    ),
    (
        frozenset({"1 or 2", "3 or 4", "5 to 6", "7 to 9", "10 or more"}),
        "alcohol_quantity",
        "AUDIT-C Q2 quantity per drinking day.  Must be combined with Q1 and Q3 to compute score.",
    ),
    (
        frozenset({"never", "less than monthly", "monthly", "weekly", "daily or almost daily"}),
        "alcohol_binge_frequency",
        "AUDIT-C Q3 binge frequency.  Must be combined with Q1 and Q2 to compute score.",
    ),
]

_PHQ_PATTERNS = {"phq", "score", "total", "edinburgh"}


def _strip_specials(values: set) -> set:
    """Remove special/refusal values from an answer set for classification."""
    return {v for v in values if v.lower() not in SPECIAL_ANSWER_LOWER}


def classify_answer_set(
    display_name: str,
    answer_values: list,
) -> tuple[str, str, str]:
    """
    Classify the answer set for one question.

    Returns (type_label, coding_recommendation, requires_review_flag).
    requires_review_flag: "auto-codeable" | "needs_threshold" | "needs_human_review"
    """
    clean = _strip_specials({str(v).strip() for v in answer_values})
    lower = {v.lower() for v in clean}
    dn    = display_name.lower()

    # Check for clearly numeric PHQ / depression scores.
    if any(p in dn for p in _PHQ_PATTERNS):
        return (
            "numeric_depression_score",
            "PHQ-2: threshold ≥ 3 is standard positive screen for depression. "
            "Edinburgh Postnatal Depression Scale: threshold ≥ 10 is common, "
            "but clinical cutoff varies by context. Requires human review.",
            "needs_threshold",
        )

    # Check for exercise minutes / days (numeric count questions).
    if ("minutes" in dn or "how many minutes" in dn) and clean:
        return (
            "numeric_exercise_minutes",
            "Duration in minutes per session. Must combine with days-per-week column "
            "to compute weekly volume. Do not threshold in isolation.",
            "needs_threshold",
        )

    if ("days per week" in dn or "how many days" in dn or "how many times" in dn) and \
       all(v.replace(" days", "").replace(" times", "").strip().isdigit() for v in clean if v):
        return (
            "numeric_frequency_count",
            "Integer count (days or times per unit time). Standard thresholds vary by domain. "
            "For physical activity: CDC guideline is ≥5 days/week or ≥150 min/week total.",
            "needs_threshold",
        )

    # How many places / times have you moved (numeric).
    if "how many" in dn and clean and all(
        v.replace("+", "").strip().isdigit() for v in clean if v
    ):
        return (
            "numeric_housing_count",
            "Integer count (places lived / moves). Higher values suggest housing instability. "
            "Threshold requires human review.",
            "needs_threshold",
        )

    # Match against known answer-set patterns.
    for expected, type_label, note in _ANSWER_PATTERNS:
        # Allow subset match — observed set may be a subset of the full scale.
        if lower and lower <= expected:
            review = "auto-codeable" if type_label == "binary_yes_no" \
                     else ("auto-codeable" if type_label == "food_insecurity_scale"
                           else "needs_threshold")
            return (type_label, note, review)

    # Marital status (Social Connections sub-question).
    MARITAL = {"married", "widowed", "divorced", "separated", "never married", "living with partner"}
    if lower & MARITAL:
        return (
            "categorical_marital_status",
            "Marital status is a Social Connections sub-item, not a standalone need indicator. "
            "Useful as a contextual covariate but requires human review to incorporate as risk flag.",
            "needs_human_review",
        )

    # Club / church membership (Social Connections — Yes/No variants).
    if ("belong" in dn or "church" in dn or "attend" in dn or "club" in dn) and \
       lower <= {"yes", "no"}:
        return (
            "binary_yes_no",
            "YES/NO social participation item. NO may indicate lower social engagement, "
            "but interpretation requires human review in isolation.",
            "needs_human_review",
        )

    # General frequency (named but not matched above).
    FREQ_TERMS = {"never", "rarely", "sometimes", "often", "always",
                  "1 to 4 times per year", "more than 4 times per year",
                  "once a week", "twice a week", "2-3 times a week",
                  "more than three times a week"}
    if lower & FREQ_TERMS:
        return (
            "frequency_scale",
            "Ordinal frequency scale. Threshold for 'low' frequency requires human review. "
            "Do not assume a single cutpoint without domain-specific guidance.",
            "needs_threshold",
        )

    # Fallback.
    return (
        "categorical_other",
        "Answer values do not match a known pattern. Manual review required before coding.",
        "needs_human_review",
    )


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

def load_clean_sdoh() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load social_determinants.csv, apply domain corrections, exclude
    unclean rows.  Returns (sdoh_full, sdoh_clean, counts_dict).
    """
    sdoh = safe_read(SDOH_PATH)
    sdoh["PatientDurableKey"] = normalize_numeric_key(sdoh["PatientDurableKey"])
    sdoh["EncounterKey"]      = normalize_numeric_key(sdoh["EncounterKey"])
    n_total = len(sdoh)

    sdoh["Domain_raw"]   = sdoh["Domain"].astype(str).str.strip()
    sdoh["Domain_clean"] = apply_domain_corrections(sdoh["Domain"])

    mask_unspec     = sdoh["DisplayName"] == "*Unspecified"
    mask_domain_na  = sdoh["Domain_raw"]  == "NA"
    n_excl_unspec   = int(mask_unspec.sum())
    n_excl_na       = int(mask_domain_na.sum())
    n_excl_either   = int((mask_unspec | mask_domain_na).sum())

    sdoh_clean = sdoh[~mask_unspec & ~mask_domain_na].copy()
    n_clean    = len(sdoh_clean)
    n_patients = int(sdoh_clean["PatientDurableKey"].dropna().nunique())

    counts = {
        "n_rows_total":                           n_total,
        "n_rows_excluded_displayname_unspecified": n_excl_unspec,
        "n_rows_excluded_domain_na":               n_excl_na,
        "n_rows_excluded_either_rule":             n_excl_either,
        "n_rows_retained":                         n_clean,
        "n_unique_patients_in_clean_rows":         n_patients,
    }
    return sdoh, sdoh_clean, counts


# ---------------------------------------------------------------------------
# 2. Journey-cohort patient set (for coverage stats only)
# ---------------------------------------------------------------------------

def load_journey_patient_keys() -> set:
    """Return the set of Int64 PatientDurableKey values in the journey cohort."""
    jdf = pd.read_parquet(JOURNEY_PATH, columns=["PatientDurableKey"])
    return set(jdf["PatientDurableKey"].dropna().unique())


# ---------------------------------------------------------------------------
# 3. Aggregate tables
# ---------------------------------------------------------------------------

def answer_counts_by_domain_question(sdoh_clean: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Domain_clean, DisplayName, AnswerText): count and
    percentage within that question.
    """
    grp = (
        sdoh_clean
        .groupby(["Domain_clean", "DisplayName", "AnswerText"], dropna=False)
        .size()
        .reset_index(name="n_responses")
    )
    q_totals = grp.groupby(["Domain_clean", "DisplayName"])["n_responses"].transform("sum")
    grp["pct_within_question"] = (grp["n_responses"] / q_totals * 100).round(2)
    return grp.sort_values(
        ["Domain_clean", "DisplayName", "n_responses"], ascending=[True, True, False]
    ).reset_index(drop=True)


def question_counts_by_domain(
    sdoh_clean: pd.DataFrame,
    cohort_keys: set,
) -> pd.DataFrame:
    """
    For each (Domain_clean, DisplayName): total responses, unique patients,
    unique encounters, and coverage within the journey cohort.
    """
    grp = (
        sdoh_clean
        .groupby(["Domain_clean", "DisplayName"], dropna=False)
        .agg(
            n_responses     =("AnswerText",        "count"),
            n_unique_patients=("PatientDurableKey", "nunique"),
            n_unique_encounters=("EncounterKey",    "nunique"),
        )
        .reset_index()
    )

    # Journey-cohort patient coverage per question.
    def cohort_patients(subdf):
        return int(subdf["PatientDurableKey"].dropna().isin(cohort_keys).sum()
                   if len(cohort_keys) > 0 else 0)

    cohort_counts = (
        sdoh_clean
        .groupby(["Domain_clean", "DisplayName"], dropna=False)
        .apply(cohort_patients, include_groups=False)
        .reset_index(name="n_journey_cohort_patients")
    )
    grp = grp.merge(cohort_counts, on=["Domain_clean", "DisplayName"], how="left")
    grp["pct_cohort_patients"] = (
        grp["n_journey_cohort_patients"] / len(cohort_keys) * 100
    ).round(2) if cohort_keys else 0.0

    return grp.sort_values(
        ["Domain_clean", "n_responses"], ascending=[True, False]
    ).reset_index(drop=True)


def domain_answer_top_values(sdoh_clean: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Top AnswerText values aggregated across all questions within each domain.
    """
    grp = (
        sdoh_clean
        .groupby(["Domain_clean", "AnswerText"], dropna=False)
        .size()
        .reset_index(name="n_responses")
    )
    dom_totals = grp.groupby("Domain_clean")["n_responses"].transform("sum")
    grp["pct_within_domain"] = (grp["n_responses"] / dom_totals * 100).round(2)

    top = (
        grp.sort_values(["Domain_clean", "n_responses"], ascending=[True, False])
           .groupby("Domain_clean")
           .head(top_n)
           .reset_index(drop=True)
    )
    return top


# ---------------------------------------------------------------------------
# 4. Coding notes generator
# ---------------------------------------------------------------------------

def build_coding_notes(
    answer_df:    pd.DataFrame,
    question_df:  pd.DataFrame,
    counts:       dict,
) -> str:
    """
    Generate the full sdoh_need_coding_notes.txt content.
    Logic is entirely data-driven from observed AnswerText values.
    """
    SEP  = "=" * 68
    DASH = "-" * 68
    W    = 66

    def wrap(text, indent="  "):
        return textwrap.fill(text, width=W, initial_indent=indent,
                             subsequent_indent=indent)

    lines = [
        SEP,
        "  SDOH ANSWER CODING AUDIT — NEED/RISK CODING NOTES",
        "  Generated by: code/06_sdoh_answer_coding_audit.py",
        SEP,
        "",
        "  PURPOSE",
        DASH,
        wrap(
            "This file summarises the observed AnswerText distributions for "
            "each SDOH domain and question, and recommends a coding approach "
            "for the need/risk flag to be built in a later script. "
            "No need/risk flags are assigned here."
        ),
        "",
        "  EXCLUSIONS APPLIED",
        DASH,
        f"  Total SDOH rows in file                  : {counts['n_rows_total']:,}",
        f"  Excluded (DisplayName = '*Unspecified')  : {counts['n_rows_excluded_displayname_unspecified']:,}",
        f"  Excluded (Domain = 'NA')                 : {counts['n_rows_excluded_domain_na']:,}",
        f"  Excluded (either rule, may overlap)      : {counts['n_rows_excluded_either_rule']:,}",
        f"  Retained for audit                       : {counts['n_rows_retained']:,}",
        f"  Unique patients in retained rows         : {counts['n_unique_patients_in_clean_rows']:,}",
        "",
    ]

    # Per-domain sections.
    for domain in CANONICAL_DOMAINS:
        dom_q   = question_df[question_df["Domain_clean"] == domain]
        dom_ans = answer_df[answer_df["Domain_clean"]    == domain]

        if dom_q.empty:
            lines += [SEP, f"  DOMAIN: {domain}  [NO DATA IN FILE]", ""]
            continue

        n_dom_responses = int(dom_q["n_responses"].sum())
        n_dom_patients  = int(dom_q["n_unique_patients"].max())

        lines += [
            SEP,
            f"  DOMAIN: {domain}",
            DASH,
            f"  Total responses in domain : {n_dom_responses:,}",
            f"  Max unique patients       : {n_dom_patients:,}",
            "",
        ]

        for _, qrow in dom_q.iterrows():
            qname   = str(qrow["DisplayName"])
            q_ans   = dom_ans[dom_ans["DisplayName"] == qname].sort_values(
                "n_responses", ascending=False
            )
            observed_answers = q_ans["AnswerText"].tolist()

            type_label, coding_note, review_flag = classify_answer_set(qname, observed_answers)

            # Shorten long question text for readability.
            q_display = qname if len(qname) <= 70 else qname[:67] + "…"

            lines += [
                f"  QUESTION: {q_display}",
                f"    n_responses        : {qrow['n_responses']:,}",
                f"    n_unique_patients  : {qrow['n_unique_patients']:,}",
                f"    n_unique_encounters: {qrow['n_unique_encounters']:,}",
                f"    Journey-cohort pts : {qrow.get('n_journey_cohort_patients', '?'):,}  "
                f"({qrow.get('pct_cohort_patients', '?')}% of journey cohort)",
                f"    Answer type        : {type_label}",
                f"    Review status      : {review_flag}",
                "",
                f"    Observed answers (top 12, n = count):",
            ]

            for _, arow in q_ans.head(12).iterrows():
                atext = str(arow["AnswerText"])
                atext_disp = atext if len(atext) <= 48 else atext[:45] + "…"
                lines.append(
                    f"      {atext_disp:<50} {arow['n_responses']:>8,}  "
                    f"({arow['pct_within_question']:>5.1f}%)"
                )

            # Count special values.
            special_rows = q_ans[
                q_ans["AnswerText"].str.lower().str.strip().isin(SPECIAL_ANSWER_LOWER)
            ]
            if not special_rows.empty:
                n_special = int(special_rows["n_responses"].sum())
                pct_spec  = round(n_special / qrow["n_responses"] * 100, 1) if qrow["n_responses"] else 0
                lines.append(
                    f"\n    Special/refusal values total: {n_special:,}  ({pct_spec}% of question responses)"
                )

            lines += [
                "",
                wrap(f"    CODING NOTE: {coding_note}", indent="    "),
                "",
            ]

    # Summary table: all questions with their review status.
    lines += [
        SEP,
        "  CODING DECISION SUMMARY TABLE",
        DASH,
        f"  {'Domain':<30}  {'Review status':<22}  {'#Resp':>8}",
        f"  {'-'*62}",
    ]
    for domain in CANONICAL_DOMAINS:
        dom_q   = question_df[question_df["Domain_clean"] == domain]
        dom_ans = answer_df[answer_df["Domain_clean"]    == domain]
        for _, qrow in dom_q.iterrows():
            qname   = str(qrow["DisplayName"])
            obs_ans = dom_ans[dom_ans["DisplayName"] == qname]["AnswerText"].tolist()
            _, _, review_flag = classify_answer_set(qname, obs_ans)
            q_short = qname[:28] + "…" if len(qname) > 30 else qname
            lines.append(
                f"  {domain[:28]:<30}  {review_flag:<22}  "
                f"{qrow['n_responses']:>8,}"
            )

    auto    = sum(
        1 for domain in CANONICAL_DOMAINS
        for _, qrow in question_df[question_df["Domain_clean"] == domain].iterrows()
        if classify_answer_set(
            str(qrow["DisplayName"]),
            answer_df[answer_df["DisplayName"] == qrow["DisplayName"]]["AnswerText"].tolist()
        )[2] == "auto-codeable"
    )
    needs_t = sum(
        1 for domain in CANONICAL_DOMAINS
        for _, qrow in question_df[question_df["Domain_clean"] == domain].iterrows()
        if classify_answer_set(
            str(qrow["DisplayName"]),
            answer_df[answer_df["DisplayName"] == qrow["DisplayName"]]["AnswerText"].tolist()
        )[2] == "needs_threshold"
    )
    needs_h = sum(
        1 for domain in CANONICAL_DOMAINS
        for _, qrow in question_df[question_df["Domain_clean"] == domain].iterrows()
        if classify_answer_set(
            str(qrow["DisplayName"]),
            answer_df[answer_df["DisplayName"] == qrow["DisplayName"]]["AnswerText"].tolist()
        )[2] == "needs_human_review"
    )

    lines += [
        "",
        f"  auto-codeable      : {auto}  questions",
        f"  needs_threshold    : {needs_t}  questions (threshold choice needed)",
        f"  needs_human_review : {needs_h}  questions (complex / not yet matched)",
        "",
        SEP,
        "  INTERPRETATION CAUTIONS",
        SEP,
        "",
        wrap(
            "1. PRESENCE ≠ NEED. A recorded response means the question was asked "
            "and answered. It does not on its own indicate a need or risk unless the "
            "AnswerText meets a positive-screen threshold."
        ),
        "",
        wrap(
            "2. REFUSALS AND SPECIAL CODES. 'Patient declined' and 'Patient unable "
            "to answer' are substantively different — declined may indicate discomfort, "
            "while unable may indicate cognitive or language barriers. Do not treat "
            "them identically when building need flags."
        ),
        "",
        wrap(
            "3. SDOH ROLLOUT WAS GRADUAL. Earlier encounters are less likely to have "
            "domain coverage. A patient with no responses in a domain may simply not "
            "have been asked yet, not have no need."
        ),
        "",
        wrap(
            "4. MULTI-QUESTION DOMAINS. Alcohol Use (AUDIT-C) requires combining Q1, "
            "Q2, and Q3 into a composite score. Physical Activity requires combining "
            "days-per-week and minutes-per-session. Do not threshold individual items "
            "from these domains without first computing the composite."
        ),
        "",
        wrap(
            "5. USE CAREFUL LANGUAGE. In all downstream analysis, say 'associated with', "
            "'observed among', or 'linked to' — not 'causes', 'leads to', or 'results in'. "
            "SDOH results should always be framed as patterns among patients with recorded "
            "responses, not conclusions about the full population."
        ),
        "",
        SEP,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Console summary (aggregate only)
# ---------------------------------------------------------------------------

def print_summary(
    counts:       dict,
    question_df:  pd.DataFrame,
    top_values_df: pd.DataFrame,
) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  SDOH ANSWER CODING AUDIT — AGGREGATE SUMMARY")
    print(sep)

    print(f"\n  File totals:")
    print(f"    Total SDOH rows          : {counts['n_rows_total']:,}")
    print(f"    Excluded                 : {counts['n_rows_excluded_either_rule']:,}")
    print(f"    Retained                 : {counts['n_rows_retained']:,}")
    print(f"    Unique patients retained : {counts['n_unique_patients_in_clean_rows']:,}")

    print(f"\n  Clean responses per domain:")
    print(f"  {'Domain':<32}  {'#Responses':>10}  {'#Questions':>10}  {'#Patients':>10}")
    print(f"  {'-'*66}")
    for domain in CANONICAL_DOMAINS:
        dom_q = question_df[question_df["Domain_clean"] == domain]
        if dom_q.empty:
            continue
        n_resp = int(dom_q["n_responses"].sum())
        n_q    = len(dom_q)
        n_pat  = int(dom_q["n_unique_patients"].max())
        print(f"  {domain:<32}  {n_resp:>10,}  {n_q:>10}  {n_pat:>10,}")

    print(f"\n  Top 5 AnswerText values per domain:")
    for domain in CANONICAL_DOMAINS:
        dom = top_values_df[top_values_df["Domain_clean"] == domain].head(5)
        if dom.empty:
            continue
        print(f"\n  {domain}:")
        for _, row in dom.iterrows():
            val_disp = str(row["AnswerText"])[:45]
            print(f"    {val_disp:<47} {row['n_responses']:>8,}  ({row['pct_within_domain']:>4.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 68)
    print("  SDOH ANSWER CODING AUDIT — DataFest 2026")
    print("=" * 68)

    # Load and clean SDOH
    print(f"\n  Loading {SDOH_PATH} …")
    _, sdoh_clean, counts = load_clean_sdoh()
    print(f"  Retained clean rows: {counts['n_rows_retained']:,}")

    # Journey cohort keys (for coverage stats)
    print(f"  Loading journey cohort patient keys from {JOURNEY_PATH} …")
    cohort_keys = load_journey_patient_keys()
    print(f"  Journey-cohort patients: {len(cohort_keys):,}")

    # Build aggregate tables
    print("\n  Building aggregate tables …")
    answer_df   = answer_counts_by_domain_question(sdoh_clean)
    question_df = question_counts_by_domain(sdoh_clean, cohort_keys)
    top_vals_df = domain_answer_top_values(sdoh_clean, top_n=15)

    # Save CSVs
    answer_df.to_csv(  AUDIT_DIR / "sdoh_answer_counts_by_domain_question.csv", index=False)
    question_df.to_csv(AUDIT_DIR / "sdoh_question_counts_by_domain.csv",        index=False)
    top_vals_df.to_csv(AUDIT_DIR / "sdoh_domain_answer_top_values.csv",         index=False)
    for fname in [
        "sdoh_answer_counts_by_domain_question.csv",
        "sdoh_question_counts_by_domain.csv",
        "sdoh_domain_answer_top_values.csv",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    # Generate and save coding notes
    print("\n  Generating coding notes …")
    notes = build_coding_notes(answer_df, question_df, counts)
    notes_path = AUDIT_DIR / "sdoh_need_coding_notes.txt"
    with open(notes_path, "w") as fh:
        fh.write(notes)
    print(f"  Saved: outputs/audit/sdoh_need_coding_notes.txt")

    # Console summary
    print_summary(counts, question_df, top_vals_df)

    print(f"""
{'='*68}
  NEXT STEP
{'='*68}

Read outputs/audit/sdoh_need_coding_notes.txt and confirm the coding
threshold for each 'needs_threshold' question before building
SDOH need/risk flags in a later script.

Key decisions to confirm before proceeding:
  - Stress: cutpoint on the 5-level scale
  - Financial Resource Strain: cutpoint on hardness scale
  - Alcohol Use: confirm AUDIT-C scoring formula and threshold
  - Physical Activity: confirm weekly-minutes threshold
  - Depression: confirm PHQ-2 and Edinburgh threshold
  - Social Connections: decide composite approach or sub-item flags
  - Housing Stability: housing moves count — threshold unclear
{'='*68}
""")


if __name__ == "__main__":
    main()
