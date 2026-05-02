"""
06_sdoh_answer_coding_audit.py
-------------------------------
Audit AnswerText values by SDOH domain and question to inform
the decision of which answers indicate a recorded need or risk.

Does NOT assign final need/risk flags. It only characterizes the
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

import pandas as pd

DATA_DIR = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR = Path("outputs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SDOH_PATH = DATA_DIR / "social_determinants.csv"
JOURNEY_PATH = TABLES_DIR / "journeys_classified_with_patient_sdoh.parquet"


# ---------------------------------------------------------------------------
# Domain normalization — keep aligned with 05_add_patient_sdoh_context.py
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
    "alcohol use": "Alcohol Use",
    "depression": "Depression",
    "financial resource strain": "Financial Resource Strain",
    "food insecurity": "Food Insecurity",
    "housing stability": "Housing Stability",
    "intimate partner violence": "Intimate Partner Violence",
    "intimate partner violance": "Intimate Partner Violence",
    "physical activity": "Physical Activity",
    "social connections": "Social Connections",
    "stress": "Stress",
    "transportation needs": "Transportation Needs",
    "utilities": "Utilities",
}


# ---------------------------------------------------------------------------
# Answer-set classifiers
# ---------------------------------------------------------------------------

# Values treated as non-substantive, refusal, system-generated, or unknown.
# Important: "Patient does not drink" is NOT listed here because it is a
# substantive Alcohol Use response, not missing/refusal.
SPECIAL_ANSWER_LOWER = {
    "patient declined",
    "patient unable to answer",
    "unknown",
    "*unknown",
    "na",
    "*not applicable",
    "",
    "never assessed",
}

# Known generic answer-set patterns.
# Question-specific rules are applied before these generic patterns.
_ANSWER_PATTERNS = [
    (
        frozenset({"yes", "no"}),
        "binary_yes_no",
        "YES = positive screen / need indicator for generic HRSN yes/no need questions. "
        "NO = not screened positive. Question-specific exceptions are handled before this rule.",
    ),
    (
        frozenset({"never true", "sometimes true", "often true"}),
        "food_insecurity_scale",
        "SOMETIMES TRUE or OFTEN TRUE = food insecurity need per standard HRSN-style coding.",
    ),
    (
        frozenset({"not hard at all", "not very hard", "somewhat hard", "hard", "very hard"}),
        "hardness_likert",
        "SOMEWHAT HARD, HARD, or VERY HARD may indicate utility / financial strain. "
        "Threshold should be stated explicitly.",
    ),
    (
        frozenset({"not at all", "only a little", "to some extent", "rather much", "very much"}),
        "stress_likert",
        "RATHER MUCH or VERY MUCH likely indicates elevated stress. "
        "TO SOME EXTENT is borderline and should be reviewed before coding.",
    ),
    (
        frozenset({"never", "monthly or less", "2-4 times a month",
                   "2-3 times a week", "4 or more times a week"}),
        "alcohol_frequency",
        "AUDIT-C Q1 frequency. Must be combined with Q2 and Q3 to compute AUDIT-C score.",
    ),
    (
        frozenset({"patient does not drink", "1 or 2", "3 or 4",
                   "5 or 6", "5 to 6", "7 to 9", "10 or more"}),
        "alcohol_quantity",
        "AUDIT-C Q2 quantity per drinking day. Must be combined with Q1 and Q3 to compute score. "
        "'Patient does not drink' is a substantive zero-risk quantity response, not missing/refusal.",
    ),
    (
        frozenset({"never", "less than monthly", "monthly", "weekly", "daily or almost daily"}),
        "alcohol_binge_frequency",
        "AUDIT-C Q3 binge frequency. Must be combined with Q1 and Q2 to compute score.",
    ),
]

_PHQ_PATTERNS = {"phq", "score", "total", "edinburgh", "epds"}


def _strip_specials(values: set) -> set:
    """Remove special/refusal values from an answer set for classification."""
    return {
        str(v).strip()
        for v in values
        if str(v).strip().lower() not in SPECIAL_ANSWER_LOWER
    }


def classify_answer_set(
    display_name: str,
    answer_values: list,
) -> tuple[str, str, str]:
    """
    Classify the answer set for one question.

    Returns:
        type_label
        coding_recommendation
        requires_review_flag:
            "auto-codeable" | "needs_threshold" | "needs_human_review"
    """
    clean = _strip_specials({str(v).strip() for v in answer_values})
    lower = {v.lower() for v in clean}
    dn = str(display_name).lower()

    # ------------------------------------------------------------------
    # Question-specific rules come before generic answer-set rules.
    # ------------------------------------------------------------------

    # Social Connections yes/no item. For these, NO may be the risk direction,
    # so generic YES = positive need would be wrong.
    if (
        ("belong" in dn or "club" in dn or "organization" in dn or "organisation" in dn)
        and lower
        and lower <= {"yes", "no"}
    ):
        return (
            "social_participation_yes_no",
            "Social participation item. NO may indicate lower social engagement, while YES indicates "
            "participation. Do not code this using the generic YES = positive-need rule.",
            "needs_human_review",
        )

    # Other Social Connections frequency items.
    if any(term in dn for term in [
        "talk on the phone",
        "get together",
        "church",
        "religious",
        "attend meetings",
        "attend religious",
    ]):
        freq_terms = {
            "never",
            "rarely",
            "sometimes",
            "often",
            "always",
            "1 to 4 times per year",
            "more than 4 times per year",
            "once a week",
            "twice a week",
            "2-3 times a week",
            "three times a week",
            "more than three times a week",
        }
        if lower & freq_terms:
            return (
                "social_connections_frequency_scale",
                "Social Connections frequency item. Lower frequency may indicate lower social connection, "
                "but the threshold is not automatic and should be reviewed before coding.",
                "needs_threshold",
            )

    # Marital status is contextual, not a direct need indicator.
    marital = {
        "married",
        "widowed",
        "divorced",
        "separated",
        "never married",
        "living with partner",
    }
    if lower & marital:
        return (
            "categorical_marital_status",
            "Marital status is a Social Connections sub-item, not a standalone need indicator. "
            "It may be useful as a contextual covariate but requires human review to use as a risk flag.",
            "needs_human_review",
        )

    # Financial Resource Strain utility-shutoff question.
    if (
        "electric, gas, oil, or water company" in dn
        or "shut off services" in dn
        or "threatened to shut off" in dn
        or "already shut off" in lower
    ) and lower and lower <= {"yes", "no", "already shut off"}:
        return (
            "utility_shutoff_yes_no",
            "Financial Resource Strain utility-shutoff item. YES or ALREADY SHUT OFF = positive screen; "
            "NO = not screened positive. Patient declined / unable / unknown should not count as need.",
            "auto-codeable",
        )

    # Rent or mortgage payment question.
    if ("mortgage" in dn or "rent" in dn) and lower and lower <= {"yes", "no"}:
        return (
            "rent_mortgage_yes_no",
            "Rent / mortgage payment item. YES = positive screen for difficulty paying housing costs; "
            "NO = not screened positive.",
            "auto-codeable",
        )

    # Homeless / shelter / steady place to sleep question.
    if any(term in dn for term in ["homeless", "shelter", "steady place to sleep"]) and lower and lower <= {"yes", "no"}:
        return (
            "housing_instability_yes_no",
            "Housing Stability item. YES = positive screen for housing instability; "
            "NO = not screened positive.",
            "auto-codeable",
        )

    # IPV questions. Keep separate because of sensitivity.
    if any(term in dn for term in [
        "partner",
        "ex-partner",
        "humiliated",
        "emotionally abused",
        "kicked",
        "hit",
        "slapped",
        "raped",
        "forced",
    ]) and lower and lower <= {"yes", "no"}:
        return (
            "ipv_yes_no",
            "Intimate Partner Violence item. YES = positive screen; NO = not screened positive. "
            "Patient declined / unable should be kept separate.",
            "auto-codeable",
        )

    # Transportation questions.
    if "transportation" in dn and lower and lower <= {"yes", "no"}:
        return (
            "transportation_yes_no",
            "Transportation Needs item. YES = positive screen; NO = not screened positive.",
            "auto-codeable",
        )

    # Depression scores.
    if any(p in dn for p in _PHQ_PATTERNS):
        return (
            "numeric_depression_score",
            "PHQ-2 threshold ≥ 3 is a standard positive screen for depression. "
            "Edinburgh / EPDS threshold ≥ 10 is common, but clinical cutoff varies by context. "
            "Requires human review before inclusion in a burden score.",
            "needs_threshold",
        )

    # Self-harm frequency item under depression.
    if "harming myself" in dn or "self-harm" in dn or "self harm" in dn:
        return (
            "self_harm_frequency",
            "Self-harm item. Any answer above NEVER may warrant separate flagging, "
            "but this should be handled carefully and separately from the general SDOH burden score.",
            "needs_threshold",
        )

    # Exercise minutes / physical activity duration.
    if ("minutes" in dn or "how many minutes" in dn) and clean:
        return (
            "numeric_exercise_minutes",
            "Duration in minutes per session. Must combine with days-per-week to compute weekly volume. "
            "Do not threshold minutes alone.",
            "needs_threshold",
        )

    # Exercise days / physical activity frequency.
    if ("days per week" in dn or "how many days" in dn) and clean:
        return (
            "numeric_exercise_days",
            "Days per week of moderate to strenuous exercise. Must combine with minutes-per-session "
            "to compute weekly physical activity volume.",
            "needs_threshold",
        )

    # Housing move count / places lived count.
    if "how many" in dn and clean and all(
        v.replace("+", "").replace(" times", "").replace(" days", "").strip().isdigit()
        for v in clean if v
    ):
        return (
            "numeric_housing_count",
            "Integer count for housing moves / places lived. Higher values may suggest housing instability. "
            "A threshold such as >= 2 can be used as a conservative flag, but should be stated clearly.",
            "needs_threshold",
        )

    # ------------------------------------------------------------------
    # Generic answer-set pattern matching.
    # ------------------------------------------------------------------
    for expected, type_label, note in _ANSWER_PATTERNS:
        expected_lower = {x.lower() for x in expected}
        if lower and lower <= expected_lower:
            if type_label in {"binary_yes_no", "food_insecurity_scale"}:
                review = "auto-codeable"
            else:
                review = "needs_threshold"
            return (type_label, note, review)

    # General frequency scale fallback.
    freq_terms = {
        "never",
        "rarely",
        "sometimes",
        "often",
        "always",
        "1 to 4 times per year",
        "more than 4 times per year",
        "once a week",
        "twice a week",
        "2-3 times a week",
        "three times a week",
        "more than three times a week",
        "less than monthly",
        "monthly",
        "weekly",
        "daily or almost daily",
    }
    if lower & freq_terms:
        return (
            "frequency_scale",
            "Ordinal frequency scale. Threshold requires human review. "
            "Do not assume a single cutpoint without domain-specific guidance.",
            "needs_threshold",
        )

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
        path,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        usecols=usecols,
    )


def normalize_numeric_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def apply_domain_corrections(domain_raw: pd.Series) -> pd.Series:
    stripped = domain_raw.astype(str).str.strip()
    lower = stripped.str.lower()
    return lower.map(DOMAIN_CORRECTION_MAP).fillna(stripped.str.title())


# ---------------------------------------------------------------------------
# 1. Load and clean SDOH
# ---------------------------------------------------------------------------

def load_clean_sdoh() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load social_determinants.csv, apply domain corrections, and exclude rows
    without usable SDOH question/domain information.
    """
    sdoh = safe_read(SDOH_PATH)
    sdoh["PatientDurableKey"] = normalize_numeric_key(sdoh["PatientDurableKey"])
    sdoh["EncounterKey"] = normalize_numeric_key(sdoh["EncounterKey"])

    n_total = len(sdoh)

    sdoh["Domain_raw"] = sdoh["Domain"].astype(str).str.strip()
    sdoh["Domain_clean"] = apply_domain_corrections(sdoh["Domain"])

    mask_unspec = sdoh["DisplayName"] == "*Unspecified"
    mask_domain_na = sdoh["Domain_raw"] == "NA"

    n_excl_unspec = int(mask_unspec.sum())
    n_excl_na = int(mask_domain_na.sum())
    n_excl_either = int((mask_unspec | mask_domain_na).sum())

    sdoh_clean = sdoh[~mask_unspec & ~mask_domain_na].copy()

    counts = {
        "n_rows_total": n_total,
        "n_rows_excluded_displayname_unspecified": n_excl_unspec,
        "n_rows_excluded_domain_na": n_excl_na,
        "n_rows_excluded_either_rule": n_excl_either,
        "n_rows_retained": len(sdoh_clean),
        "n_unique_patients_in_clean_rows": int(sdoh_clean["PatientDurableKey"].dropna().nunique()),
    }

    return sdoh, sdoh_clean, counts


# ---------------------------------------------------------------------------
# 2. Journey-cohort patient set
# ---------------------------------------------------------------------------

def load_journey_patient_keys() -> set:
    """Return the set of PatientDurableKey values in the journey cohort."""
    jdf = pd.read_parquet(JOURNEY_PATH, columns=["PatientDurableKey"])
    return set(jdf["PatientDurableKey"].dropna().unique())


# ---------------------------------------------------------------------------
# 3. Aggregate tables
# ---------------------------------------------------------------------------

def answer_counts_by_domain_question(sdoh_clean: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Domain_clean, DisplayName, AnswerText), count responses and compute
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
        ["Domain_clean", "DisplayName", "n_responses"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def question_counts_by_domain(
    sdoh_clean: pd.DataFrame,
    cohort_keys: set,
) -> pd.DataFrame:
    """
    For each (Domain_clean, DisplayName), count total responses, unique patients,
    unique encounters, and unique-patient coverage within the journey cohort.
    """
    grp = (
        sdoh_clean
        .groupby(["Domain_clean", "DisplayName"], dropna=False)
        .agg(
            n_responses=("AnswerText", "count"),
            n_unique_patients=("PatientDurableKey", "nunique"),
            n_unique_encounters=("EncounterKey", "nunique"),
        )
        .reset_index()
    )

    def cohort_patients(subdf: pd.DataFrame) -> int:
        if len(cohort_keys) == 0:
            return 0
        return int(
            subdf.loc[
                subdf["PatientDurableKey"].isin(cohort_keys),
                "PatientDurableKey",
            ]
            .dropna()
            .nunique()
        )

    cohort_counts = (
        sdoh_clean
        .groupby(["Domain_clean", "DisplayName"], dropna=False)
        .apply(cohort_patients, include_groups=False)
        .reset_index(name="n_journey_cohort_patients")
    )

    grp = grp.merge(cohort_counts, on=["Domain_clean", "DisplayName"], how="left")

    if len(cohort_keys) > 0:
        grp["pct_cohort_patients"] = (
            grp["n_journey_cohort_patients"] / len(cohort_keys) * 100
        ).round(2)
    else:
        grp["pct_cohort_patients"] = 0.0

    return grp.sort_values(
        ["Domain_clean", "n_responses"],
        ascending=[True, False],
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

    return (
        grp.sort_values(["Domain_clean", "n_responses"], ascending=[True, False])
        .groupby("Domain_clean")
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. Coding notes generator
# ---------------------------------------------------------------------------

def _classify_for_question(domain: str, qname: str, answer_df: pd.DataFrame) -> tuple[str, str, str]:
    q_answers = answer_df[
        (answer_df["Domain_clean"] == domain)
        & (answer_df["DisplayName"] == qname)
    ]["AnswerText"].tolist()
    return classify_answer_set(qname, q_answers)


def build_coding_notes(
    answer_df: pd.DataFrame,
    question_df: pd.DataFrame,
    counts: dict,
) -> str:
    """
    Generate sdoh_need_coding_notes.txt.
    Logic is data-driven from observed AnswerText values.
    """
    sep = "=" * 68
    dash = "-" * 68
    width = 66

    def wrap(text: str, indent: str = "  ") -> str:
        return textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
        )

    lines = [
        sep,
        "  SDOH ANSWER CODING AUDIT — NEED/RISK CODING NOTES",
        "  Generated by: code/06_sdoh_answer_coding_audit.py",
        sep,
        "",
        "  PURPOSE",
        dash,
        wrap(
            "This file summarizes the observed AnswerText distributions for each SDOH domain "
            "and question, and recommends a coding approach for later need/risk flags. "
            "No final need/risk flags are assigned here."
        ),
        "",
        "  EXCLUSIONS APPLIED",
        dash,
        f"  Total SDOH rows in file                  : {counts['n_rows_total']:,}",
        f"  Excluded (DisplayName = '*Unspecified')  : {counts['n_rows_excluded_displayname_unspecified']:,}",
        f"  Excluded (Domain = 'NA')                 : {counts['n_rows_excluded_domain_na']:,}",
        f"  Excluded (either rule, may overlap)      : {counts['n_rows_excluded_either_rule']:,}",
        f"  Retained for audit                       : {counts['n_rows_retained']:,}",
        f"  Unique patients in retained rows         : {counts['n_unique_patients_in_clean_rows']:,}",
        "",
    ]

    for domain in CANONICAL_DOMAINS:
        dom_q = question_df[question_df["Domain_clean"] == domain]
        dom_ans = answer_df[answer_df["Domain_clean"] == domain]

        if dom_q.empty:
            lines += [sep, f"  DOMAIN: {domain}  [NO DATA IN FILE]", ""]
            continue

        n_dom_responses = int(dom_q["n_responses"].sum())
        n_dom_patients_max = int(dom_q["n_unique_patients"].max())

        lines += [
            sep,
            f"  DOMAIN: {domain}",
            dash,
            f"  Total responses in domain             : {n_dom_responses:,}",
            f"  Max unique patients across questions  : {n_dom_patients_max:,}",
            "",
        ]

        for _, qrow in dom_q.iterrows():
            qname = str(qrow["DisplayName"])
            q_ans = dom_ans[dom_ans["DisplayName"] == qname].sort_values(
                "n_responses",
                ascending=False,
            )
            observed_answers = q_ans["AnswerText"].tolist()
            type_label, coding_note, review_flag = classify_answer_set(qname, observed_answers)

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
                "    Observed answers (top 12, n = count):",
            ]

            for _, arow in q_ans.head(12).iterrows():
                atext = str(arow["AnswerText"])
                atext_disp = atext if len(atext) <= 48 else atext[:45] + "…"
                lines.append(
                    f"      {atext_disp:<50} {arow['n_responses']:>8,}  "
                    f"({arow['pct_within_question']:>5.1f}%)"
                )

            special_rows = q_ans[
                q_ans["AnswerText"].astype(str).str.lower().str.strip().isin(SPECIAL_ANSWER_LOWER)
            ]
            if not special_rows.empty:
                n_special = int(special_rows["n_responses"].sum())
                pct_spec = (
                    round(n_special / qrow["n_responses"] * 100, 1)
                    if qrow["n_responses"] else 0.0
                )
                lines.append(
                    f"\n    Special/refusal values total: {n_special:,}  "
                    f"({pct_spec}% of question responses)"
                )

            lines += [
                "",
                wrap(f"    CODING NOTE: {coding_note}", indent="    "),
                "",
            ]

    # Summary table.
    lines += [
        sep,
        "  CODING DECISION SUMMARY TABLE",
        dash,
        f"  {'Domain':<30}  {'Review status':<22}  {'#Resp':>8}",
        f"  {'-' * 62}",
    ]

    review_counts = {
        "auto-codeable": 0,
        "needs_threshold": 0,
        "needs_human_review": 0,
    }

    for domain in CANONICAL_DOMAINS:
        dom_q = question_df[question_df["Domain_clean"] == domain]
        for _, qrow in dom_q.iterrows():
            qname = str(qrow["DisplayName"])
            _, _, review_flag = _classify_for_question(domain, qname, answer_df)
            review_counts[review_flag] = review_counts.get(review_flag, 0) + 1

            lines.append(
                f"  {domain[:28]:<30}  {review_flag:<22}  "
                f"{qrow['n_responses']:>8,}"
            )

    lines += [
        "",
        f"  auto-codeable      : {review_counts.get('auto-codeable', 0)}  questions",
        f"  needs_threshold    : {review_counts.get('needs_threshold', 0)}  questions",
        f"  needs_human_review : {review_counts.get('needs_human_review', 0)}  questions",
        "",
        sep,
        "  INTERPRETATION CAUTIONS",
        sep,
        "",
        wrap(
            "1. PRESENCE ≠ NEED. A recorded response means the question was asked and answered. "
            "It does not by itself indicate need or risk unless the AnswerText meets a positive-screen threshold."
        ),
        "",
        wrap(
            "2. REFUSALS AND SPECIAL CODES. 'Patient declined' and 'Patient unable to answer' "
            "are substantively different from negative responses. Keep them separate from positive-screen flags."
        ),
        "",
        wrap(
            "3. SDOH ROLLOUT WAS GRADUAL. Earlier encounters may have less domain coverage. "
            "A patient with no response in a domain may simply not have been asked."
        ),
        "",
        wrap(
            "4. MULTI-QUESTION DOMAINS. Alcohol Use requires AUDIT-C scoring. Physical Activity "
            "requires combining days-per-week and minutes-per-session. Depression requires PHQ / EPDS thresholding. "
            "Social Connections requires a composite or reviewed sub-item approach."
        ),
        "",
        wrap(
            "5. USE CAREFUL LANGUAGE. Say 'associated with', 'observed among', or 'linked to'. "
            "Do not use 'causes', 'leads to', or 'results in'."
        ),
        "",
        sep,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Console summary
# ---------------------------------------------------------------------------

def print_summary(
    counts: dict,
    question_df: pd.DataFrame,
    top_values_df: pd.DataFrame,
) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  SDOH ANSWER CODING AUDIT — AGGREGATE SUMMARY")
    print(sep)

    print("\n  File totals:")
    print(f"    Total SDOH rows          : {counts['n_rows_total']:,}")
    print(f"    Excluded                 : {counts['n_rows_excluded_either_rule']:,}")
    print(f"    Retained                 : {counts['n_rows_retained']:,}")
    print(f"    Unique patients retained : {counts['n_unique_patients_in_clean_rows']:,}")

    print("\n  Clean responses per domain:")
    print(f"  {'Domain':<32}  {'#Responses':>10}  {'#Questions':>10}  {'#Patients':>10}")
    print(f"  {'-' * 66}")

    for domain in CANONICAL_DOMAINS:
        dom_q = question_df[question_df["Domain_clean"] == domain]
        if dom_q.empty:
            continue

        n_resp = int(dom_q["n_responses"].sum())
        n_q = len(dom_q)
        n_pat = int(dom_q["n_unique_patients"].max())

        print(f"  {domain:<32}  {n_resp:>10,}  {n_q:>10}  {n_pat:>10,}")

    print("\n  Top 5 AnswerText values per domain:")
    for domain in CANONICAL_DOMAINS:
        dom = top_values_df[top_values_df["Domain_clean"] == domain].head(5)
        if dom.empty:
            continue

        print(f"\n  {domain}:")
        for _, row in dom.iterrows():
            val_disp = str(row["AnswerText"])[:45]
            print(
                f"    {val_disp:<47} {row['n_responses']:>8,}  "
                f"({row['pct_within_domain']:>4.1f}%)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 68)
    print("  SDOH ANSWER CODING AUDIT — DataFest 2026")
    print("=" * 68)

    print(f"\n  Loading {SDOH_PATH} …")
    _, sdoh_clean, counts = load_clean_sdoh()
    print(f"  Retained clean rows: {counts['n_rows_retained']:,}")

    print(f"  Loading journey cohort patient keys from {JOURNEY_PATH} …")
    cohort_keys = load_journey_patient_keys()
    print(f"  Journey-cohort patients: {len(cohort_keys):,}")

    print("\n  Building aggregate tables …")
    answer_df = answer_counts_by_domain_question(sdoh_clean)
    question_df = question_counts_by_domain(sdoh_clean, cohort_keys)
    top_vals_df = domain_answer_top_values(sdoh_clean, top_n=15)

    answer_df.to_csv(AUDIT_DIR / "sdoh_answer_counts_by_domain_question.csv", index=False)
    question_df.to_csv(AUDIT_DIR / "sdoh_question_counts_by_domain.csv", index=False)
    top_vals_df.to_csv(AUDIT_DIR / "sdoh_domain_answer_top_values.csv", index=False)

    for fname in [
        "sdoh_answer_counts_by_domain_question.csv",
        "sdoh_question_counts_by_domain.csv",
        "sdoh_domain_answer_top_values.csv",
    ]:
        print(f"  Saved: outputs/audit/{fname}")

    print("\n  Generating coding notes …")
    notes = build_coding_notes(answer_df, question_df, counts)
    with open(AUDIT_DIR / "sdoh_need_coding_notes.txt", "w") as fh:
        fh.write(notes)
    print("  Saved: outputs/audit/sdoh_need_coding_notes.txt")

    print_summary(counts, question_df, top_vals_df)

    print(f"""
{'=' * 68}
  NEXT STEP
{'=' * 68}

Read outputs/audit/sdoh_need_coding_notes.txt and confirm the coding
threshold for each 'needs_threshold' question before adding those
domains to a final SDOH burden score.

Current conservative core flags in later scripts should remain:
  - Transportation Needs
  - Food Insecurity
  - Housing Stability
  - Financial Resource Strain
  - Intimate Partner Violence
  - Utilities

Domains still requiring threshold/composite review:
  - Stress: cutpoint on the 5-level scale
  - Alcohol Use: AUDIT-C scoring formula and threshold
  - Physical Activity: weekly-minutes threshold
  - Depression: PHQ-2 and Edinburgh / EPDS threshold
  - Social Connections: composite approach or sub-item flags
{'=' * 68}
""")


if __name__ == "__main__":
    main()