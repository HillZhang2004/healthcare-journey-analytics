"""
11_make_pipeline_aligned_visualizations.py
--------------------------------------------
Pipeline-aligned, presentation-ready plots for DataFest 2026.

Corrects methodology from earlier team drafts:
  - SDOH plots use screened-only outputs; unscreened ≠ "no need"
  - Diagnosis join uses clean 1-to-1 DiagnosisKey→DiagnosisValue mapping
  - High-acuity follow-up uses Date, not sparse AdmissionInstant
  - Flag labels use exact column names from encounters.csv
  - All outputs are aggregate only; no patient/encounter identifiers

Framing: utilisation, follow-up timing, operational bottleneck.
Not direct quality-of-care measurement. Descriptive, non-causal.

Run from project root:
    python code/11_make_pipeline_aligned_visualizations.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR   = Path("data")
TABLES_DIR = Path("outputs/tables")
AUDIT_DIR  = Path("outputs/audit")
VIZ_DIR    = Path("outputs/visualizations")

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DPI   = 150
PLOT_STYLE = "whitegrid"

SCREENED_GROUP_ORDER = [
    "screened_no_positive_need",
    "1 need",
    "2-3 needs",
    "4+ needs",
]

SCREENED_GROUP_LABELS = {
    "screened_no_positive_need": "Screened,\nno positive need",
    "1 need":                    "1 need",
    "2-3 needs":                 "2–3 needs",
    "4+ needs":                  "4+ needs",
}

CORE_NEED_DISPLAY = {
    "transportation_need":       "Transportation",
    "food_insecurity_need":      "Food insecurity",
    "housing_instability_need":  "Housing instability",
    "financial_strain_need":     "Financial strain",
    "ipv_need":                  "IPV positive screen",
    "utilities_need":            "Utilities hardship",
}

MYCHART_INCLUDE = ["Activated", "Pending Activation", "Inactivated", "Patient Declined"]

ENC_USECOLS = [
    "PatientDurableKey",
    "PrimaryDiagnosisKey",
    "Date",
    "IsEdVisit",
    "IsInpatientAdmission",
    "IsHospitalAdmission",
    "IsHospitalOutpatientVisit",
    "IsOutpatientFaceToFaceVisit",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False, **kwargs)


def normalize_numeric_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _pct(num, den, d: int = 2) -> float:
    return round(num / den * 100, d) if den else 0.0


def _bar_labels(ax, fmt="{:.1f}%", fontsize=8, pad=1):
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize,
            )


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# A. Preflight checks
# ---------------------------------------------------------------------------

def run_preflight_checks() -> None:
    rows = []

    def chk(cid, desc, ok, detail=""):
        rows.append({"check_id": cid, "description": desc,
                     "status": "PASS" if ok else "FAIL", "detail": detail})

    # 1. SDOH plots use screened-only CSV (not full journey parquet unfiltered)
    sdoh_path = AUDIT_DIR / "validation_screened_sdoh_hospital_share.csv"
    sdoh_ok   = sdoh_path.exists()
    if sdoh_ok:
        df = pd.read_csv(sdoh_path)
        has_screened_group = "screened_core_need_group" in df["stratification"].values
        chk(1, "SDOH plots use screened-only output (not unscreened as 'no need')",
            has_screened_group,
            f"stratification values: {sorted(df['stratification'].unique())}")
    else:
        chk(1, "SDOH plots use screened-only output", False,
            f"missing: {sdoh_path}")

    # 2. Diagnosis plots use clean 1-to-1 mapping
    diag_path = DATA_DIR / "diagnosis.csv"
    enc_path  = DATA_DIR / "encounters.csv"
    if diag_path.exists():
        diag = safe_read(diag_path)
        diag["_dk"] = normalize_numeric_key(diag["DiagnosisKey"])
        n_ambiguous = (
            diag.groupby("_dk")["DiagnosisValue"].nunique().pipe(lambda s: s[s > 1]).sum()
        )
        chk(2, "Diagnosis group plot uses clean 1-to-1 mapping (not naive join)",
            True,
            f"Ambiguous DiagnosisKey rows in source: {n_ambiguous:,} — "
            "script excludes these before plotting")
    else:
        chk(2, "Diagnosis group plot uses clean 1-to-1 mapping", False,
            f"missing: {diag_path}")

    # 3. High-acuity follow-up uses Date (not sparse AdmissionInstant)
    if enc_path.exists():
        sample = safe_read(enc_path, usecols=["Date", "AdmissionInstant"], nrows=50000)
        date_ok_rate = (
            pd.to_datetime(sample["Date"], format="%m/%d/%y", errors="coerce")
            .notna().mean()
        )
        admit_ok_rate = (
            pd.to_datetime(sample["AdmissionInstant"], errors="coerce")
            .notna().mean()
        )
        chk(3, "High-acuity follow-up timing uses Date (not AdmissionInstant alone)",
            date_ok_rate > 0.99,
            f"Date parse rate: {date_ok_rate*100:.1f}%  "
            f"AdmissionInstant fill rate: {admit_ok_rate*100:.1f}%")
    else:
        chk(3, "High-acuity follow-up uses Date", False, f"missing: {enc_path}")

    # 4. High-utilizer flag names match actual encounter columns
    if enc_path.exists():
        enc_cols = safe_read(enc_path, nrows=0).columns.tolist()
        expected = ["IsEdVisit", "IsInpatientAdmission", "IsHospitalAdmission",
                    "IsHospitalOutpatientVisit", "IsOutpatientFaceToFaceVisit"]
        missing  = [c for c in expected if c not in enc_cols]
        chk(4, "High-utilizer plot uses correct flag column names",
            len(missing) == 0,
            f"All expected flags present" if not missing else f"Missing: {missing}")
    else:
        chk(4, "Flag column names verified", False, f"missing: {enc_path}")

    # 5. Outputs aggregate-only (no patient/encounter identifiers in viz dir)
    id_cols = {"PatientDurableKey", "EncounterKey", "DiagnosisKey"}
    id_found = []
    for csv in AUDIT_DIR.glob("q*.csv"):
        df = pd.read_csv(csv)
        hits = id_cols & set(df.columns)
        if hits:
            id_found.append(f"{csv.name}: {hits}")
    chk(5, "Aggregate output CSVs contain no patient/encounter identifiers",
        len(id_found) == 0,
        "Clean" if not id_found else "; ".join(id_found))

    # 6. Required input files exist
    required = [
        TABLES_DIR / "journeys_with_core_sdoh_need_flags.parquet",
        AUDIT_DIR  / "validation_screened_sdoh_hospital_share.csv",
        AUDIT_DIR  / "ed_followup_by_screened_need_group.csv",
        AUDIT_DIR  / "ed_followup_after_index_by_screening.csv",
        DATA_DIR   / "encounters.csv",
        DATA_DIR   / "diagnosis.csv",
        DATA_DIR   / "patients.csv",
    ]
    missing_files = [str(p) for p in required if not p.exists()]
    chk(6, "All required input files present",
        len(missing_files) == 0,
        "All present" if not missing_files else "Missing: " + ", ".join(missing_files))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(AUDIT_DIR / "plot_preflight_checks.csv", index=False)
    n_fail = (df_out["status"] == "FAIL").sum()
    print(f"  Preflight: {len(rows)} checks, {n_fail} failed.")
    if n_fail:
        for r in rows:
            if r["status"] == "FAIL":
                print(f"    FAIL [{r['check_id']}] {r['description']}: {r['detail']}")


# ---------------------------------------------------------------------------
# B. Load raw encounter data (shared across plots 1, 3, 5, 6)
# ---------------------------------------------------------------------------

def load_encounters() -> pd.DataFrame:
    enc = safe_read(DATA_DIR / "encounters.csv", usecols=ENC_USECOLS)
    enc["_date"] = pd.to_datetime(enc["Date"], format="%m/%d/%y", errors="coerce")
    enc["PatientDurableKey"] = normalize_numeric_key(enc["PatientDurableKey"])
    for col in ["IsEdVisit", "IsInpatientAdmission", "IsHospitalAdmission",
                "IsHospitalOutpatientVisit", "IsOutpatientFaceToFaceVisit"]:
        enc[col] = enc[col].astype(int).astype(bool)
    enc = enc[enc["_date"].notna()].copy()
    print(f"  Encounters loaded: {len(enc):,}  ({enc['_date'].dt.year.value_counts().sort_index().to_dict()})")
    return enc


def build_clean_map() -> pd.DataFrame:
    """Return clean 1-to-1 DiagnosisKey→DiagnosisValue/GroupName mapping."""
    diag = safe_read(DATA_DIR / "diagnosis.csv")
    diag["_dk"] = normalize_numeric_key(diag["DiagnosisKey"])
    clean_dk = (
        diag.groupby("_dk")["DiagnosisValue"]
        .nunique().pipe(lambda s: s[s == 1]).index
    )
    clean_key_set = set(clean_dk.dropna())
    cmap = (
        diag[diag["_dk"].isin(clean_key_set)]
        [["_dk", "DiagnosisValue", "GroupName"]]
        .drop_duplicates("_dk")
    )
    print(f"  Clean diagnosis keys: {len(cmap):,}")
    return cmap


# ---------------------------------------------------------------------------
# C. Compute high-acuity follow-up (shared between plots 1 and 3)
# ---------------------------------------------------------------------------

def compute_ha_followup(enc: pd.DataFrame) -> pd.DataFrame:
    """
    For each high-acuity event (IsEdVisit or IsInpatientAdmission),
    find the first outpatient face-to-face visit strictly after it.

    merge_asof requires the join key to be globally sorted (not per-group).
    We sort by date first, then patient as a tiebreaker, which satisfies
    that requirement while keeping per-patient ordering correct.
    If merge_asof still fails, falls back to per-patient np.searchsorted.
    """
    ha  = enc[enc["IsEdVisit"] | enc["IsInpatientAdmission"]].copy()
    ftf = enc[enc["IsOutpatientFaceToFaceVisit"]].copy()

    ha["ha_type"] = np.where(ha["IsEdVisit"] & ~ha["IsInpatientAdmission"], "ed_only",
                    np.where(~ha["IsEdVisit"] & ha["IsInpatientAdmission"], "inpatient_only",
                             "ed_and_inpatient"))

    # Shift by 1 day to find strictly-later FTF events
    ha["_search"] = ha["_date"] + pd.Timedelta(days=1)

    # Drop rows with missing join keys before sorting
    ha  = ha.dropna(subset=["_search",  "PatientDurableKey"]).copy()
    ftf = ftf.dropna(subset=["_date", "PatientDurableKey"]).copy()

    # Sort by the join key first (date), then patient as tiebreaker.
    # This satisfies merge_asof's global-sort requirement while keeping
    # per-patient dates in order within each group.
    ha_s  = ha.sort_values(["_search",  "PatientDurableKey"],
                            kind="mergesort").reset_index(drop=True)
    ftf_s = (
        ftf[["PatientDurableKey", "_date"]]
        .rename(columns={"_date": "_ftf_date"})
        .sort_values(["_ftf_date", "PatientDurableKey"], kind="mergesort")
        .reset_index(drop=True)
    )

    try:
        matched = pd.merge_asof(
            ha_s[["PatientDurableKey", "_date", "_search", "ha_type"]],
            ftf_s,
            left_on="_search", right_on="_ftf_date",
            by="PatientDurableKey",
            direction="forward",
        )
    except ValueError:
        # Fallback: per-patient np.searchsorted (reliable, O(P·log F))
        print("  merge_asof failed; using per-patient searchsorted fallback …")

        ftf_dates_by_pat = {
            pat: np.sort(grp["_date"].values)
            for pat, grp in ftf.groupby("PatientDurableKey")
        }
        chunks = []
        for pat, grp in ha.groupby("PatientDurableKey"):
            grp = grp.copy()
            dates = ftf_dates_by_pat.get(pat)
            if dates is None or len(dates) == 0:
                grp["_ftf_date"] = pd.NaT
            else:
                search_arr = grp["_search"].values
                idxs       = np.searchsorted(dates, search_arr, side="left")
                grp["_ftf_date"] = pd.array(
                    np.where(idxs < len(dates), dates[idxs], np.datetime64("NaT")),
                    dtype="datetime64[ns]",
                )
            chunks.append(
                grp[["PatientDurableKey", "_date", "_search", "ha_type", "_ftf_date"]]
            )
        matched = pd.concat(chunks, ignore_index=True)

    matched["days_to_ftf"] = (matched["_ftf_date"] - matched["_date"]).dt.days
    matched["fu_7d"]  = matched["days_to_ftf"].notna() & (matched["days_to_ftf"] <= 7)
    matched["fu_14d"] = matched["days_to_ftf"].notna() & (matched["days_to_ftf"] <= 14)
    matched["fu_30d"] = matched["days_to_ftf"].notna() & (matched["days_to_ftf"] <= 30)
    print(f"  High-acuity events: {len(matched):,}  "
          f"(with any FTF follow-up: {matched['fu_30d'].sum():,})")
    return matched


# ---------------------------------------------------------------------------
# Plot 1: High-acuity follow-up rates by event type and window
# ---------------------------------------------------------------------------

def plot_q1(ha_events: pd.DataFrame) -> None:
    rows = []
    for ha_label, mask in [
        ("ED visit\n(IsEdVisit=1)",        ha_events["IsEdVisit"] if "IsEdVisit" in ha_events
                                            else ha_events["ha_type"].isin(["ed_only", "ed_and_inpatient"])),
        ("Inpatient admission\n(IsInpatientAdmission=1)",
                                            ha_events["ha_type"].isin(["inpatient_only", "ed_and_inpatient"])),
    ]:
        sub = ha_events[mask]
        n   = len(sub)
        for window, col in [("7 days", "fu_7d"), ("14 days", "fu_14d"), ("30 days", "fu_30d")]:
            rows.append({
                "ha_type":     ha_label.replace("\n", " "),
                "window":      window,
                "n_events":    n,
                "n_with_fu":   int(sub[col].sum()),
                "pct_with_fu": _pct(int(sub[col].sum()), n),
            })
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "q1_high_acuity_followup_rates.csv", index=False)

    # Rebuild with correct IsEdVisit column
    ha = ha_events.copy()
    if "IsEdVisit" not in ha.columns:
        ha["IsEdVisit"] = ha["ha_type"].isin(["ed_only", "ed_and_inpatient"])

    windows = ["7 days", "14 days", "30 days"]
    cols    = ["fu_7d", "fu_14d", "fu_30d"]
    groups  = [
        ("ED visit",        ha["ha_type"].isin(["ed_only", "ed_and_inpatient"])),
        ("Inpatient admit", ha["ha_type"].isin(["inpatient_only", "ed_and_inpatient"])),
    ]

    x      = np.arange(len(windows))
    width  = 0.35
    colors = ["#2176AE", "#F4845F"]

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (lbl, mask) in enumerate(groups):
        sub    = ha[mask]
        n      = len(sub)
        pcts   = [_pct(int(sub[c].sum()), n) for c in cols]
        bars   = ax.bar(x + i * width, pcts, width, label=f"{lbl} (n={n:,})", color=colors[i])
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(windows)
    ax.set_ylabel("% of high-acuity events with outpatient\nface-to-face follow-up")
    ax.set_xlabel("Follow-up window")
    ax.set_title("Outpatient follow-up after high-acuity visits\n"
                 "(IsOutpatientFaceToFaceVisit within window, strictly after event)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(ax.get_ylim()[1], 15))
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q1_high_acuity_followup.png")


# ---------------------------------------------------------------------------
# Plot 2: Screened-only SDOH burden vs hospital/ED/observation involvement
# ---------------------------------------------------------------------------

def plot_q2() -> None:
    df = pd.read_csv(AUDIT_DIR / "validation_screened_sdoh_hospital_share.csv")
    grp = df[df["stratification"] == "screened_core_need_group"].copy()
    grp = grp.set_index("stratum").reindex(SCREENED_GROUP_ORDER).reset_index()
    grp.columns = ["group"] + list(grp.columns[1:])

    labels = [SCREENED_GROUP_LABELS.get(g, g) for g in grp["group"]]
    pcts   = grp["pct_hospital_involved"].values
    ns     = grp["n_journeys"].values

    colors = ["#B8D4E8", "#6BAED6", "#2171B5", "#08306B"]
    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct, n in zip(bars, pcts, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("% of journeys with hospital/ED/observation involvement")
    ax.set_xlabel("Screened core need group")
    ax.set_title("Acute-care involvement by recorded SDOH need burden\n"
                 "(Screened journeys only — 'no positive need' ≠ unscreened)",
                 fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.25)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q2_screened_sdoh_burden_acute_involvement.png")


# ---------------------------------------------------------------------------
# Plot 2b: Specific needs vs hospital/ED/observation involvement
# ---------------------------------------------------------------------------

def plot_q2b() -> None:
    df = pd.read_csv(AUDIT_DIR / "validation_screened_sdoh_hospital_share.csv")
    specific = df[df["stratification"].str.startswith("specific_need:")].copy()
    specific["flag"] = specific["stratification"].str.replace("specific_need:", "", regex=False)
    specific["has_need"] = specific["stratum"].str.endswith("=True")

    rows = []
    for flag, disp in CORE_NEED_DISPLAY.items():
        sub_t = specific[(specific["flag"] == flag) & specific["has_need"]]
        sub_f = specific[(specific["flag"] == flag) & ~specific["has_need"]]
        if sub_t.empty or sub_f.empty:
            continue
        rows.append({
            "display":    disp,
            "pct_with":   float(sub_t["pct_hospital_involved"].iloc[0]),
            "pct_without": float(sub_f["pct_hospital_involved"].iloc[0]),
            "n_with":     int(sub_t["n_journeys"].iloc[0]),
            "n_without":  int(sub_f["n_journeys"].iloc[0]),
        })
    pdf = pd.DataFrame(rows)

    y      = np.arange(len(pdf))
    height = 0.35
    colors = ["#2171B5", "#C6DBEF"]

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars_w  = ax.barh(y + height / 2, pdf["pct_with"],   height, label="Positive screen",    color=colors[0])
    bars_wo = ax.barh(y - height / 2, pdf["pct_without"], height, label="No positive screen", color=colors[1])

    for bar, pct in zip(bars_w, pdf["pct_with"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)
    for bar, pct in zip(bars_wo, pdf["pct_without"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(pdf["display"])
    ax.set_xlabel("% of journeys with hospital/ED/observation involvement")
    ax.set_title("Acute-care involvement by specific recorded SDOH need\n"
                 "(Screened journeys only — positive screen vs. no positive screen)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(pdf["pct_with"].max(), pdf["pct_without"].max()) * 1.2)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q2b_specific_sdoh_needs_acute_involvement.png")


# ---------------------------------------------------------------------------
# Plot 3: MyChart status vs high-acuity follow-up
# ---------------------------------------------------------------------------

def plot_q3(ha_events: pd.DataFrame) -> None:
    pat = safe_read(DATA_DIR / "patients.csv", usecols=["DurableKey", "MyChartStatus"])
    pat["PatientDurableKey"] = normalize_numeric_key(pat["DurableKey"])

    ha = ha_events.merge(
        pat[["PatientDurableKey", "MyChartStatus"]],
        on="PatientDurableKey", how="left",
    )
    ha = ha[ha["MyChartStatus"].isin(MYCHART_INCLUDE)]

    rows = []
    for status in MYCHART_INCLUDE:
        sub = ha[ha["MyChartStatus"] == status]
        n   = len(sub)
        if n == 0:
            continue
        for window, col in [("7 days", "fu_7d"), ("14 days", "fu_14d"), ("30 days", "fu_30d")]:
            rows.append({
                "mychart_status": status,
                "window":         window,
                "n_ha_events":    n,
                "n_with_fu":      int(sub[col].sum()),
                "pct_with_fu":    _pct(int(sub[col].sum()), n),
            })
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "q3_mychart_followup_rates.csv", index=False)

    windows  = ["7 days", "14 days", "30 days"]
    statuses = [s for s in MYCHART_INCLUDE
                if s in ha["MyChartStatus"].unique()]
    palette  = sns.color_palette("Blues_d", len(statuses))

    x     = np.arange(len(windows))
    width = 0.8 / len(statuses)

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (status, color) in enumerate(zip(statuses, palette)):
        sub   = ha[ha["MyChartStatus"] == status]
        n     = len(sub)
        pcts  = [_pct(int(sub[c].sum()), n) for c in ["fu_7d", "fu_14d", "fu_30d"]]
        offset = (i - len(statuses) / 2 + 0.5) * width
        bars  = ax.bar(x + offset, pcts, width * 0.9,
                       label=f"{status} (n={n:,})", color=color)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{pct:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("% of high-acuity events with\noutpatient face-to-face follow-up")
    ax.set_xlabel("Follow-up window")
    ax.set_title("Outpatient follow-up after high-acuity visits by MyChart status\n"
                 "(Descriptive — not a quality-of-care measure)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q3_mychart_followup.png")


# ---------------------------------------------------------------------------
# Plot 5: High-utilizer care-setting flag profile
# ---------------------------------------------------------------------------

def plot_q5(enc: pd.DataFrame) -> None:
    enc_per_pat = enc.groupby("PatientDurableKey").size()
    p95         = enc_per_pat.quantile(0.95)
    hi_pats     = set(enc_per_pat[enc_per_pat >= p95].index)

    flags = {
        "ED visit\n(IsEdVisit)":              "IsEdVisit",
        "Inpatient admission\n(IsInpatientAdmission)": "IsInpatientAdmission",
        "Hospital admission\n(IsHospitalAdmission)":   "IsHospitalAdmission",
        "Hospital outpatient\n(IsHospitalOutpatientVisit)": "IsHospitalOutpatientVisit",
        "Outpatient face-to-face\n(IsOutpatientFaceToFaceVisit)": "IsOutpatientFaceToFaceVisit",
    }

    rows = []
    groups = [("High utilizer\n(top 5%, ≥{:.0f} enc)".format(p95), enc["PatientDurableKey"].isin(hi_pats)),
              ("Other patients\n(bottom 95%)",                      ~enc["PatientDurableKey"].isin(hi_pats))]
    for grp_lbl, mask in groups:
        sub = enc[mask]
        n   = len(sub)
        for disp, col in flags.items():
            pct = _pct(int(sub[col].sum()), n)
            rows.append({
                "patient_group": grp_lbl.replace("\n", " "),
                "flag":          col,
                "n_encounters":  n,
                "n_flag":        int(sub[col].sum()),
                "pct_flag":      pct,
            })
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "q5_high_utilizer_care_setting_profile.csv", index=False)

    flag_labels = list(flags.keys())
    y       = np.arange(len(flag_labels))
    height  = 0.35
    colors  = ["#D62728", "#1F77B4"]

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (grp_lbl, mask) in enumerate(groups):
        sub  = enc[mask]
        n    = len(sub)
        pcts = [_pct(int(sub[col].sum()), n) for col in flags.values()]
        offset = (i - 0.5) * height
        bars = ax.barh(y + offset, pcts, height * 0.9,
                       label=f"{grp_lbl.replace(chr(10), ' ')} (n={n:,})",
                       color=colors[i], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(flag_labels, fontsize=9)
    ax.set_xlabel("% of encounters with flag (flags overlap — not a composition)")
    ax.set_title(f"Care-setting flag profile: high utilizers (top 5%, ≥{p95:.0f} enc) vs others\n"
                 "(Each flag counted independently — % may exceed 100% in sum)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(df["pct_flag"]) * 1.2)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q5_high_utilizer_care_setting_profile.png")


# ---------------------------------------------------------------------------
# Plot 6: Clean diagnosis-group bubble chart
# ---------------------------------------------------------------------------

def plot_q6() -> None:
    jdf = pd.read_parquet(
        TABLES_DIR / "journeys_with_core_sdoh_need_flags.parquet",
        columns=["GroupName", "n_encounters", "has_ed",
                 "first_to_second_gap_days", "median_gap_days"],
    )

    # Exclude unusable GroupName values
    jdf = jdf[
        jdf["GroupName"].notna()
        & (jdf["GroupName"] != "")
        & (~jdf["GroupName"].str.startswith("*", na=False))
        & (~jdf["GroupName"].str.startswith("Reserved for", na=False))
    ].copy()

    # Encounter volume and ED journey share per GroupName
    base = jdf.groupby("GroupName").agg(
        total_encounters=("n_encounters", "sum"),
        n_journeys=("n_encounters", "count"),
        pct_ed_journeys=("has_ed", "mean"),
    ).reset_index()
    base["pct_ed_journeys"] *= 100

    # Median gap from multi-encounter journeys only
    gap = (
        jdf[jdf["n_encounters"] > 1]
        .groupby("GroupName")["first_to_second_gap_days"]
        .median()
        .rename("median_gap_days")
        .reset_index()
    )
    bubble = base.merge(gap, on="GroupName", how="left")

    # Filter: min 500 encounters, have a gap value, top 30 by volume
    bubble = bubble[bubble["total_encounters"] >= 500].dropna(subset=["median_gap_days"])
    bubble = bubble.nlargest(30, "total_encounters").reset_index(drop=True)

    bubble.to_csv(AUDIT_DIR / "q6_clean_diagnosis_bubble_table.csv", index=False)

    # --- plot ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    enc_max  = bubble["total_encounters"].max()
    sizes    = (bubble["total_encounters"] / enc_max * 1200).clip(lower=50)
    norm     = mcolors.Normalize(vmin=bubble["pct_ed_journeys"].min(),
                                  vmax=bubble["pct_ed_journeys"].max())
    cmap     = cm.get_cmap("YlOrRd")
    colors   = [cmap(norm(v)) for v in bubble["pct_ed_journeys"]]

    sc = ax.scatter(
        bubble["median_gap_days"], bubble["pct_ed_journeys"],
        s=sizes, c=colors, alpha=0.75, edgecolors="grey", linewidths=0.4,
    )

    # Label top 8 by encounter volume + top 3 by ED share not already labeled
    top_vol = set(bubble.nlargest(8, "total_encounters").index)
    top_ed  = set(bubble.nlargest(3, "pct_ed_journeys").index) - top_vol
    for idx in (top_vol | top_ed):
        row  = bubble.loc[idx]
        name = row["GroupName"]
        name = name[:35] + "…" if len(name) > 35 else name
        ax.annotate(
            name,
            (row["median_gap_days"], row["pct_ed_journeys"]),
            fontsize=7, xytext=(5, 4), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("% journeys with any ED visit", fontsize=9)

    ax.set_xlabel("Median first-to-second encounter gap (days)\namong multi-encounter journeys")
    ax.set_ylabel("% of journeys with any ED visit")
    ax.set_title("Clean diagnosis group: encounter volume, ED involvement, and follow-up gap\n"
                 "(bubble size = encounter volume; clean 1-to-1 diagnosis mapping)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q6_clean_diagnosis_bubble.png")


# ---------------------------------------------------------------------------
# Plot 7: ED-indexed follow-up by screened SDOH burden
# ---------------------------------------------------------------------------

def plot_q7() -> None:
    ng = pd.read_csv(AUDIT_DIR / "ed_followup_by_screened_need_group.csv")
    sc = pd.read_csv(AUDIT_DIR / "ed_followup_after_index_by_screening.csv")

    ng = ng.set_index("screened_core_need_group").reindex(SCREENED_GROUP_ORDER).reset_index()
    labels = [SCREENED_GROUP_LABELS.get(g, g) for g in ng["screened_core_need_group"]]
    ra30   = ng["pct_repeat_acute_care_30d"].values
    med_d  = ng["median_days_to_next_encounter_after_ed"].values
    ns     = ng["n_ed_journeys"].values

    colors = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    sns.set_style(PLOT_STYLE)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    bars = ax1.bar(labels, ra30, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct, n in zip(bars, ra30, ns):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{pct:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8.5)

    ax1.set_ylabel("% repeat acute-care event within 30 days\nof first ED visit")
    ax1.set_xlabel("Screened core need group (screened journeys only)")
    ax1.set_ylim(0, max(ra30) * 1.35)

    # Overlay median days to next encounter as a line
    ax2 = ax1.twinx()
    valid_d = np.where(np.isnan(med_d), 0, med_d)
    ax2.plot(labels, valid_d, "o--", color="#D62728", linewidth=1.8,
             markersize=7, label="Median days to next enc.")
    ax2.set_ylabel("Median days to next encounter\nafter first ED visit", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")
    ax2.set_ylim(0, max(valid_d) * 1.6)

    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, fontsize=8, loc="upper right")

    ax1.set_title("Post-ED follow-up and repeat acute care by recorded SDOH need burden\n"
                  "(Screened journeys with ED visit — descriptive, non-causal)",
                  fontsize=10)
    fig.tight_layout()
    _save(fig, VIZ_DIR / "q7_ed_indexed_followup_by_sdoh_burden.png")


# ---------------------------------------------------------------------------
# Notes file
# ---------------------------------------------------------------------------

def write_notes() -> None:
    import textwrap
    SEP  = "=" * 68
    DASH = "-" * 68
    W    = 66

    def wrap(t):
        return textwrap.fill(t, width=W, initial_indent="  ", subsequent_indent="  ")

    lines = [
        SEP,
        "  PIPELINE-ALIGNED VISUALIZATION NOTES",
        "  Generated by: code/11_make_pipeline_aligned_visualizations.py",
        SEP,
        "",
        "  FRAMING",
        DASH,
        wrap(
            "These plots are pipeline-aligned corrections, not exact copies of "
            "teammate drafts. They describe utilisation, follow-up timing, and "
            "operational bottleneck patterns observed in the data. They are NOT "
            "direct quality-of-care measurements. Use 'associated with', "
            "'observed among', or 'linked to' — never 'causes' or 'leads to'."
        ),
        "",
        "  Q2 REPLACEMENT — SDOH BURDEN",
        DASH,
        wrap(
            "The original Q2 plot treated unscreened patients as the 'no need' "
            "group. This is incorrect: unscreened patients have no recorded SDOH "
            "data — they cannot be treated as having no social needs. "
            "The corrected plot uses only screened journeys and labels the "
            "zero-burden group 'screened with no recorded positive core need'. "
            "Source: validation_screened_sdoh_hospital_share.csv (script 09)."
        ),
        "",
        "  Q6 CORRECTION — DIAGNOSIS GROUPS",
        DASH,
        wrap(
            "A naive join on PrimaryDiagnosisKey = DiagnosisKey can inflate "
            "encounter counts when DiagnosisKey maps to multiple DiagnosisValues. "
            "The corrected bubble chart uses the same clean 1-to-1 mapping built "
            "in 03_build_journeys.py: only DiagnosisKeys with exactly one unique "
            "DiagnosisValue are used. GroupName values starting with 'Reserved for' "
            "or '*' are excluded as unusable."
        ),
        "",
        "  Q1/Q3 — HIGH-ACUITY FOLLOW-UP TIMING",
        DASH,
        wrap(
            "Follow-up timing uses the Date column (parsed as MM/DD/YY), not "
            "AdmissionInstant, because AdmissionInstant is sparse (fill rate << 50%). "
            "Same-calendar-day encounters are excluded from 'follow-up' (strictly "
            "after, not same-day), because same-day events may be part of the same "
            "care episode. A 1-day offset is applied to the search key before "
            "merge_asof so that only encounters on later dates are matched."
        ),
        "",
        "  SDOH PLOTS — SCREENED SUBSET ONLY",
        DASH,
        wrap(
            "All SDOH burden comparisons (Q2, Q2b, Q7) are restricted to journeys "
            "linked to patients with at least one clean SDOH screening response. "
            "A 'no positive screen' patient is not the same as an unscreened patient. "
            "Results cannot be generalised to the full journey cohort without "
            "acknowledging selection into SDOH screening."
        ),
        "",
        "  Q7 — ED-INDEXED RESULTS",
        DASH,
        wrap(
            "ED-indexed journeys come from script 10 (238,966 clean journeys). "
            "Only 36% of ED-indexed journeys are screened, so the SDOH-stratified "
            "results represent a selected subset. The repeat-acute-care metric "
            "counts any later ED/hospital/inpatient/observation event within 30 days "
            "strictly after the first ED visit date."
        ),
        "",
        "  INTERPRETATION REQUIREMENTS",
        DASH,
        "  Use: 'associated with', 'observed among', 'linked to', 'suggests'",
        "  Do not use: 'causes', 'leads to', 'results in', 'quality of care'",
        "",
        wrap(
            "For SDOH findings: 'screened patients with a recorded positive "
            "core need' and 'screened with no recorded positive core need'. "
            "Never 'patients with no social needs'."
        ),
        "",
        SEP,
    ]
    with open(AUDIT_DIR / "pipeline_aligned_visualization_notes.txt", "w") as fh:
        fh.write("\n".join(lines))
    print("  Saved: pipeline_aligned_visualization_notes.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEP = "=" * 68
    print(f"\n{SEP}")
    print("  PIPELINE-ALIGNED VISUALIZATIONS — DataFest 2026")
    print(SEP)

    # -- Preflight --
    print("\n[0] Preflight checks …")
    run_preflight_checks()

    # -- Load shared encounter data (used for plots 1, 3, 5) --
    print("\n[1/7] Loading encounter data …")
    enc = load_encounters()

    # -- Compute high-acuity follow-up (shared between plots 1 and 3) --
    print("\n  Computing high-acuity follow-up …")
    ha_events = compute_ha_followup(enc)

    # -- Plots --
    print("\n[2/7] Plot Q1: High-acuity follow-up …")
    plot_q1(ha_events)

    print("\n[3/7] Plot Q2/Q2b: Screened SDOH burden …")
    plot_q2()
    plot_q2b()

    print("\n[4/7] Plot Q3: MyChart follow-up …")
    plot_q3(ha_events)

    print("\n[5/7] Plot Q5: High-utilizer profile …")
    plot_q5(enc)
    del enc, ha_events  # free memory before bubble chart

    print("\n[6/7] Plot Q6: Diagnosis bubble chart …")
    plot_q6()

    print("\n[7/7] Plot Q7: ED-indexed SDOH follow-up …")
    plot_q7()

    write_notes()

    print(f"\n{SEP}")
    print("  DONE — outputs saved to:")
    print("    outputs/visualizations/  (7 PNGs)")
    print("    outputs/audit/           (support tables + notes)")
    print(SEP)


if __name__ == "__main__":
    main()
