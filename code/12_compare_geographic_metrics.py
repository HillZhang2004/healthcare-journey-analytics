"""
12_compare_geographic_metrics.py

Three comparable geographic bubble maps on the same canvas:
  Metric 1 (teammate-style): average patient-level hospital admission share
                              per census block group
  Metric 2 (pipeline journey): pct of clean journeys with hospital/ED/obs involvement
  Metric 3 (ED-indexed):       pct of clean journeys with at least one ED visit

Run from project root:
    python code/12_compare_geographic_metrics.py

Outputs (audit/):
    q4_geo_metric_preflight_checks.csv
    q4a_geo_patient_admission_share_table.csv
    q4b_geo_journey_acute_involvement_table.csv
    q4c_geo_ed_indexed_journey_share_table.csv
    q4_geo_metric_comparison_table.csv
    q4_geo_metric_comparison_notes.txt

Outputs (visualizations/):
    q4a_geo_patient_admission_share.png
    q4b_geo_journey_acute_involvement.png
    q4c_geo_ed_indexed_journey_share.png
    q4_geo_three_metric_comparison.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PARQUET   = "outputs/tables/journeys_with_core_sdoh_need_flags.parquet"
ENC_CSV   = "data/encounters.csv"
PAT_CSV   = "data/patients.csv"
TIGER_CSV = "data/tigercensuscodes.csv"
AUDIT_DIR = "outputs/audit"
VIZ_DIR   = "outputs/visualizations"

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_PATIENTS = 30

# Values to treat as missing geographic identifiers
INVALID_GEO = {"*Unspecified", "Unknown", "NA", "", "nan", "None"}

# Kansas service area bounds: (lon_min, lon_max, lat_min, lat_max)
MAP_EXTENT = (-101.9, -94.6, 37.0, 40.0)

# Reference cities for map orientation
CITIES = {
    "Topeka":      (-95.69, 39.05),
    "Wichita":     (-97.34, 37.69),
    "Kansas City": (-94.63, 39.10),
}

# ── Geography helpers ─────────────────────────────────────────────────────────

def normalize_geo_str(series: pd.Series) -> pd.Series:
    """Return a Series of clean 12-char geo-ID strings; invalid values become pd.NA."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.where(~s.isin(INVALID_GEO) & (s != "nan"), pd.NA)
    return s


def load_tiger() -> pd.DataFrame:
    """Load tiger census codes; keep rows with valid numeric lat/lon."""
    tiger = pd.read_csv(TIGER_CSV, keep_default_na=False, na_values=[], low_memory=False)
    tiger["geo_id"]    = tiger["GEOID"].astype(str).str.zfill(12)
    tiger["lat"]       = pd.to_numeric(tiger["CENTLAT"],      errors="coerce")
    tiger["lon"]       = pd.to_numeric(tiger["CENTLON"],      errors="coerce")
    tiger["population"]= pd.to_numeric(tiger["PopulationValue"], errors="coerce")
    tiger = tiger.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return tiger[["geo_id", "lat", "lon", "population"]]


def load_patient_geo() -> pd.DataFrame:
    """Return unique patients with normalized census block group IDs."""
    pat = pd.read_csv(
        PAT_CSV,
        keep_default_na=False, na_values=[], low_memory=False,
        usecols=["DurableKey", "CensusBlockGroupFipsCode"],
    )
    pat["PatientDurableKey"] = (
        pd.to_numeric(pat["DurableKey"], errors="coerce")
        .round()
        .astype("Int64")
    )
    pat["geo_id"] = normalize_geo_str(pat["CensusBlockGroupFipsCode"])
    pat = pat.dropna(subset=["PatientDurableKey", "geo_id"])
    return pat[["PatientDurableKey", "geo_id"]].drop_duplicates("PatientDurableKey")

# ── Preflight ─────────────────────────────────────────────────────────────────

def run_preflight_checks(
    jdf: pd.DataFrame,
    tiger: pd.DataFrame,
    pat_geo: pd.DataFrame,
) -> dict:
    """Log data-coverage facts; save to audit CSV; return dict."""
    jdf_geo_norm = normalize_geo_str(jdf["CensusBlockGroupFipsCode"])
    valid_journeys   = int(jdf_geo_norm.notna().sum())
    tiger_geo_set    = set(tiger["geo_id"])
    matched_journeys = int(jdf_geo_norm.isin(tiger_geo_set).sum())

    rows = [
        ("total_journeys_in_parquet",        jdf.shape[0]),
        ("journeys_with_valid_geo_id",        valid_journeys),
        ("journeys_matching_tiger",           matched_journeys),
        ("tiger_rows_with_valid_coords",      tiger.shape[0]),
        ("unique_geo_ids_in_tiger",           tiger["geo_id"].nunique()),
        ("unique_geo_ids_in_parquet",         jdf_geo_norm.dropna().nunique()),
        ("patient_geo_rows_valid",            pat_geo.shape[0]),
        ("unique_geo_ids_in_patients",        pat_geo["geo_id"].nunique()),
        ("min_patients_threshold",            MIN_PATIENTS),
    ]

    pd.DataFrame(rows, columns=["check", "value"]).to_csv(
        f"{AUDIT_DIR}/q4_geo_metric_preflight_checks.csv", index=False
    )
    print("[Preflight] Coverage checks:")
    for k, v in rows:
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    return dict(rows)

# ── Metric 1: patient-level hospital admission share ──────────────────────────

def compute_metric1(pat_geo: pd.DataFrame, tiger: pd.DataFrame) -> pd.DataFrame:
    """
    Per patient: admission_share = n_hospital_admissions / n_total_encounters.
    Per geo block group: mean of patient-level shares.
    Filter: n_patients >= MIN_PATIENTS.
    """
    enc = pd.read_csv(
        ENC_CSV,
        keep_default_na=False, na_values=[], low_memory=False,
        usecols=["PatientDurableKey", "IsHospitalAdmission"],
    )
    enc["PatientDurableKey"] = (
        pd.to_numeric(enc["PatientDurableKey"], errors="coerce")
        .round()
        .astype("Int64")
    )
    enc["IsHospitalAdmission"] = (
        pd.to_numeric(enc["IsHospitalAdmission"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    enc = enc.dropna(subset=["PatientDurableKey"])

    pat_stats = enc.groupby("PatientDurableKey").agg(
        n_encounters  =("IsHospitalAdmission", "count"),
        n_admissions  =("IsHospitalAdmission", "sum"),
    ).reset_index()
    pat_stats["admission_share"] = (
        pat_stats["n_admissions"]
        / pat_stats["n_encounters"].replace(0, np.nan)
    )

    merged = pat_stats.merge(pat_geo, on="PatientDurableKey", how="inner")

    geo_agg = merged.groupby("geo_id").agg(
        n_patients          =("PatientDurableKey", "nunique"),
        mean_admission_share=("admission_share",    "mean"),
        total_admissions    =("n_admissions",        "sum"),
        total_encounters    =("n_encounters",        "sum"),
    ).reset_index()

    geo_agg = geo_agg[geo_agg["n_patients"] >= MIN_PATIENTS].copy()
    geo_agg = geo_agg.merge(tiger, on="geo_id", how="inner")
    geo_agg["metric_value"] = geo_agg["mean_admission_share"] * 100
    geo_agg["metric_label"] = "Mean patient hospital admission share (%)"

    geo_agg.to_csv(f"{AUDIT_DIR}/q4a_geo_patient_admission_share_table.csv", index=False)
    print(f"[Metric 1] {len(geo_agg):,} geo blocks (n_patients >= {MIN_PATIENTS})")
    return geo_agg

# ── Metric 2: journey acute-care involvement rate ─────────────────────────────

def compute_metric2(jdf: pd.DataFrame, tiger: pd.DataFrame) -> pd.DataFrame:
    """
    Per geo block group: pct of clean journeys with hospital_involved_journey == True.
    Filter: n_patients >= MIN_PATIENTS.
    """
    sub = jdf[["PatientDurableKey", "CensusBlockGroupFipsCode", "hospital_involved_journey"]].copy()
    sub["geo_id"] = normalize_geo_str(sub["CensusBlockGroupFipsCode"])
    sub = sub.dropna(subset=["geo_id"])
    sub["hospital_involved_journey"] = sub["hospital_involved_journey"].fillna(False).astype(bool)

    geo_agg = sub.groupby("geo_id").agg(
        n_journeys         =("PatientDurableKey",          "count"),
        n_patients         =("PatientDurableKey",          "nunique"),
        n_hospital_involved=("hospital_involved_journey",  "sum"),
    ).reset_index()
    geo_agg["pct_hospital_involved"] = (
        geo_agg["n_hospital_involved"] / geo_agg["n_journeys"] * 100
    )

    geo_agg = geo_agg[geo_agg["n_patients"] >= MIN_PATIENTS].copy()
    geo_agg = geo_agg.merge(tiger, on="geo_id", how="inner")
    geo_agg["metric_value"] = geo_agg["pct_hospital_involved"]
    geo_agg["metric_label"] = "Journeys with acute-care involvement (%)"

    geo_agg.to_csv(f"{AUDIT_DIR}/q4b_geo_journey_acute_involvement_table.csv", index=False)
    print(f"[Metric 2] {len(geo_agg):,} geo blocks (n_patients >= {MIN_PATIENTS})")
    return geo_agg

# ── Metric 3: ED-indexed journey share ────────────────────────────────────────

def compute_metric3(jdf: pd.DataFrame, tiger: pd.DataFrame) -> pd.DataFrame:
    """
    Per geo block group: pct of clean journeys with has_ed == True.
    Filter: n_patients >= MIN_PATIENTS.
    """
    sub = jdf[["PatientDurableKey", "CensusBlockGroupFipsCode", "has_ed"]].copy()
    sub["geo_id"] = normalize_geo_str(sub["CensusBlockGroupFipsCode"])
    sub = sub.dropna(subset=["geo_id"])
    sub["has_ed"] = sub["has_ed"].fillna(False).astype(bool)

    geo_agg = sub.groupby("geo_id").agg(
        n_journeys  =("PatientDurableKey", "count"),
        n_patients  =("PatientDurableKey", "nunique"),
        n_ed_journeys=("has_ed",           "sum"),
    ).reset_index()
    geo_agg["pct_ed_indexed"] = (
        geo_agg["n_ed_journeys"] / geo_agg["n_journeys"] * 100
    )

    geo_agg = geo_agg[geo_agg["n_patients"] >= MIN_PATIENTS].copy()
    geo_agg = geo_agg.merge(tiger, on="geo_id", how="inner")
    geo_agg["metric_value"] = geo_agg["pct_ed_indexed"]
    geo_agg["metric_label"] = "ED-indexed journeys (%)"

    geo_agg.to_csv(f"{AUDIT_DIR}/q4c_geo_ed_indexed_journey_share_table.csv", index=False)
    print(f"[Metric 3] {len(geo_agg):,} geo blocks (n_patients >= {MIN_PATIENTS})")
    return geo_agg

# ── Shared bubble map ─────────────────────────────────────────────────────────

def plot_bubble_map(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    cbar_label: str,
    map_extent: tuple,
    output_path: str | None = None,
    ax=None,
) -> None:
    """
    Geographic bubble map. Bubble size = sqrt(n_patients) * 3; color = metric_value.
    If ax is provided, draws into that axes (combined plot mode) and does not save.
    If ax is None, creates a standalone figure and saves to output_path.
    """
    if df.empty:
        print(f"  [Warning] No data to plot — skipping: {title}")
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    lon_min, lon_max, lat_min, lat_max = map_extent

    ax.set_facecolor("#f0f4f8")
    ax.add_patch(plt.Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=1.5, edgecolor="#aaa", facecolor="#dce8f5", zorder=0,
    ))

    sizes = np.sqrt(df["n_patients"].values.astype(float)) * 3
    vmin  = df[metric_col].quantile(0.05)
    vmax  = df[metric_col].quantile(0.95)
    if vmin == vmax:
        vmin, vmax = df[metric_col].min(), df[metric_col].max()

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df[metric_col], cmap="YlOrRd",
        s=sizes, alpha=0.78,
        linewidths=0.4, edgecolors="#555",
        vmin=vmin, vmax=vmax,
        zorder=2,
    )

    for city, (cx, cy) in CITIES.items():
        if lon_min <= cx <= lon_max and lat_min <= cy <= lat_max:
            ax.plot(cx, cy, "k^", ms=5, zorder=3)
            ax.text(cx + 0.05, cy + 0.06, city, fontsize=7, color="#222", zorder=4)

    plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.03, pad=0.02)

    ref_sizes = [30, 200, 1000]
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#888", markeredgecolor="#555",
            markersize=np.sqrt(np.sqrt(s) * 3),
            label=f"n={s:,}",
        )
        for s in ref_sizes
    ]
    ax.legend(
        handles=legend_handles, title="Patients", loc="lower left",
        fontsize=7, title_fontsize=7, framealpha=0.85,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.set_title(title, fontsize=10, pad=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.45)

    if standalone:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {output_path}")
        plt.close()

# ── Individual plot wrappers ───────────────────────────────────────────────────

def plot_q4a(df1: pd.DataFrame) -> None:
    plot_bubble_map(
        df1, metric_col="metric_value",
        title="Q4a — Mean Patient Hospital Admission Share\nby Census Block Group",
        cbar_label="Mean admission share (%)",
        map_extent=MAP_EXTENT,
        output_path=f"{VIZ_DIR}/q4a_geo_patient_admission_share.png",
    )


def plot_q4b(df2: pd.DataFrame) -> None:
    plot_bubble_map(
        df2, metric_col="metric_value",
        title="Q4b — Journeys with Acute-Care Involvement\nby Census Block Group",
        cbar_label="% journeys with hospital/ED/obs",
        map_extent=MAP_EXTENT,
        output_path=f"{VIZ_DIR}/q4b_geo_journey_acute_involvement.png",
    )


def plot_q4c(df3: pd.DataFrame) -> None:
    plot_bubble_map(
        df3, metric_col="metric_value",
        title="Q4c — ED-Indexed Journey Share\nby Census Block Group",
        cbar_label="% journeys with ED visit",
        map_extent=MAP_EXTENT,
        output_path=f"{VIZ_DIR}/q4c_geo_ed_indexed_journey_share.png",
    )

# ── Combined 3-panel ──────────────────────────────────────────────────────────

def plot_combined(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Geographic Comparison of Three Healthcare Utilization Metrics\n"
        "(bubble size = √n patients per block group; color = metric value)",
        fontsize=12, y=1.01,
    )

    specs = [
        (df1, "Q4a: Patient Admission\nShare",         "Mean admission share (%)"),
        (df2, "Q4b: Journey Acute-Care\nInvolvement",  "% journeys w/ hosp/ED/obs"),
        (df3, "Q4c: ED-Indexed\nJourney Share",        "% journeys w/ ED visit"),
    ]
    for ax, (df, title, cbar) in zip(axes, specs):
        plot_bubble_map(
            df, metric_col="metric_value",
            title=title, cbar_label=cbar,
            map_extent=MAP_EXTENT, ax=ax,
        )

    plt.tight_layout()
    out = f"{VIZ_DIR}/q4_geo_three_metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    tiger: pd.DataFrame,
) -> None:
    t1 = df1[["geo_id", "n_patients", "metric_value"]].rename(columns={
        "metric_value": "m1_mean_admission_share_pct",
        "n_patients":   "m1_n_patients",
    })
    t2 = df2[["geo_id", "n_patients", "n_journeys", "metric_value"]].rename(columns={
        "metric_value": "m2_pct_hospital_involved",
        "n_patients":   "m2_n_patients",
        "n_journeys":   "m2_n_journeys",
    })
    t3 = df3[["geo_id", "n_patients", "n_journeys", "metric_value"]].rename(columns={
        "metric_value": "m3_pct_ed_indexed",
        "n_patients":   "m3_n_patients",
        "n_journeys":   "m3_n_journeys",
    })

    comp = (
        t1.merge(t2, on="geo_id", how="outer")
          .merge(t3, on="geo_id", how="outer")
          .merge(tiger[["geo_id", "lat", "lon", "population"]], on="geo_id", how="left")
    )

    out = f"{AUDIT_DIR}/q4_geo_metric_comparison_table.csv"
    comp.to_csv(out, index=False)
    print(f"[Comparison table] {len(comp):,} geo blocks; saved to {out}")

# ── Notes ─────────────────────────────────────────────────────────────────────

def write_notes(pf: dict) -> None:
    lines = [
        "Q4 Geographic Metric Comparison — Analysis Notes",
        "=" * 62,
        "",
        "Three geographic bubble maps compare healthcare utilization patterns",
        "across census block groups in the Stormont Vail Health service area.",
        "",
        "Metric 1 — Mean Patient Hospital Admission Share (Q4a)",
        "  Source: encounters.csv (PatientDurableKey, IsHospitalAdmission),",
        "          patients.csv (CensusBlockGroupFipsCode)",
        "  Definition: For each patient, compute the proportion of their encounters",
        "  that are hospital admissions. Average across patients within each census",
        "  block group. This mirrors a teammate's analysis approach.",
        f"  Patients with valid geo IDs: {pf.get('patient_geo_rows_valid', 'N/A'):,}",
        "",
        "Metric 2 — Journey Acute-Care Involvement Rate (Q4b)",
        "  Source: journeys_with_core_sdoh_need_flags.parquet",
        "          Column: hospital_involved_journey",
        "  Definition: Proportion of clean diagnosis-anchored journeys per census",
        "  block group where the journey includes at least one hospital admission,",
        "  ED visit, or observation stay (hospital_involved_journey == True).",
        "",
        "Metric 3 — ED-Indexed Journey Share (Q4c)",
        "  Source: journeys_with_core_sdoh_need_flags.parquet",
        "          Column: has_ed",
        "  Definition: Proportion of clean journeys per census block group that",
        "  include at least one ED visit (has_ed == True).",
        "",
        "Common filters applied to all three metrics:",
        f"  - Minimum {MIN_PATIENTS} patients per census block group (suppresses sparse cells)",
        "  - Geo IDs must match tiger census codes (requires valid lat/lon)",
        "  - Excluded geo values: *Unspecified, Unknown, NA, blank, nan, None",
        "",
        "Geographic coverage summary:",
        f"  - Total journeys in parquet:          {pf.get('total_journeys_in_parquet', 'N/A'):,}",
        f"  - Journeys with valid geo ID:         {pf.get('journeys_with_valid_geo_id', 'N/A'):,}",
        f"  - Journeys matching tiger GEOID:      {pf.get('journeys_matching_tiger', 'N/A'):,}",
        f"  - Tiger rows with valid coordinates:  {pf.get('tiger_rows_with_valid_coords', 'N/A'):,}",
        f"  - Unique geo IDs in tiger:            {pf.get('unique_geo_ids_in_tiger', 'N/A'):,}",
        f"  - Unique geo IDs in parquet journeys: {pf.get('unique_geo_ids_in_parquet', 'N/A'):,}",
        f"  - Unique geo IDs in patients file:    {pf.get('unique_geo_ids_in_patients', 'N/A'):,}",
        "",
        "Visual encoding:",
        "  - Bubble size = sqrt(n_patients) to avoid visual dominance of large groups.",
        "  - Color scale spans 5th to 95th percentile per map to reduce outlier influence.",
        "  - Map background is a plain coordinate rectangle (geopandas not installed).",
        "  - Service area bounds: lon -101.9 to -94.6, lat 37.0 to 40.0 (Kansas).",
        "",
        "Interpretation caution:",
        "  Geographic patterns are descriptive and observational. Associations between",
        "  block group location and utilization may reflect population demographics,",
        "  proximity to facilities, insurance coverage, or other unmeasured factors.",
        "  Language throughout uses 'associated with' and 'observed among', not 'causes'.",
    ]

    out = f"{AUDIT_DIR}/q4_geo_metric_comparison_notes.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[Notes] Saved to {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR,   exist_ok=True)

    print("Loading tiger census codes...")
    tiger = load_tiger()
    print(f"  {len(tiger):,} rows with valid coordinates")

    print("Loading patient geo IDs...")
    pat_geo = load_patient_geo()
    print(f"  {len(pat_geo):,} patients with valid geo IDs")

    print("Loading journey parquet (4 columns)...")
    jdf = pd.read_parquet(
        PARQUET,
        columns=["PatientDurableKey", "CensusBlockGroupFipsCode",
                 "hospital_involved_journey", "has_ed"],
    )
    print(f"  {len(jdf):,} journeys loaded")

    print("\nRunning preflight checks...")
    pf = run_preflight_checks(jdf, tiger, pat_geo)

    print("\nComputing Metric 1 (patient-level admission share)...")
    df1 = compute_metric1(pat_geo, tiger)

    print("\nComputing Metric 2 (journey acute-care involvement)...")
    df2 = compute_metric2(jdf, tiger)

    print("\nComputing Metric 3 (ED-indexed journey share)...")
    df3 = compute_metric3(jdf, tiger)

    print("\nPlotting individual maps...")
    plot_q4a(df1)
    plot_q4b(df2)
    plot_q4c(df3)

    print("\nPlotting combined 3-panel comparison...")
    plot_combined(df1, df2, df3)

    print("\nBuilding comparison table...")
    build_comparison_table(df1, df2, df3, tiger)

    print("\nWriting analysis notes...")
    write_notes(pf)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Metric 1 geo blocks plotted:        {len(df1):>6,}")
    print(f"  Metric 2 geo blocks plotted:        {len(df2):>6,}")
    print(f"  Metric 3 geo blocks plotted:        {len(df3):>6,}")
    common = set(df1["geo_id"]) & set(df2["geo_id"]) & set(df3["geo_id"])
    print(f"  Geo blocks common to all 3 metrics: {len(common):>6,}")
    for label, df, col in [
        ("M1 mean admission share (%)",    df1, "metric_value"),
        ("M2 pct hospital involved (%)",   df2, "metric_value"),
        ("M3 pct ED-indexed (%)",          df3, "metric_value"),
    ]:
        q = df[col].quantile([0.25, 0.5, 0.75])
        print(
            f"  {label}: "
            f"median={q[0.5]:.1f}%, IQR {q[0.25]:.1f}–{q[0.75]:.1f}%"
        )
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
