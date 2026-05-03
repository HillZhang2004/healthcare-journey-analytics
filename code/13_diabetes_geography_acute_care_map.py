"""
13_diabetes_geography_acute_care_map.py

Map: Diabetes clean journeys with acute-care involvement by census block group.

Metric:
    pct_diabetes_acute_care_involved =
        diabetes clean journeys with hospital_involved_journey == True
        / total diabetes clean journeys * 100

Diabetes filter (union of two signals):
    - GroupName contains "diabet" (case-insensitive)
    - OR DiagnosisValue starts with "E10" or "E11"

Sample-size filter: n_diabetes_journeys >= 30 per census block group.

Visual style: teammate Q4 style — GeoJSON background if available, RdYlGn_r,
linear griddata contour, bubble scatter, city labels, colorbar.

Run from project root:
    python code/13_diabetes_geography_acute_care_map.py

Outputs:
    outputs/visualizations/q4_diabetes_acute_care_geography.png
    outputs/audit/q4_diabetes_acute_care_geography_table.csv
    outputs/audit/q4_diabetes_acute_care_geography_notes.txt
    outputs/audit/q4_diabetes_geo_preflight_checks.csv
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# ── Paths ─────────────────────────────────────────────────────────────────────

PARQUET   = Path("outputs/tables/journeys_with_core_sdoh_need_flags.parquet")
TIGER_CSV = Path("data/tigercensuscodes.csv")
VIZ_DIR   = Path("outputs/visualizations")
AUDIT_DIR = Path("outputs/audit")
CACHE_DIR = Path("outputs/cache")

for _d in (VIZ_DIR, AUDIT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_DIABETES_JOURNEYS = 30
INVALID_GEO = {"*Unspecified", "Unknown", "NA", "", "nan", "None"}
BG = "#F8F9FA"

CITIES = {
    "Topeka":      (-95.69, 39.05),
    "Wichita":     (-97.34, 37.69),
    "Kansas City": (-94.58, 39.10),
    "Lawrence":    (-95.24, 38.97),
    "Salina":      (-97.61, 38.84),
}

# ── GeoJSON helpers (Q4 style) ────────────────────────────────────────────────

def fetch_json(url: str, cache_name: str, timeout: int = 30) -> dict:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    data = urllib.request.urlopen(url, timeout=timeout).read()
    cache_path.write_bytes(data)
    return json.loads(data)


def ring_to_patch(ring: list) -> MplPolygon:
    return MplPolygon(np.array(ring), closed=True)


def add_geojson_layer(
    ax,
    features: list,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    patches = []
    for feat in features:
        geom = feat["geometry"]
        coords = geom["coordinates"]
        rings = coords if geom["type"] == "MultiPolygon" else [coords]
        for poly in rings:
            patches.append(ring_to_patch(poly[0]))
    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def draw_cities(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    for city, (lon, lat) in CITIES.items():
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            ax.plot(lon, lat, "k^", markersize=5, zorder=6)
            ax.annotate(
                city,
                (lon, lat),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
            )


def load_geojson_layers() -> tuple[list, list, list]:
    """Fetch Kansas county/state and neighbor-state GeoJSON; return empty lists on failure."""
    try:
        states = fetch_json(
            "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
            "us-states.json",
        )
        counties = fetch_json(
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            "us-counties.json",
        )
        NEIGHBOR_STATES = {"Kansas", "Missouri", "Nebraska", "Oklahoma", "Colorado"}
        KS_FIPS = "20"
        neighbor_feats = [
            f for f in states["features"]
            if f["properties"].get("name") in NEIGHBOR_STATES
        ]
        ks_county_feats = [
            f for f in counties["features"]
            if f["properties"].get("STATE") == KS_FIPS
        ]
        ks_state_feats = [
            f for f in states["features"]
            if f["properties"].get("name") == "Kansas"
        ]
        print(
            f"  GeoJSON loaded: {len(neighbor_feats)} neighbor states, "
            f"{len(ks_county_feats)} KS counties"
        )
        return neighbor_feats, ks_county_feats, ks_state_feats
    except Exception as exc:
        print(f"  GeoJSON unavailable ({exc}); map will show bubbles only.")
        return [], [], []

# ── Geography normalization ───────────────────────────────────────────────────

def normalize_geo_str(series: pd.Series) -> pd.Series:
    """Clean geo IDs to 12-char strings; invalid values become pd.NA."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.where(~s.isin(INVALID_GEO) & (s != "nan"), pd.NA)
    return s

# ── Tiger loader ──────────────────────────────────────────────────────────────

def load_tiger() -> pd.DataFrame:
    tiger = pd.read_csv(TIGER_CSV, keep_default_na=False, na_values=[], low_memory=False)
    tiger["geo_id"]     = tiger["GEOID"].astype(str).str.zfill(12)
    tiger["CENTLAT"]    = pd.to_numeric(tiger["CENTLAT"],      errors="coerce")
    tiger["CENTLON"]    = pd.to_numeric(tiger["CENTLON"],      errors="coerce")
    tiger["population"] = pd.to_numeric(tiger["PopulationValue"], errors="coerce")
    tiger = tiger.dropna(subset=["CENTLAT", "CENTLON"]).reset_index(drop=True)
    return tiger[["geo_id", "CENTLAT", "CENTLON", "population"]]

# ── Preflight checks ──────────────────────────────────────────────────────────

def run_preflight_checks(jdf: pd.DataFrame, tiger: pd.DataFrame) -> dict:
    """
    Verify diabetes filter, geography join, and sample-size availability.
    Saves q4_diabetes_geo_preflight_checks.csv.  Returns dict of key facts.
    """
    total_journeys = len(jdf)

    mask_dv = jdf["DiagnosisValue"].str.startswith(("E10", "E11"), na=False)
    mask_gn = jdf["GroupName"].str.contains("diabet", case=False, na=False)
    diab_mask = mask_dv | mask_gn
    n_diab_e10e11_only  = int(mask_dv.sum())
    n_diab_groupname    = int(mask_gn.sum())
    n_diab_union        = int(diab_mask.sum())
    n_overlap           = int((mask_dv & mask_gn).sum())

    diab = jdf[diab_mask].copy()
    assert "hospital_involved_journey" in diab.columns, \
        "hospital_involved_journey column missing from parquet"

    n_diab_acute = int(diab["hospital_involved_journey"].fillna(False).astype(bool).sum())

    diab["geo_id"] = normalize_geo_str(diab["CensusBlockGroupFipsCode"])
    diab_geo = diab.dropna(subset=["geo_id"])
    n_diab_valid_geo     = len(diab_geo)
    n_diab_unique_geo    = diab_geo["geo_id"].nunique()

    tiger_geo_set = set(tiger["geo_id"])
    n_diab_matching_tiger = int(diab_geo["geo_id"].isin(tiger_geo_set).sum())

    agg = diab_geo.groupby("geo_id").agg(
        n_diabetes_journeys=("PatientDurableKey", "count"),
    ).reset_index()
    n_blocks_before = len(agg)
    n_blocks_after  = int((agg["n_diabetes_journeys"] >= MIN_DIABETES_JOURNEYS).sum())

    rows = [
        ("total_journeys_in_parquet",              total_journeys),
        ("diabetes_journeys_E10_E11_only",          n_diab_e10e11_only),
        ("diabetes_journeys_GroupName_diabet",      n_diab_groupname),
        ("diabetes_journeys_overlap_both_signals",  n_overlap),
        ("diabetes_journeys_union",                 n_diab_union),
        ("diabetes_journeys_with_acute_care",       n_diab_acute),
        ("diabetes_journeys_with_valid_geo",        n_diab_valid_geo),
        ("diabetes_journeys_matching_tiger",        n_diab_matching_tiger),
        ("unique_geo_ids_in_diabetes_journeys",     n_diab_unique_geo),
        ("census_blocks_before_minjourney_filter",  n_blocks_before),
        ("census_blocks_after_minjourney_filter",   n_blocks_after),
        ("min_diabetes_journeys_threshold",         MIN_DIABETES_JOURNEYS),
        ("hospital_involved_journey_col_present",   1),
    ]

    pd.DataFrame(rows, columns=["check", "value"]).to_csv(
        AUDIT_DIR / "q4_diabetes_geo_preflight_checks.csv", index=False
    )
    print("[Preflight] Checks:")
    for k, v in rows:
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    return dict(rows)

# ── Build diabetes aggregate table ────────────────────────────────────────────

def build_diabetes_geo_table(jdf: pd.DataFrame, tiger: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate diabetes journeys by census block group.

    Returns a DataFrame with columns:
        geo_id, CENTLAT, CENTLON, population,
        n_diabetes_journeys, n_diabetes_patients,
        n_diabetes_acute_care_journeys, pct_diabetes_acute_care_involved

    Filtered to n_diabetes_journeys >= MIN_DIABETES_JOURNEYS and
    matched to tiger for valid coordinates.
    """
    mask = (jdf["DiagnosisValue"].str.startswith(("E10", "E11"), na=False)) | \
           (jdf["GroupName"].str.contains("diabet", case=False, na=False))
    diab = jdf[mask].copy()

    diab["geo_id"] = normalize_geo_str(diab["CensusBlockGroupFipsCode"])
    diab = diab.dropna(subset=["geo_id"])
    diab["hospital_involved_journey"] = (
        diab["hospital_involved_journey"].fillna(False).astype(bool)
    )

    geo_agg = diab.groupby("geo_id").agg(
        n_diabetes_journeys           =("PatientDurableKey", "count"),
        n_diabetes_patients           =("PatientDurableKey", "nunique"),
        n_diabetes_acute_care_journeys=("hospital_involved_journey", "sum"),
    ).reset_index()

    geo_agg["pct_diabetes_acute_care_involved"] = (
        geo_agg["n_diabetes_acute_care_journeys"]
        / geo_agg["n_diabetes_journeys"] * 100
    )

    # Apply sample-size filter before joining coordinates
    geo_agg = geo_agg[geo_agg["n_diabetes_journeys"] >= MIN_DIABETES_JOURNEYS].copy()

    # Join tiger coordinates
    geo_agg = geo_agg.merge(tiger, on="geo_id", how="inner")

    print(f"[Aggregate table] {len(geo_agg):,} census block groups "
          f"(n_diabetes_journeys >= {MIN_DIABETES_JOURNEYS})")
    return geo_agg

# ── Plot (Q4 visual style) ────────────────────────────────────────────────────

def plot_diabetes_map(
    geo_plot: pd.DataFrame,
    neighbor_feats: list,
    ks_county_feats: list,
    ks_state_feats: list,
) -> None:
    """
    Q4-style bubble map for diabetes acute-care involvement.

    Bubble size  = n_diabetes_journeys (clipped 15–350)
    Bubble color = pct_diabetes_acute_care_involved
    Contour      = light linear griddata interpolation (exploratory visual only)
    """
    metric_col = "pct_diabetes_acute_care_involved"
    size_col   = "n_diabetes_journeys"

    lon_min = geo_plot["CENTLON"].min() - 0.4
    lon_max = geo_plot["CENTLON"].max() + 0.4
    lat_min = geo_plot["CENTLAT"].min() - 0.3
    lat_max = geo_plot["CENTLAT"].max() + 0.3

    # Linear contour grid (exploratory guide only; bubble table is source of truth)
    grid_lon = np.linspace(lon_min, lon_max, 300)
    grid_lat = np.linspace(lat_min, lat_max, 300)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    pts    = geo_plot[["CENTLON", "CENTLAT"]].values
    vals   = geo_plot[metric_col].values
    grid_z = griddata(pts, vals, (grid_x, grid_y), method="linear")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.set_facecolor("#D6EAF8")

    # GeoJSON map background
    if neighbor_feats:
        add_geojson_layer(ax, neighbor_feats,
                          facecolor="#EEF0E5", edgecolor="#999999", linewidth=0.8)
    if ks_county_feats:
        add_geojson_layer(ax, ks_county_feats,
                          facecolor="#E8EDE0", edgecolor="#AAAAAA", linewidth=0.4)
    if ks_state_feats:
        add_geojson_layer(ax, ks_state_feats,
                          facecolor="none",    edgecolor="#444444", linewidth=1.5)

    # Linear contour fill + grey contour lines
    ax.contourf(grid_x, grid_y, grid_z, levels=8,
                cmap="RdYlGn_r", alpha=0.30, zorder=2)
    cs = ax.contour(grid_x, grid_y, grid_z, levels=8,
                    colors="dimgrey", linewidths=0.6, alpha=0.7, zorder=3)
    ax.clabel(cs, fmt="%.1f", fontsize=6.5, inline=True, inline_spacing=4)

    # Bubble scatter — color scale capped at 0–20% for slide readability;
    # values above 20% saturate at the top color; no rows are dropped.
    sc = ax.scatter(
        geo_plot["CENTLON"],
        geo_plot["CENTLAT"],
        c=geo_plot[metric_col],
        s=np.clip(geo_plot[size_col] / 4, 15, 350),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=20,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(
        "% diabetes journeys with acute-care involvement",
        fontsize=9,
    )

    # Bubble size legend
    for n_journeys, label in [
        (30,  "30 journeys"),
        (100, "100 journeys"),
        (268, "268 journeys (max)"),
    ]:
        ax.scatter([], [], s=np.clip(n_journeys / 4, 15, 350),
                   c="grey", alpha=0.6, edgecolors="white", label=label)

    ax.legend(
        title="Diabetes journey count",
        frameon=True, fontsize=8, title_fontsize=8,
        loc="lower left", framealpha=0.8,
    )

    draw_cities(ax, lon_min, lon_max, lat_min, lat_max)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Q4 — Diabetes Journeys with Acute-Care Involvement by Census Block Group\n"
        "(color = % with acute-care involvement  |  "
        "bubble size = diabetes journey count  |  county lines for Kansas)",
        fontweight="bold",
        pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)

    out = VIZ_DIR / "q4_diabetes_acute_care_geography.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Save audit table ──────────────────────────────────────────────────────────

def save_audit_table(geo_plot: pd.DataFrame) -> None:
    """
    Save block-group aggregate table. Excludes patient, encounter, and
    diagnosis identifiers — only geo_id and aggregate metrics are written.
    """
    safe_cols = [
        "geo_id",
        "CENTLAT",
        "CENTLON",
        "population",
        "n_diabetes_journeys",
        "n_diabetes_patients",
        "n_diabetes_acute_care_journeys",
        "pct_diabetes_acute_care_involved",
    ]
    out = AUDIT_DIR / "q4_diabetes_acute_care_geography_table.csv"
    geo_plot[safe_cols].to_csv(out, index=False)
    print(f"  Saved: {out}")
    # Privacy check: no patient/encounter/diagnosis ID columns
    forbidden = {"PatientDurableKey", "EncounterKey", "DiagnosisKey",
                 "DiagnosisValue", "DiagnosisName"}
    leaked = forbidden & set(geo_plot[safe_cols].columns)
    if leaked:
        raise RuntimeError(f"Privacy violation: identifier columns found in output: {leaked}")

# ── Notes ─────────────────────────────────────────────────────────────────────

def write_notes(pf: dict, geo_plot: pd.DataFrame) -> None:
    metric_q = geo_plot["pct_diabetes_acute_care_involved"].quantile([0.25, 0.5, 0.75])
    lines = [
        "Q4 Diabetes Acute-Care Geography — Analysis Notes",
        "=" * 62,
        "",
        "This map is diabetes-specific. It describes the geographic distribution",
        "of acute-care involvement among patients with clean diabetes diagnosis journeys.",
        "",
        "Metric definition",
        "  pct_diabetes_acute_care_involved =",
        "    (diabetes clean journeys with hospital_involved_journey == True)",
        "    / (total diabetes clean journeys) * 100",
        "",
        "  This is a journey-level metric, NOT a patient-level raw encounter share.",
        "  Acute-care involvement = hospital_involved_journey == True.",
        "  hospital_involved_journey is a combined signal: a journey is flagged True",
        "  when it contains at least one hospital admission, ED visit, OR observation",
        "  stay. It is NOT limited to pure hospital admissions.",
        "",
        "Diabetes journey definition",
        "  A clean diagnosis-anchored journey (PatientDurableKey + DiagnosisValue)",
        "  is classified as a diabetes journey when:",
        "    - GroupName contains 'diabet' (case-insensitive), OR",
        "    - DiagnosisValue starts with 'E10' or 'E11'",
        "  Both signals are applied as a union (OR). The GroupName signal is a",
        "  superset of E10/E11 codes; it also captures gestational diabetes (O24.x),",
        "  other specified diabetes (E13.x), and secondary diabetes (E08.x, E09.x).",
        "",
        "Diagnosis filter counts",
        f"  Diabetes journeys (E10/E11 DiagnosisValue):    {pf.get('diabetes_journeys_E10_E11_only', 'N/A'):>8,}",
        f"  Diabetes journeys (GroupName 'diabet'):         {pf.get('diabetes_journeys_GroupName_diabet', 'N/A'):>8,}",
        f"  Diabetes journeys (union):                      {pf.get('diabetes_journeys_union', 'N/A'):>8,}",
        f"  Diabetes journeys with acute-care involvement:  {pf.get('diabetes_journeys_with_acute_care', 'N/A'):>8,}",
        "",
        "Geography",
        f"  Diabetes journeys with valid geo ID:    {pf.get('diabetes_journeys_with_valid_geo', 'N/A'):>8,}",
        f"  Matching tiger census GEOID:            {pf.get('diabetes_journeys_matching_tiger', 'N/A'):>8,}",
        f"  Census blocks before sample filter:     {pf.get('census_blocks_before_minjourney_filter', 'N/A'):>8,}",
        f"  Census blocks after n>=30 filter:       {pf.get('census_blocks_after_minjourney_filter', 'N/A'):>8,}",
        f"  Census blocks plotted (after coordinate join): {len(geo_plot):>5,}",
        "  Note: the plotted count may be lower than the preflight 'after n>=30' count",
        "  because the final inner join to tiger requires a matching GEOID with valid",
        "  CENTLAT/CENTLON coordinates. Block groups with no matching tiger row are",
        "  excluded from the map but are still counted in the preflight.",
        "",
        f"  Geography = CensusBlockGroupFipsCode joined to GEOID in tigercensuscodes.csv.",
        "  Invalid geo values excluded: *Unspecified, Unknown, NA, blank, nan, None.",
        f"  Minimum diabetes journeys per block group: {MIN_DIABETES_JOURNEYS}",
        "",
        "Metric distribution (plotted block groups)",
        f"  25th pct:  {metric_q[0.25]:.1f}%",
        f"  Median:    {metric_q[0.50]:.1f}%",
        f"  75th pct:  {metric_q[0.75]:.1f}%",
        "",
        "Map visual encoding",
        "  Bubble size  = n_diabetes_journeys (clipped to scatter size 15–350)",
        "  Bubble color = pct_diabetes_acute_care_involved (RdYlGn_r palette)",
        "  Contour fill = light linear griddata interpolation (alpha 0.30)",
        "  Contour lines are exploratory visual guides only.",
        "  The bubble-level aggregate table is the authoritative source of truth.",
        "",
        "Interpretation caution",
        "  Results are descriptive. Geographic patterns are observational and may",
        "  reflect population age structure, comorbidity burden, proximity to",
        "  acute-care facilities, insurance coverage, or other unmeasured factors.",
        "  Language throughout uses 'associated with', not 'causes' or 'leads to'.",
    ]
    out = AUDIT_DIR / "q4_diabetes_acute_care_geography_notes.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading tiger census codes...")
    tiger = load_tiger()
    print(f"  {len(tiger):,} rows with valid coordinates")

    print("\nLoading journey parquet (diabetes-relevant columns)...")
    jdf = pd.read_parquet(
        PARQUET,
        columns=[
            "PatientDurableKey",
            "DiagnosisValue",
            "GroupName",
            "hospital_involved_journey",
            "CensusBlockGroupFipsCode",
        ],
    )
    print(f"  {len(jdf):,} total journeys")

    print("\nRunning preflight checks...")
    pf = run_preflight_checks(jdf, tiger)

    print("\nBuilding diabetes geographic aggregate table...")
    geo_plot = build_diabetes_geo_table(jdf, tiger)

    print("\nFetching GeoJSON map layers...")
    neighbor_feats, ks_county_feats, ks_state_feats = load_geojson_layers()

    print("\nPlotting diabetes acute-care geography map...")
    plot_diabetes_map(geo_plot, neighbor_feats, ks_county_feats, ks_state_feats)

    print("\nSaving audit table...")
    save_audit_table(geo_plot)

    print("\nWriting analysis notes...")
    write_notes(pf, geo_plot)

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Total diabetes journeys (union filter):  {pf['diabetes_journeys_union']:>8,}")
    print(f"  With acute-care involvement:             {pf['diabetes_journeys_with_acute_care']:>8,}")
    print(f"  With valid geo ID:                       {pf['diabetes_journeys_with_valid_geo']:>8,}")
    print(f"  Census block groups plotted:             {len(geo_plot):>8,}")
    q = geo_plot["pct_diabetes_acute_care_involved"].quantile([0.25, 0.5, 0.75])
    print(
        f"  Metric: median={q[0.5]:.1f}%, "
        f"IQR {q[0.25]:.1f}–{q[0.75]:.1f}%"
    )
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
