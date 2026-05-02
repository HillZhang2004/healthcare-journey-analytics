# ASA DataFest 2026 — Stormont Vail Health

## Project overview

This project analyzes anonymized healthcare encounter data from Stormont Vail Health
as part of ASA DataFest 2026. The central research question is:

> **How do patients move through the healthcare system over time, and how does
> journey structure vary by care-setting mix, follow-up gaps, social determinant
> context, MyChart engagement, and geography?**

All analysis uses encounter-level and patient-level data spanning
January 2022 – December 31, 2025.

---

## Privacy rules

- Raw CSVs are stored **locally only** and are excluded from version control via `.gitignore`.
- No raw patient-level rows are printed or exported.
- All outputs are aggregate summaries, counts, and plots.

---

## Folder structure

```
DataFest2026/
  data/               ← raw CSVs (local only, git-ignored)
  code/
    00_data_audit.py          ← per-table shape, dtypes, missing-value audit
    01_join_validation.py     ← cross-table join checks and diagnosis inflation check
    02_build_journeys.py      ← construct patient×diagnosis journey table
    03_sdoh_features.py       ← aggregate SDOH responses to patient or journey level
    04_eda_tables.py          ← grouped aggregate tables for analysis
    05_plots.py               ← visualizations
  outputs/
    audit/            ← CSV/text summaries from 00 and 01
    tables/           ← aggregate tables from 04
    plots/            ← figures from 05
  README.md
  requirements.txt
  .gitignore
```

---

## Data files (place in `data/`)

| File | Description |
|------|-------------|
| `encounters.csv` | One row per encounter, Jan 2022 – Dec 2025 |
| `patients.csv` | Patient background (demographics, MyChart status) |
| `diagnosis.csv` | Diagnosis reference: DiagnosisKey → DiagnosisValue / GroupCode |
| `departments.csv` | Care-location reference |
| `providers.csv` | Provider reference |
| `social_determinants.csv` | SDOH survey responses at encounter level |
| `tigercensuscodes.csv` | Kansas census block group reference |

> **Note:** Raw CSVs must be copied into `data/` manually before running scripts.
> They are not committed to this repository.

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the pipeline

Run scripts from the **project root** (`DataFest2026/`):

```bash
python code/00_data_audit.py        # per-table audit
python code/01_join_validation.py   # join and inflation checks
```

Outputs land in `outputs/audit/`.

---

## Analysis roadmap

The core analytical frame is **longitudinal patient journeys**, defined as a
patient × diagnosis grouping over time (pending validation in script 01).

Planned research questions:
1. How do patients move across care settings during a diagnosis-based journey?
2. What are the most common first-to-second encounter care-setting transitions?
3. Which journeys are office-centred, lab-heavy, imaging-heavy, or hospital-involved?
4. Is MyChart status associated with observed follow-up timing or continuity?
5. Are recorded SDOH needs associated with longer gaps, more fragmented care, or
   higher hospital involvement (among patients with recorded responses)?

---

## Interpretation cautions

- Some journeys may start before or extend past the data window.
- A single-encounter journey is not inherently incomplete; acute, screening,
  procedural, newborn, obstetric, and administrative episodes may naturally
  involve only one observed encounter.
- Social determinant responses are not available for all patients; results
  should be framed as patterns **among patients with recorded responses**.
- Use language such as *associated with*, *observed among*, *linked to*, or
  *suggests* — not *causes* or *leads to*.
