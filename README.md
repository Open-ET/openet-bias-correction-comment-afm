# Comment on Golkar & de Oliveira (2026): code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23383493.svg)](https://doi.org/10.5281/zenodo.23383493)
[![Tests](https://github.com/Open-ET/openet-bias-correction-comment-afm/actions/workflows/tests.yml/badge.svg)](https://github.com/Open-ET/openet-bias-correction-comment-afm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-data.md)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](environment.yml)
[![Comment on 10.1016/j.agrformet.2026.111524](https://img.shields.io/badge/Comment%20on-10.1016%2Fj.agrformet.2026.111524-orange.svg)](https://doi.org/10.1016/j.agrformet.2026.111524)

## What this is

Code for the comment

> Kim, Y., Majumdar, S., Volk, J.M., ReVelle, P., Caldwell, T.G., Huntington, J.L.
> *Comment on Golkar and de Oliveira (2026): Missing eddy covariance observations undermine
> reported OpenET biases.* Submitted to *Agricultural and Forest Meteorology*.

on

> Golkar, M.K., de Oliveira, L.A., 2026. Diagnosing and correcting OpenET satellite
> evapotranspiration across US climate zones: Error decomposition, explainable AI, and
> uncertainty quantification. *Agric. For. Meteorol.* 391, 111524.
> https://doi.org/10.1016/j.agrformet.2026.111524 (GO26, CC BY 4.0)

The analysis uses publicly available data: GO26's own figures (digitized), the Volk et al.
(2023a) eddy covariance benchmark with the OpenET model data at its stations (Volk et al.,
2023d), the current OpenET collection extracted for the same 200 m boxes as GO26, AmeriFlux
public metadata, and the AmeriFlux BASE files of GO26's 22 towers (the versions GO26 cite;
they need an AmeriFlux account and are not redistributed). Shared code is in
`src/etcomment/`, the numbered analysis scripts are in `scripts/` and unit tests are in
`tests/`.

## Repository and Zenodo archive

This repository holds the code. The Zenodo archive (https://doi.org/10.5281/zenodo.23383493)
holds a snapshot of this repository together with the inputs and outputs that are not code,
which are not tracked in git:

| Folder | Contents |
|---|---|
| `config/` | GO26 Table 1 (sites, study periods, mean observed ET, BASE DOIs and versions) and Table 3, site notes, the Volk et al. (2024) station list |
| `data/external/golkar_deoliveira_2026/` | GO26 Figs. 8 and S2 (CC BY 4.0), digitized by `scripts/04`, and Fig. 2, kept for the visual check of its zero-valued observations (Section 1 of the comment) |
| `data/raw/ameriflux_public/` | cached public AmeriFlux metadata read by `scripts/11` (site pages and web-service responses change over time; the site pages without the site teams' e-mail addresses) |
| `results/` | derived tables (CSV, JSON) of scripts 02 (Earth Engine ETo check) and 04-12; in the deposit with `results/README.txt`, a one-line guide to every file |
| `figures/` | Figure 1 of the comment (`fig_table1_reconstruction.*`), the further figures of the analysis (`fig1_*` to `fig5_*`) and the QC overlays of the digitization (without `figures/site_plots/`, which is deposited as `benchmark_site_plots/`) |
| `benchmark_site_plots/` | benchmark plots for the seven towers shared with GO26: the benchmark's Bokeh pages (`html/`) and static summaries (`static/`). `scripts/03` writes them to `figures/site_plots/html/` and `static/`; `scripts/13` packs them, with a README.txt, as `benchmark_site_plots/` |

`scripts/13_make_zenodo_deposit.py` builds the deposit in `zenodo/` (gitignored): one zip per
folder in the table above, with paths relative to the repository root, a zip of the code, a
README (the deposit description), `description.html` (the same as HTML for the Description
field of the Zenodo record) and MD5SUMS. It checks the inputs of every archive first and
stops, writing nothing, if one is missing or incomplete; it then builds the deposit in a
temporary folder next to `zenodo/` and moves it into place, so a failed build leaves the
previous deposit as it was. The zips are deterministic (sorted entries, fixed timestamps,
permissions and creator fields), so a rebuild from unchanged files with the same Python and
zlib (`environment.yml`) gives the same md5 sums. Check a download with `md5sum -c MD5SUMS`
(macOS: `md5 -r *.zip`, and compare with `MD5SUMS`).

Download the deposit and copy `config/`, `data/external/golkar_deoliveira_2026/`,
`data/raw/ameriflux_public/`, `results/` and `figures/` into the root of your clone. Most
scripts need `config/`;
`results/` holds the outputs that later scripts read (for example the digitized series
used by `scripts/08` and `scripts/12`). The other downloads in `data/raw/` are not
deposited; the scripts below recreate them (the OpenET extracts with your own API key, the
BASE files with an AmeriFlux account).

## Setup

```bash
git clone https://github.com/Open-ET/openet-bias-correction-comment-afm.git
cd openet-bias-correction-comment-afm
conda env create -f environment.yml   # exact versions used for the published numbers
conda activate etcomment
python -m pytest -q                   # unit tests on synthetic data: offline, a few seconds
```

Without conda, `python -m pip install -e ".[dev]"` installs the package with mostly unpinned
dependencies (`fluxdataqaqc`, `kgcpy` and `basemap-data` are pinned; Python 3.11 or later; CI
tests Python 3.11 and 3.12 with the latest versions). The `ee` extra (`".[dev,ee]"`) adds
`earthengine-api` for the optional Earth Engine check of ETo.

`scripts/03` renders PNG snapshots of the benchmark's Bokeh pages with Playwright: run
`python -m playwright install chromium` once, or pass `--no-render`. The pages load
BokehJS 2.1.1 from cdn.bokeh.org; offline, pass `--bokehjs-dir` with `bokeh.min.js` and
`bokeh-widgets.min.js` from the `bokeh==2.1.1` sdist (`bokeh/server/static/js/`).

Credentials are needed only for the OpenET extraction and the optional AmeriFlux download.
Set them as environment variables or in a `.env` file in the repository root, which git
ignores. Never commit the file or add it to the archive.

```text
OPENET_API_KEY=...          # scripts/02
AMERIFLUX_USER_ID=...       # scripts/10 --download (optional)
AMERIFLUX_USER_EMAIL=...    # scripts/10 --download (optional)
```

## Data streams

| # | Data | Access | Script | Stored in |
|---|---|---|---|---|
| 1 | Benchmark flux data (Volk et al., 2023c) and OpenET Phase II model ET at its stations (Volk et al., 2023d), Zenodo | public, md5-verified | `01_download_data.py` | `data/raw/` |
| 2 | OpenET API, collection 2.1: monthly ET (ensemble and six models), ETo and precipitation for 75 boxes, 2016-2025; daily ensemble ET at nine towers | free OpenET account and API key | `02_download_openet_api.py` | `data/raw/openet_api/` |
| 3 | AmeriFlux public metadata: BASE variable availability (amerifluxr), BASE versions and years | public, no login | `11_ameriflux_availability.py` | `data/raw/ameriflux_public/` |
| 4 | AmeriFlux BASE files (BASE-BADM) of the 22 GO26 towers, the versions GO26 cite; needed only to rebuild `results/go26_*` | AmeriFlux account; CC BY 4.0, not redistributed | `10_reconstruct_go26_observed_et.py` | `data/raw/ameriflux_base/` |
| 5 | GO26 Figs. 8 and S2 (digitized) and Fig. 2 (CC BY 4.0) | Zenodo deposit | `04_digitize_afm_figures.py` | `data/external/golkar_deoliveira_2026/` |

`scripts/01` uses zips that are already in `data/raw/` or `data/external/` if their md5
matches the published sum, whatever their names, and downloads them from Zenodo otherwise.

### OpenET API (stream 2)

1. Create a free account at https://account.etdata.org and copy your API key.
2. Set `OPENET_API_KEY` (environment variable or `.env`, see Setup).
3. `python scripts/02_download_openet_api.py --dry-run` lists the requests that are not
   cached yet; run it without `--dry-run` to download.

* The boxes follow GO26 (Section 2.1): 200 m x 200 m, centred on each tower, for GO26's
  22 towers and the 60 cropland stations of the benchmark (75 boxes; seven towers are in
  both). Collection 2.1 starts in 2016 in the API.
* A run from scratch uses 24 of the 100 monthly requests of a free account and took about
  30 min. Every response is cached in `data/raw/openet_api/` with a request log
  (`manifest.json`), so reruns fetch only missing files; a complete cache needs no key.
* ETo is OpenET's bias-corrected gridMET grass reference ET (Volk et al., 2026; Earth
  Engine collections `projects/openet/assets/reference_et/conus/gridmet/{daily,monthly}/v1`);
  precipitation is gridMET. `python scripts/02_download_openet_api.py --verify-eto-ee`
  compares the cached monthly ETo with the Earth Engine collection and writes
  `results/openet_api_eto_ee_check.csv`. It needs the `ee` extra, an Earth Engine account
  authenticated with `earthengine authenticate`, and the Cloud project in the `EE_PROJECT`
  environment variable.
* OpenET data are subject to the OpenET Terms of Service
  (https://etdata.org/terms-of-service/) and must be attributed ("OpenET,
  https://openetdata.org") wherever values are shown. The raw extracts stay in `data/raw/`;
  the OpenET values in `results/` are statistics over study periods, seasons, climate
  classes and stations, apart from GO26's own plotted OpenET line in `results/digitized_*`
  (digitized from their CC BY 4.0 Figs. 8 and S2). Before writing, `scripts/12` and
  `scripts/10` each check that no single monthly or daily OpenET value, and no sum or
  difference over fewer than six site-months, can be solved for from their outputs and the
  public data, and they withhold what would allow it: small stations; the MBE per day of
  single stations and of the six models' strata; the seasonal rows of US-xSL and of the seven
  shared towers pooled; the per-tower zero-fill predictions and daily-based means; and some
  zone-by-season rows of `results/go26_reconstructed_openet_bias.csv`. `scripts/14` then
  checks all their published numbers together (rebuilt as linear functionals of the raw
  extracts and validated against the published values; nonlinear statistics through their
  gradients; an exact search for combinations of fewer than six site-months, with positive
  controls), and `scripts/13` builds the deposit only from files that passed it. The tables
  with OpenET values (the full per-tower table, the GO26 figure check by day set and month,
  the benchmark station-months and the US-UiC winter comparison) are written to
  `data/raw/openet_api/derived/` and are not deposited.

### AmeriFlux (streams 3 and 4)

* `scripts/11` needs no login. It reads per-year LE availability from amerifluxr's BASE
  variable-availability summary, pinned to commit `7319e632` (summary dated 2025-10-20;
  `--latest` uses the newest), BASE publish years and tower start years from the AmeriFlux
  web services, and each site's current BASE version from its site page. Files are cached
  with URL, access time and md5 in `data/raw/ameriflux_public/` (in the deposit; fetched on
  15 October 2026, 04:26-04:29 UTC), because the web services and site pages change over
  time; `--refresh` fetches them again. Without the cache the script stops rather than
  overwrite the archived snapshot in `results/`. The site pages list the site teams' e-mail
  addresses, which the script does not need: it replaces each one, including `mailto:`
  links, by `[email removed]` before it writes a page to the cache. `provenance.json` gives
  the md5 of each page as served (`md5_original`, the `page_md5` in
  `results/ameriflux_public_provenance.json`) and of the cached copy (`md5`).
* `scripts/10` rebuilds GO26's observed ET from the BASE files and writes `results/go26_*`.
  The files behind the archived results (per site a BASE CSV and its BASE-BADM zip, the
  versions GO26 cite at all 22 sites) were downloaded on 15 October 2026, 08:02:54-08:03:22
  UTC, in one request through the AmeriFlux data-download service, with the intended use
  "Research - Remote sensing" and a neutral description of this re-analysis (the default
  `--description`), which AmeriFlux passes on to the site teams;
  `data/raw/ameriflux_base/provenance.json` records versions, md5 sums and access times.
  To rerun, download the versions in `config/go26_sites.csv` (`base_version_go26`,
  `base_doi`) from https://ameriflux.lbl.gov into `data/raw/ameriflux_base/`, or run
  `python scripts/10_reconstruct_go26_observed_et.py --download --agree-policy --qc-flags` with
  `AMERIFLUX_USER_ID` and `AMERIFLUX_USER_EMAIL` set; `--agree-policy` confirms that you
  accept the AmeriFlux CC BY 4.0 data policy, and AmeriFlux logs each download. `--download`
  fetches the current versions; where several versions of a site are present the script
  uses the one GO26 cite, and `results/go26_observed_et_reconstruction.csv` records the
  version used (`base_version`, `version_matches_go26`).
* The script writes GO26's Table 1 as reproduced (half-hourly ET summed with missing records
  and days counted as zero, over calendar days, no closure), valid-day means with missing
  records filled from the mean diurnal cycle, energy-balance closure by flux-data-qaqc
  (Volk et al., 2021; version 0.5.0, pinned in `environment.yml`; run on the daily means of
  LE, H, NETRAD and G over the half-hours in which all four are valid, its daily factor
  multiplying the daily LE: the Bowen-ratio method as GO26 state it, flux-data-qaqc's `br`
  correction, unscreened, on the valid-day means and on the gap-filled ones; its `ebr`
  correction as the main variant; our screened Bowen ratio; the unscreened Bowen ratio on
  every day only as a diagnostic), a comparison of the cited files with AmeriFlux's October
  2025 availability summary (the `scripts/11` cache), a day-by-day check of GO26's Figs. 8
  and S2 (digitized by `scripts/04`) and, with the OpenET monthly extract (`scripts/02`), the
  OpenET bias by climate zone and season. GO26 also kept only records passing the AmeriFlux
  quality flags: `--qc-flags` adds columns `*_qc` without the LE records flagged 2 in
  `LE_SSITC_TEST`. The default uses every record, which reproduces Table 1; `results/go26_*`
  were written with `--qc-flags`.
* Without BASE files the script reports "BASE file not found" for each site but still
  rewrites `results/go26_observed_et_reconstruction.csv` (status rows only), so do not run
  it in a copy of the deposit that lacks the BASE files.

### GO26 figures (stream 5)

The figures are in the Zenodo deposit. They can also be taken from the article (CC BY 4.0):
Fig. S2 is supplementary file mmc2 (large version), and Figs. 8 and 2 are images embedded in
the article PDF (`pdfimages -png`). Only Figs. S2 and 8 are digitized; Fig. 2 is kept for the
visual check of the zero stripe in its scatter plot.

## Pipeline

Run the scripts from the repository root in numeric order, except that `10`, `11` and `12`
must run before `08` (`figures/fig2_*` uses their outputs, Figure 1 of the comment those of
`10`), `11` before `10` (`10` reads the availability summary cached by `11`) and `14` before
`13` (`13` requires the record of the joint disclosure check). Without the BASE files, skip
`10` and keep the deposited `results/go26_*`. Times are for a recent laptop with the inputs
in place.

| Script | Needs | Writes | Time |
|---|---|---|---|
| `01_download_data.py` | internet, or the zips in `data/raw/` or `data/external/` | `data/raw/flux_ET_dataset/`, `data/raw/OpenET_PhaseII_model_ET_dataset/` | 183 MB download |
| `02_download_openet_api.py` | `OPENET_API_KEY`, `config/`, benchmark station list; `--verify-eto-ee`: Earth Engine (`ee` extra, `EE_PROJECT`) | `data/raw/openet_api/`; with `--verify-eto-ee`, `results/openet_api_eto_ee_check.csv` | about 30 min the first time |
| `03_site_plots.py` | benchmark, OpenET Phase II; Playwright or `--no-render` | `figures/site_plots/` (`html/`, `static/`; `png_bokeh/` with Playwright) | seconds without rendering |
| `04_digitize_afm_figures.py` | GO26 figures | `results/digitized_*`, `figures/qc/` | seconds |
| `05_benchmark_checks.py` | benchmark, OpenET Phase II | `results/benchmark_*.csv` | seconds |
| `06_shared_site_comparison.py` | benchmark, OpenET Phase II, `config/` | `results/shared_site_*.csv`, `figures/fig4_*`, `figures/fig5_*` | seconds |
| `07_leakage_experiment.py` | benchmark, OpenET Phase II | `results/leakage_experiment*` | about 10 min (two meteorology variants) |
| `08_make_figures.py` | outputs of 04, 11 and 12, and of 10 for the GO26-tower markers in `fig2_*` and for Figure 1 of the comment; `config/` | `figures/fig1_*`, `figures/fig2_*`, `figures/fig_table1_*` (Figure 1 of the comment: GO26 Table 1 against the site means rebuilt from BASE) | seconds |
| `09_make_maps.py` | `config/` | `figures/fig3_site_map.*`, `results/site_koppen_check.csv` | seconds |
| `10_reconstruct_go26_observed_et.py` | BASE files in `data/raw/ameriflux_base/`; outputs of 02, 04 and 11; `config/` | `results/go26_observed_et_reconstruction.csv`, `results/go26_reconstructed_*.csv`, `results/go26_base_version_check.csv` | about 12 min (flux-data-qaqc runs for every site, closure and variant) |
| `11_ameriflux_availability.py` | internet on the first run, `config/` | `data/raw/ameriflux_public/`, `results/ameriflux_*` | seconds |
| `12_openet_api_analysis.py` | outputs of 02, 04, 06 and 11; benchmark; `config/` | `results/openet_api_*`, `data/raw/openet_api/derived/` | seconds |
| `13_make_zenodo_deposit.py` | the outputs above, `config/`, `data/`, the record of 14 (`--no-disclosure-check` without it) | `zenodo/` (archives, README.md, description.html, MD5SUMS) | seconds |
| `14_check_openet_disclosure.py` | outputs of 10 and 12, `data/raw/openet_api/` (raw and `derived/`), benchmark, `config/` | `data/raw/openet_api/derived/disclosure_check.json` (not deposited) | about 10 min |

```bash
python scripts/01_download_data.py
python scripts/02_download_openet_api.py
python scripts/03_site_plots.py                       # add --no-render without Chromium
G=data/external/golkar_deoliveira_2026
python scripts/04_digitize_afm_figures.py --figS2 $G/figS2_US-UTJ_mmc2_lrg.jpg --fig8 $G/fig8_US-UiC.png
python scripts/05_benchmark_checks.py
python scripts/06_shared_site_comparison.py
python scripts/07_leakage_experiment.py
python scripts/11_ameriflux_availability.py
python scripts/12_openet_api_analysis.py
python scripts/10_reconstruct_go26_observed_et.py --qc-flags   # needs the BASE files
python scripts/08_make_figures.py                     # after 10, 11 and 12
python scripts/09_make_maps.py
python scripts/14_check_openet_disclosure.py          # joint OpenET disclosure check (needs the extracts)
python scripts/13_make_zenodo_deposit.py              # package the Zenodo deposit in zenodo/
```

## Reproducibility

* Pinned environment: `environment.yml` pins the exact versions of the direct dependencies
  (conda-forge and PyPI) used for the published numbers.
* Fixed seeds: the leakage experiment uses seeds 0, 1 and 2 and records package versions,
  seeds, model settings and input md5 sums in `results/leakage_experiment_versions.json`
  (without a run time, which it prints instead, so that a rerun reproduces the file byte
  for byte).
* Byte-reproducible figures: PDFs are written without a creation date, and the figure
  scripts (`03`, `06`, `08`, `09`) rerun themselves with `PYTHONHASHSEED=0`
  (`etcomment.style.rerun_with_fixed_hash_seed`), because the font subsets matplotlib embeds
  in PDFs otherwise vary with Python's random hash seed. PNGs depend on neither.
* Verified inputs: `scripts/01` checks the md5 sums published on Zenodo; `scripts/11` pins
  the amerifluxr summary to a commit, checks its md5 and records URL, access time and md5
  of every AmeriFlux file (`data/raw/ameriflux_public/provenance.json`,
  `results/ameriflux_public_provenance.json`).
* Data dates (UTC, from the provenance files): OpenET API extraction 15 October 2026,
  03:38-04:09 UTC; AmeriFlux public metadata 15 October 2026, 04:26-04:29 UTC; AmeriFlux
  BASE files 15 October 2026, 08:02:54-08:03:22 UTC, one request (the versions GO26 cite at
  all 22 sites; `data/raw/ameriflux_base/provenance.json`, not deposited).
* OpenET extract: collection 2.1. The request log (`data/raw/openet_api/manifest.json`)
  records the parameters (including the collection version), status and UTC time of every
  call; the cache makes reruns use the same extract. OpenET revises its collections, so a
  new extraction can differ slightly; if the md5 sums of the raw extracts match those
  recorded in `results/openet_api_go26_summary.json` (`provenance`: `md5_openet_monthly.csv`,
  `md5_openet_daily.csv`, `md5_sites.csv`, `md5_sites_200m.geojson`), `results/openet_api_*` and
  `figures/fig2_*` are reproduced exactly.
* Deposit: `scripts/13` writes deterministic zips; a rebuild from unchanged files gives the
  same md5 sums with the same Python and zlib (`environment.yml`; the deposit README records
  the versions used).
* Tests: `python -m pytest -q` runs the unit tests on synthetic data (offline, a few
  seconds). GitHub Actions runs them on every push and pull request to `main` with Python
  3.11 and 3.12 and the latest dependency versions.

## Key results

The results reported in the comment, in its order, with the scripts and files in `results/`
they come from (ET in mm/day).

**Missing observations counted as zero reproduce GO26's Table 1 (Section 2, Figure 1;
`scripts/10`, `go26_observed_et_reconstruction.csv`).**

* Summing the available half-hourly ET from the AmeriFlux BASE files, with missing
  observations and entirely missing days treated as zero (pandas' `resample("D").sum()`
  default, over the calendar days of each study period, no closure), reproduces the mean
  observed ET in GO26's Table 1 within 0.04 mm/day at 20 of 22 sites (Figure 1a; mean
  difference +0.02, RMSD 0.05; within 0.005 at 14 sites and 0.01 at 17). At the two labeled
  sites, US-UTD and US-UTV (0.12 and 0.21 mm/day higher), the BASE versions examined contain
  more LE observations than the earlier versions GO26 appear to have processed.
* Restricting the analysis to valid observations, gap filling and correcting for energy
  balance closure give substantially different site means (Figure 1b-d; mean differences
  from Table 1 +0.62, +0.50 and +1.07 mm/day): (b) daily means of the valid half-hours on
  days with at least 50% of them; (c) the same, with the missing half-hours gap-filled from
  the mean diurnal cycle of a 15-day moving window; (d) as (c), on the days with all energy
  balance components (92% of the days in (c)), closed with the Bowen-ratio method described
  by GO26 (Twine et al., 2000) in flux-data-qaqc 0.5.0, without screening the daily
  correction factors.
* Daily series (`go26_reconstructed_figure_check.csv`): at US-UTJ, 206 of 353 daily
  observations in GO26's Fig. S2 are exactly zero, on dates without available LE
  measurements in the BASE file examined; at US-UiC, the zero-filled daily sums reproduce 360
  of 361 observations in GO26's Fig. 8 within 0.02 mm/day, including 79 days with incomplete
  flux records.

**Implications for OpenET biases and ML correction (Section 3).**

* On the 742 site-months at 21 sites with at least 20 days that pass GO26's 50% filter and
  have all energy balance terms (`go26_reconstructed_openet_bias.csv`, month set
  `common_valid_months`), the monthly bias of the current OpenET ensemble (collection 2.1) is
  +0.51 mm/day against zero-filled ET and +0.17 against the daily means of the valid
  observations, and -0.43 with the Bowen-ratio closure described by GO26, applied without
  screening the daily correction factors. With the missing half-hours gap-filled from the
  mean diurnal cycle (a step GO26 do not describe), it is +0.30 before and -0.24 after the
  same Bowen-ratio closure, and -0.22 with flux-data-qaqc's energy-balance-ratio (EBR)
  closure.
* Against measured monthly ET at 36 cropland stations of the Volk et al. (2023a) benchmark
  (2016-2022; closure-corrected eddy covariance at 32 stations and weighing lysimeters at
  four; 550 months with at most five gap-filled days), the ensemble bias is -0.22 mm/day
  (`scripts/12`, `openet_api_benchmark_summary.csv`).
* At US-UTJ, the observations and predictions digitized from GO26's Fig. S2 (`scripts/04`,
  `digitized_summary.json`) reproduce the reported XGBoost reduction in RMSE (52.7%; GO26:
  52.8%; OpenET RMSE 1.88 vs 1.877 printed, XGBoost 0.89 vs 0.885). Excluding the
  zero-valued reference observations reduces it to 0.5% (OpenET RMSE 1.04, XGBoost 1.03).

**Additional methodological concerns (Section 4).**

* Energy balance correction: GO26's site means are reproduced only with uncorrected
  turbulent fluxes and missing data counted as zero. With the Bowen-ratio closure as GO26
  describe it, site means are 0.29-2.12 mm/day above Table 1 (mean +1.24; `mean_et_br_fdq`),
  and 0.28-1.74 mm/day (mean +1.07; `mean_et_br_fdq_mdv`, Figure 1d) with the gap-filled
  means.
* Machine learning validation (`scripts/07`, `leakage_experiment_summary.csv`; three seeds,
  38 cropland stations of the benchmark, 5,093 overpass-day records): the RMSE reduction of
  a GO26-like XGBoost correction of OpenET ranges from 9% to 26% with the validation
  strategy: 26% with a record-level random split, 23% with a site-grouped hold-out, 19% with
  leave-one-site-out and 9% when clusters of stations within 10 km are held out together.

## Further results (not in the comment)

**GO26's Table 1 and daily series (`scripts/04`, `scripts/10`).**

* At US-UTJ, removing the zero-valued observations also changes the OpenET bias from +1.28
  to +0.45 mm/day; with them, the digitized site mean is 0.62 mm/day (0.59 in GO26's Table
  1). At US-UiC the 4 plotted days without LE are exact zeros; at US-UTJ, 11 plotted non-zero
  days have no LE in the cited file (consistent with GO26 having processed an earlier
  version). The first BASE LE record is from 5 June 2024 at US-UTJ (GO26's zero run ends on 4
  June) and from 25 May, 5 September and 27 September 2023 at US-UTD, US-NSb and US-UTV.
* Dividing the zero-filled sums by the days up to each file's last record gives 16 sites
  within 0.005 mm/day (the US-IAB, US-RGB and US-VT1 files end early; largest difference over
  calendar days 0.008, US-IAB). At the three Mead towers (US-Ne1-Ne3; hourly files, the
  version GO26 cite) the means are 0.03 mm/day (2%) higher, for reasons we could not find.
* **BASE versions.** The files are the versions GO26 cite, but at 11 sites these are newer
  than the versions in AmeriFlux's availability summary of October 2025: the cited files
  run 8-13 months beyond GO26's study periods, which end in the final year of the earlier
  versions' LE (`go26_base_version_check.csv`). GO26 accessed the data in March 2026 and
  appear to have processed the earlier versions. At US-UTD and US-UTV the cited versions
  hold at least 829 (6.8%) and 2,711 (14.2%) more LE half-hours in the study periods.
* **Valid days and closure** (`go26_observed_et_reconstruction.csv`,
  `go26_reconstructed_daily.csv`). LE is missing mostly at night at some towers, so filling
  the missing records of valid days from the mean diurnal cycle lowers their means by
  15-18% at US-MN1-MN3 and US-Mo1. Unscreened, the daily Bowen-ratio factor is negative or
  outside 0.5-2 on 2,794 of 23,357 energy-balance days (12.0%) at the 21 towers with net
  radiation (1.7-26.2% by tower; 81% in November-March), and the closed daily ET it gives
  ranges from -12,389 mm/day (US-Mo1, 4 December 2024) to 9,533 mm/day (US-UTV, 11 January
  2024). flux-data-qaqc's `br` correction leaves out 97 of the 23,357 energy-balance days
  (closed LE <= -100 or >= 850 W/m2). Its `ebr` correction gives no factor where its factors
  fall outside 0.5-2 around a day and no other year or day of year fills it: at US-UTV from
  November 2023 to January 2024, when NETRAD - G is near zero or negative, so its EBR-closed
  site mean covers mostly the growing season.
  US-WT1 has no NETRAD or G; G is 0 at US-IAB, US-MN1 and US-MN2 (passed to flux-data-qaqc as
  NETRAD + 1 and G = 1 W/m2, the same NETRAD - G, because it drops an all-zero column). The
  Utah towers have two net radiometers that disagree: with the other one, the EBR-closed
  gap-filled means change from 2.11 to 2.59 (US-UTD), 2.30 to 2.96 (US-UTJ) and 2.24 to
  1.04 mm/day (US-UTV).
* **OpenET at GO26's towers** (`go26_reconstructed_openet_bias.csv`; collection 2.1,
  monthly). Against the zero-filled monthly observations (1,019 site-months, 22 towers) the
  bias is +0.65 mm/day, and +0.09, +0.45, +1.44 and +0.56 in winter, spring, summer and
  fall, close to GO26's +0.60 and +0.08, +0.41, +1.34 and +0.57. On the 742 common
  site-months the screened Bowen ratio (our screen) gives -0.21; in summer the variants give
  +1.10 (zero-filled); +0.57 and -0.41 (valid-day, unclosed and Bowen-ratio closed); +0.77,
  -0.12, -0.14 and -0.13 (gap-filled: unclosed, Bowen ratio, EBR, screened Bowen ratio).

**Seven towers shared with the Volk et al. (2023a) benchmark (`scripts/06`; `results/shared_site_*`, `figures/fig4_*`, `fig5_*`).**

* OpenET ensemble bias against measured EBR-closed ET: -0.07 to -0.58 mm/day on overpass
  days (all seven towers); -0.12 to -0.35 mm/day for months with at most five gap-filled
  days at five of six towers (+0.14 at US-ARM). Monthly R2 (1 - SSE/SST) 0.81-0.91. GO26
  report zone biases of +0.48 to +1.10 mm/day and a monthly R2 of 0.14.
* GO26's mean observed ET is 16-32% below the benchmark's long-term mean at the six
  well-measured towers (57% at US-xSL), below the lowest annual benchmark mean at six of
  seven towers, and below the benchmark's unclosed ET at all six.

**Benchmark-wide (`scripts/05`).** 6 of 211,269 measured station-days at 161 stations are
exactly zero (none at cropland stations); the longest run within 0.015 mm/day of zero is
17 days (4 at cropland stations). Daily Bowen-ratio and EBR closure agree on average
(median difference 0.00 mm/day at 51 cropland EC stations), but the daily Bowen-ratio factor
misbehaves on a median 5% of days, mostly in November-March.

**Leakage experiment, details (`scripts/07`).** Leave-one-cluster-out raises the model RMSE
by 12% over leave-one-site-out, similar to GO26's increase from 0.804 to 0.897 mm/day.
Records given zero targets and climate-zone median meteorology were predicted at 1.0
(random split) and 1.5 mm/day (leave-one-site-out), against 2.8-2.9 mm/day for OpenET and
about 3.2 mm/day measured. Stations without tower meteorology carry the same medians on
real days, as imputation does in GO26's data, which dilutes the zero-fill signature; with
missing meteorology left unfilled on all other days
(`results/leakage_experiment_met_unfilled*`), the predictions on the zero-filled records
fall to 0.34 and 0.71 mm/day, and the clean gains are 26%, 24%, 20% and 10%.

### Current OpenET (API, collection 2.1) and AmeriFlux data availability (`scripts/11`, `scripts/12`)

* **LE availability.** In AmeriFlux's public BASE variable-availability summary, latent heat
  flux is present on 38-97% of the half-hours in GO26's study periods (mean 68%; below 60% at
  US-UTD, US-UTJ, US-NSb, US-xSL, US-UTV, US-UiC, US-MN1 and US-MN3; none for 2022 at US-UiC).
  The summary is dated 20 October 2025; at 11 sites it predates the 2026 BASE version GO26
  cite and describes an earlier version, and shares for partial final years are upper
  bounds. In the BASE files of the cited versions the share is 40-97% (mean 68%;
  `results/go26_observed_et_reconstruction.csv`, `le_record_availability`).
* **GO26's site means track availability, as zero-filling predicts.** Across the 22 towers
  the ratio of GO26's Table 1 mean to the OpenET mean over the same period follows the
  availability (r = 0.79, Spearman 0.82; ratio = 0.93 x availability). OpenET scaled by
  each year's availability tracks GO26's means (r = 0.84, slope 0.92; RMSE 0.27 mm/day, 0.25
  about the fitted line); OpenET alone does not (r = 0.33).
* **Same OpenET as GO26.** The API's daily ensemble matches the OpenET line GO26 plotted
  at US-UiC in 2020 (r = 0.991; GO26 = 0.996 x API; mean difference +0.02 mm/day over 361
  days). At US-UTJ in 2024 the day-to-day series agrees (r = 0.956), but GO26's line is
  lower (GO26 = 0.84 x API; -0.16 mm/day over 366 days;
  `results/openet_api_go26_figure_check.csv`).
* **Irrigated sites in dry climates.** OpenET puts annual ET over GO26's periods at
  589-1,045 mm/yr at US-UTJ, US-UTD, US-UTV and US-NSb, 3.3-4.8 times the mean annual
  precipitation in GO26's Table 1 (2.9-4.6 times gridMET precipitation over the periods),
  against 215-281 mm/yr in GO26's Table 1.
* **Current OpenET against the benchmark.** Against measured, closure-corrected EC (or
  weighing-lysimeter) ET at 36 benchmark cropland stations (550 months with at most five
  gap-filled days, 2016-2022), the collection 2.1 ensemble bias is -0.22 mm/day (-6.7
  mm/month, R2 = 0.88): -0.05 in winter, -0.35 in spring, -0.39 in summer and +0.00 in fall,
  where GO26 report +0.08, +0.41, +1.34 and +0.57. By class it is -0.34 (BSk), -0.13 (Cfa),
  -0.19 (Csa) and -0.21 mm/day (Dfa) against GO26's +0.48 to +1.10. SIMS is the only model
  without a negative pooled bias (+0.25 mm/month). On all measured days at the seven shared
  towers (2016-2020) the bias is -0.33 to +0.01 mm/day (pooled -0.23); against unclosed ET it
  is +0.09 to +0.44.
* **Winter at US-UiC (GO26 Fig. 8).** GO26's January-February 2020 values (mean 0.75
  mm/day) are close to the benchmark's nearest cropland tower (US-Bo1, 10 km; 0.67 mm/day)
  and to the zero-filled, unclosed BASE sums (0.76 mm/day;
  `results/go26_reconstructed_winter.csv`), so the winter check gives no evidence of
  inflated winter ET there.
* **ETo check.** The API's monthly ETo equals OpenET's bias-corrected gridMET collection on
  Earth Engine at all 75 boxes (median difference 0.00 mm/month, maximum 0.66).

## Files not tracked in git

* `data/raw/`: downloads (`scripts/01`, `02` and `11`; BASE files for `scripts/10`) and the
  tables with OpenET values that `scripts/12` writes to `data/raw/openet_api/derived/`. Not
  archived, except `data/raw/ameriflux_public/`; the scripts recreate the rest (the OpenET
  extracts need your own API key, the BASE files an AmeriFlux account). `data/README.md` is
  tracked and describes every folder.
* `config/`, `data/external/golkar_deoliveira_2026/`, `data/raw/ameriflux_public/`,
  `results/`, `figures/`: in the Zenodo deposit, together with a snapshot of the code;
  `figures/site_plots/html/` and `static/` are deposited as `benchmark_site_plots/`
  (`figures/site_plots/png_bokeh/` and `figures/.gitkeep` are not deposited).
* `paper/`, `docs/internal/`: manuscript drafts and working notes, kept locally.
* `.env`: credentials.

## Data and licenses

* Code: MIT license (`LICENSE`).
* Derived data and figures in `results/` and `figures/`, and the tables in `config/`
  (transcribed from GO26 and Volk et al., 2024, both CC BY 4.0): CC BY 4.0
  (`LICENSE-data.md`), except the values derived from the OpenET API.
* `benchmark_site_plots/` (deposit): CC BY 4.0; `html/` are the benchmark's Bokeh pages
  (Volk et al., 2023c), unchanged; `static/` is this work.
* OpenET API data (collection 2.1, extracted 15 October 2026, 03:38-04:09 UTC): subject to
  the OpenET Terms of Service (https://etdata.org/terms-of-service/); attribute "OpenET,
  https://openetdata.org" wherever values are shown. The raw extracts and the tables with
  OpenET values (`data/raw/openet_api/derived/`) are not redistributed; the derived
  statistics (`results/openet_api_*`, `results/go26_reconstructed_openet_bias.csv`) and the
  figure panels that show them (panels b-d of `figures/fig2_*`) are not covered by CC BY 4.0
  (`LICENSE-data.md`).
* `data/external/golkar_deoliveira_2026/`: figures from GO26, CC BY 4.0, reproduced
  unchanged for re-analysis with attribution.
* Benchmark flux data: Volk et al. (2023c), doi:10.5281/zenodo.7636781, CC BY 4.0.
* OpenET model data: Volk et al. (2023d), doi:10.5281/zenodo.10119477, CC BY 4.0.
* AmeriFlux public metadata (`data/raw/ameriflux_public/`, fetched 15 October 2026,
  04:26-04:29 UTC): AmeriFlux content; the metadata (amerifluxr's variable-availability
  summary, AmeriFlux web services, site pages) are CC BY 4.0 under the AmeriFlux data
  policy, and the site pages are included, with the e-mail addresses removed, for
  provenance. `results/ameriflux_*`: CC BY 4.0 (attribute AmeriFlux and the site teams).
* AmeriFlux BASE data (downloaded 15 October 2026, 08:02:54-08:03:22 UTC): CC BY 4.0 under
  the AmeriFlux CC-BY-4.0 data policy; not redistributed. The derivatives
  `results/go26_observed_et_reconstruction.csv`, `go26_reconstructed_daily.csv`,
  `go26_reconstructed_winter.csv`, `go26_base_version_check.csv` and
  `go26_reconstructed_figure_check.csv` (with GO26's digitized values, CC BY 4.0) are CC BY
  4.0: cite each site's BASE DOI (`config/go26_sites.csv`; also in the `base_doi` column and
  the header of the daily file). `results/go26_reconstructed_openet_bias.csv` holds
  OpenET-derived statistics (OpenET Terms of Service, above).
* The AmeriFlux sites and the funding of AmeriFlux data resources are acknowledged below
  (Acknowledgements), as the AmeriFlux CC-BY-4.0 data policy asks.
* Koppen-Geiger grid: Rubel et al. (2017), via the `kgcpy` package.
* State boundaries: U.S. Census Bureau county layer, via `basemap-data`.

## Acknowledgements

Funding provided by U.S. Geological Survey Water Resources Research Institute, OpenET Planning
Phase III Grant G25AC00180-00. We acknowledge the following AmeriFlux sites for their data
records: US-ARM, US-Bi1, US-Bi2, US-IAB, US-MN1, US-MN2, US-MN3, US-Mo1, US-NSb, US-Ne1,
US-Ne2, US-Ne3, US-RGA, US-RGB, US-UTD, US-UTJ, US-UTV, US-UiC, US-UiG, US-VT1, US-WT1, and
US-xSL. In addition, funding for AmeriFlux data resources was provided by the U.S. Department
of Energy's Office of Science.

## Use of AI tools

The authors identified potential methodological issues in the original study and used Claude
(Anthropic) to assist in developing and refining code to investigate these issues and
independently reproduce the reported results. All code, analyses, and interpretations were
reviewed and verified by the authors, who retain full responsibility for the findings and
conclusions. The analysis code is publicly available on GitHub to support transparency and
reproducibility.

## Citation

Cite the archive as given in `CITATION.cff` (on GitHub, "Cite this repository"):

> Kim, Y., Majumdar, S., Volk, J.M., ReVelle, P., Caldwell, T.G., & Huntington, J.L. (2026). Code
> and data for: Comment on Golkar and de Oliveira (2026): Missing eddy covariance observations
> undermine reported OpenET biases (Version 1.0.0) [dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.23383493

Please also cite the comment once published, GO26, the works below
and the AmeriFlux site DOIs of any BASE data you use (for `results/go26_*`, the 22 BASE DOIs
in `config/go26_sites.csv`).

* Melton, F.S., et al., 2022. OpenET: filling a critical data gap in water management for
  the western United States. J. Am. Water Resour. Assoc. 58, 971-994.
  https://doi.org/10.1111/1752-1688.12956
* Rubel, F., Brugger, K., Haslinger, K., Auer, I., 2017. The climate of the European Alps:
  shift of very high resolution Köppen-Geiger climate zones 1800-2100. Meteorol. Z. 26,
  115-125. https://doi.org/10.1127/metz/2016/0816
* Volk, J.M., et al., 2021. flux-data-qaqc: a Python package for energy balance closure and
  post-processing of eddy flux data. J. Open Source Softw. 6, 3418.
  https://doi.org/10.21105/joss.03418 (version 0.5.0 is used)
* Volk, J.M., et al., 2023a. Development of a benchmark eddy flux evapotranspiration
  dataset for evaluation of satellite-driven evapotranspiration models over the CONUS.
  Agric. For. Meteorol. 331, 109307. https://doi.org/10.1016/j.agrformet.2023.109307
* Volk, J.M., et al., 2023b. Post-processed data and graphical tools for a CONUS-wide eddy
  flux evapotranspiration dataset. Data Brief 48, 109274.
  https://doi.org/10.1016/j.dib.2023.109274
* Volk, J.M., et al., 2023c. Post-processed data and graphical tools for a CONUS-wide eddy
  flux evapotranspiration dataset, version 1.0.0 [dataset]. Zenodo.
  https://doi.org/10.5281/zenodo.7636781
* Volk, J.M., et al., 2023d. OpenET model data for assessing the accuracy of OpenET
  satellite-based evapotranspiration data to support water resource and land management
  applications, v1 [dataset]. Zenodo. https://doi.org/10.5281/zenodo.10119477
* Volk, J.M., et al., 2024. Assessing the accuracy of OpenET satellite-based
  evapotranspiration data to support water resource and land management applications.
  Nat. Water 2, 193-205. https://doi.org/10.1038/s44221-023-00181-7
* Volk, J.M., et al., 2026. Assessing and correcting bias in gridded reference
  evapotranspiration over agricultural lands across the contiguous United States.
  Agric. Water Manag. 333, 110647. https://doi.org/10.1016/j.agwat.2026.110647
