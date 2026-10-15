# Comment on Golkar & de Oliveira (2026): code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23383493.svg)](https://doi.org/10.5281/zenodo.23383493)
[![Tests](https://github.com/montimaj/openet-bias-correction-comment-afm/actions/workflows/tests.yml/badge.svg)](https://github.com/montimaj/openet-bias-correction-comment-afm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-data.md)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](environment.yml)
[![Comment on 10.1016/j.agrformet.2026.111524](https://img.shields.io/badge/Comment%20on-10.1016%2Fj.agrformet.2026.111524-orange.svg)](https://doi.org/10.1016/j.agrformet.2026.111524)

## What this is

Code for the comment

> Kim, Y., Majumdar, S., Volk, J.M., ReVelle, P., Caldwell, T.G., Huntington, J.L.
> *Comment on Golkar and de Oliveira (2026): Exact-zero observed evapotranspiration
> undermines the diagnosis of OpenET bias.* Submitted to *Agricultural and Forest Meteorology*.

on

> Golkar, M.K., de Oliveira, L.A., 2026. Diagnosing and correcting OpenET satellite
> evapotranspiration across US climate zones: Error decomposition, explainable AI, and
> uncertainty quantification. *Agric. For. Meteorol.* 391, 111524.
> https://doi.org/10.1016/j.agrformet.2026.111524 (GO26, CC BY 4.0)

The analysis uses publicly available data: GO26's own figures (digitized), the Volk et al.
(2023a) eddy covariance benchmark with the OpenET model data at its stations (Volk et al.,
2023d), the current OpenET collection extracted for the same 200 m boxes as GO26, and
AmeriFlux metadata and, optionally, BASE files. Shared code is in `src/etcomment/`, the
numbered analysis scripts are in `scripts/` and unit tests are in `tests/`.

## Repository and Zenodo archive

This repository holds the code. The Zenodo archive (https://doi.org/10.5281/zenodo.23383493)
holds a snapshot of this repository together with the inputs and outputs that are not code,
which are not tracked in git:

| Folder | Contents |
|---|---|
| `config/` | GO26 Table 1 (sites, study periods, mean observed ET, BASE DOIs and versions) and Table 3, site notes, the Volk et al. (2024) station list |
| `data/external/golkar_deoliveira_2026/` | GO26 Figs. 8 and S2 (CC BY 4.0), digitized by `scripts/04`, and Fig. 2, kept for the visual check in Section 2.1 of the comment |
| `data/raw/ameriflux_public/` | cached public AmeriFlux metadata read by `scripts/11` (site pages and web-service responses change over time) |
| `results/` | derived tables (CSV, JSON) |
| `figures/` | figures of the comment and QC plots |
| `benchmark_site_plots/` | benchmark plots for the seven towers shared with GO26 (Bokeh pages and static summaries, copied from `figures/site_plots/`, written by `scripts/03`) |

`scripts/13_make_zenodo_deposit.py` builds the deposit in `zenodo/` (gitignored): one zip per
folder below, with paths relative to the repository root, a zip of the code, a README (the
deposit description) and MD5SUMS. The zips are deterministic, so a rebuild from unchanged files
gives the same md5 sums.

Download the deposit and copy `config/`, `data/external/golkar_deoliveira_2026/`,
`data/raw/ameriflux_public/`, `results/` and `figures/` into the root of your clone. Most
scripts need `config/`;
`results/` holds the outputs that later scripts read (for example the digitized series
used by `scripts/08` and `scripts/12`). The other downloads in `data/raw/` are not
deposited; the scripts below recreate them (the OpenET extracts with your own API key).

## Setup

```bash
git clone https://github.com/montimaj/openet-bias-correction-comment-afm.git
cd openet-bias-correction-comment-afm
conda env create -f environment.yml   # exact versions used for the published numbers
conda activate etcomment
python -m pytest -q                   # unit tests on synthetic data: offline, a few seconds
```

Without conda, `python -m pip install -e ".[dev]"` installs the package with mostly unpinned
dependencies (`kgcpy` and `basemap-data` are pinned; Python 3.11 or later; CI tests Python 3.11
and 3.12 with the latest versions). The `ee`
extra (`".[dev,ee]"`) adds `earthengine-api` for the optional Earth Engine check of ETo.

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
| 4 | AmeriFlux BASE files of the 22 GO26 towers (optional) | AmeriFlux account | `10_reconstruct_go26_observed_et.py` | `data/raw/ameriflux_base/` |
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
  `results/` holds statistics over study periods, seasons, climate classes and stations, and
  monthly means of the API daily ensemble only for GO26's two plotted series (US-UiC 2020,
  US-UTJ 2024). Stations with fewer than six complete months, and further stations of the
  same climate class where needed, are withheld, and `scripts/12` checks before writing that
  no single monthly or daily OpenET value can be solved for from these files and the public
  data. The station-month table and the US-UiC winter comparison with OpenET values are
  written to `data/raw/openet_api/derived/` and are not deposited.

### AmeriFlux (streams 3 and 4)

* `scripts/11` needs no login. It reads per-year LE availability from amerifluxr's BASE
  variable-availability summary, pinned to commit `7319e632` (summary dated 2025-10-20;
  `--latest` uses the newest), BASE publish years and tower start years from the AmeriFlux
  web services, and each site's current BASE version from its site page. Files are cached
  with URL, access time and md5 in `data/raw/ameriflux_public/` (in the deposit), because
  the web services and site pages change over time; `--refresh` fetches them again. Without
  the cache the script stops rather than overwrite the archived snapshot in `results/`.
* `scripts/10` (optional) rebuilds GO26's observed ET from the BASE files. With an
  AmeriFlux account, download the versions GO26 cite (`config/go26_sites.csv`, columns
  `base_version_go26` and `base_doi`) from https://ameriflux.lbl.gov and save the
  `AMF_<SITE>_BASE_*.csv` files in `data/raw/ameriflux_base/`. Alternatively,
  `python scripts/10_reconstruct_go26_observed_et.py --download --agree-policy`, with
  `AMERIFLUX_USER_ID` and `AMERIFLUX_USER_EMAIL` set, fetches the current versions through
  the AmeriFlux data-download service; `--agree-policy` confirms that you accept the
  AmeriFlux CC BY 4.0 data policy, and AmeriFlux logs each download and reports it to the
  site teams. Without BASE files the script reports "BASE file not found" for each site.
  GO26 also kept only records passing the AmeriFlux quality flags; `--qc-flags` adds a
  variant that drops LE records flagged 2 in `LE_SSITC_TEST` (the co-author's match of
  Table 1 used no flags).

### GO26 figures (stream 5)

The figures are in the Zenodo deposit. They can also be taken from the article (CC BY 4.0):
Fig. S2 is supplementary file mmc2 (large version), and Figs. 8 and 2 are images embedded in
the article PDF (`pdfimages -png`). Only Figs. S2 and 8 are digitized; Fig. 2 is kept for the
visual check of the zero stripe in its scatter plot.

## Pipeline

Run the scripts from the repository root in numeric order, except that `11` and `12` must
run before `08` (Fig. 2 uses their outputs). Times are for a recent laptop with the inputs
in place.

| Script | Needs | Writes | Time |
|---|---|---|---|
| `01_download_data.py` | internet, or the zips in `data/raw/` or `data/external/` | `data/raw/flux_ET_dataset/`, `data/raw/OpenET_PhaseII_model_ET_dataset/` | 183 MB download |
| `02_download_openet_api.py` | `OPENET_API_KEY`, `config/`, benchmark station list | `data/raw/openet_api/` | about 30 min the first time |
| `03_site_plots.py` | benchmark, OpenET Phase II; Playwright or `--no-render` | `figures/site_plots/` | seconds without rendering |
| `04_digitize_afm_figures.py` | GO26 figures | `results/digitized_*`, `figures/qc/` | seconds |
| `05_benchmark_checks.py` | benchmark, OpenET Phase II | `results/benchmark_*.csv` | seconds |
| `06_shared_site_comparison.py` | benchmark, OpenET Phase II, `config/` | `results/shared_site_*.csv`, `figures/fig4_*`, `figures/fig5_*` | seconds |
| `07_leakage_experiment.py` | benchmark, OpenET Phase II | `results/leakage_experiment*` | about 10 min (two meteorology variants) |
| `08_make_figures.py` | outputs of 04, 11 and 12; `config/` | `figures/fig1_*`, `figures/fig2_*` | seconds |
| `09_make_maps.py` | `config/` | `figures/fig3_site_map.*`, `results/site_koppen_check.csv` | seconds |
| `10_reconstruct_go26_observed_et.py` | BASE files (optional), `data/raw/openet_api/`, `config/` | `results/go26_observed_et_reconstruction.csv`, `results/go26_reconstructed_*.csv` | seconds without BASE files |
| `11_ameriflux_availability.py` | internet on the first run, `config/` | `data/raw/ameriflux_public/`, `results/ameriflux_*` | seconds |
| `12_openet_api_analysis.py` | outputs of 02, 04, 06 and 11; benchmark; `config/` | `results/openet_api_*`, `data/raw/openet_api/derived/` | seconds |
| `13_make_zenodo_deposit.py` | the outputs above, `config/`, `data/` | `zenodo/` (archives, README.md, MD5SUMS) | about 30 s |

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
python scripts/08_make_figures.py                     # after 11 and 12
python scripts/09_make_maps.py
python scripts/10_reconstruct_go26_observed_et.py     # optional: needs BASE files
python scripts/13_make_zenodo_deposit.py              # package the Zenodo deposit in zenodo/
```

## Reproducibility

* Pinned environment: `environment.yml` pins the exact versions of the direct dependencies
  (conda-forge and PyPI) used for the published numbers.
* Fixed seeds: the leakage experiment uses seeds 0, 1 and 2 and records package versions,
  seeds, model settings and input md5 sums in `results/leakage_experiment_versions.json`.
* Verified inputs: `scripts/01` checks the md5 sums published on Zenodo; `scripts/11` pins
  the amerifluxr summary to a commit, checks its md5 and records URL, access time and md5
  of every AmeriFlux file (`data/raw/ameriflux_public/provenance.json`,
  `results/ameriflux_public_provenance.json`).
* OpenET extract: collection 2.1, accessed on 15 October 2026 (UTC). The request log
  (`data/raw/openet_api/manifest.json`) records the parameters (including the collection
  version), status and UTC time of every call; the cache makes reruns use the same
  extract. OpenET collections are updated over time, so a new extraction can differ
  slightly.
* Tests: `python -m pytest -q` runs the unit tests on synthetic data (offline, a few
  seconds). GitHub Actions runs them on every push and pull request to `main` with Python
  3.11 and 3.12 and the latest dependency versions.

## Key results

**GO26's own figure (US-UTJ, 2024; `scripts/04`).** 206 of 353 plotted daily observations
are exactly 0.00 mm/day (within 0.015 mm/day). Including them reproduces GO26's printed
statistics (OpenET RMSE 1.88 vs 1.877 printed; XGBoost 0.89 vs 0.885; improvement 52.7% vs
52.8%; site mean 0.62 vs 0.59 mm/day in GO26 Table 1). Removing them gives OpenET RMSE 1.04
and XGBoost RMSE 1.03 (0.5% improvement), and an OpenET bias of +0.45 instead of +1.28 mm/day.

**GO26's Table 1 from AmeriFlux BASE (`scripts/10`).** The script rebuilds GO26's observed
ET from AmeriFlux BASE files and compares their Table 1 site means with (a) means over
valid days, (b) means with missing days counted as zero and (c) the mean of daily sums of
half-hourly ET with missing half-hours and days counted as zero and no closure, which is
what pandas' `resample("D").sum()` returns by default. A co-author's run of (c) matched
GO26's Table 1 within 0.02 mm/day at 18 of the 22 sites and within 0.03 mm/day at 21
(US-UTV: 0.12 mm/day; the co-author's files were downloaded on 7 April 2026, and the Mead
files were BASE version 19-5, whereas GO26 used 20-5). The current BASE versions equal those
GO26 cite at all 22 sites (`scripts/11`), so a download today retrieves GO26's files; rerun
`scripts/10` on them (`--download --agree-policy`) before citing numbers.

**Seven towers shared with the Volk et al. (2023a) benchmark (`scripts/06`; Table 4, Figs. 4 and 5).**

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

**Leakage experiment (`scripts/07`, three seeds, 38 cropland stations, 5,093 overpass-day
records).** XGBoost RMSE gain over OpenET: 26% with a record-level random split, 23% with a
site-grouped hold-out, 19% with leave-one-site-out and 9% when clusters of stations within
10 km are held out together (leave-one-cluster-out; model RMSE 12% higher than under
leave-one-site-out, similar to GO26's increase from 0.804 to 0.897 mm/day). Records given
zero targets and climate-zone median meteorology were predicted at 1.0 (random split) and
1.5 mm/day (leave-one-site-out), against 2.8-2.9 mm/day for OpenET and about 3.2 mm/day
measured. Stations without tower meteorology carry the same medians on real days, as
imputation does in GO26's data, which dilutes the zero-fill signature; with missing
meteorology left unfilled on all other days (`results/leakage_experiment_met_unfilled*`),
the predictions on the zero-filled records fall to 0.34 and 0.71 mm/day, and the clean
gains are 26%, 24%, 20% and 10%.

### Current OpenET (API, collection 2.1) and AmeriFlux data availability (`scripts/11`, `scripts/12`)

* **LE availability.** In AmeriFlux's public BASE variable-availability summary, latent heat
  flux is present on 38-97% of the half-hours in GO26's study periods (mean 68%; below 60% at
  US-UTD, US-UTJ, US-NSb, US-xSL, US-UTV, US-UiC, US-MN1 and US-MN3; none for 2022 at US-UiC).
  The summary is dated 20 October 2025; at 11 sites it predates the 2026 BASE version GO26
  cite and describes the preceding version, and shares for partial final years are upper
  bounds.
* **GO26's site means track availability, as zero-filling predicts.** Across the 22 towers
  the ratio of GO26's Table 1 mean to the OpenET mean over the same period follows the
  availability (r = 0.79, Spearman 0.82; ratio = 0.93 x availability). OpenET scaled by
  each year's availability tracks GO26's means (r = 0.84, slope 0.92; RMSE 0.27 mm/day, 0.25
  about the fitted line); OpenET alone does not (r = 0.33).
* **Same OpenET as GO26.** The API's daily ensemble matches the OpenET line GO26 plotted
  at US-UiC in 2020 (r = 0.991; monthly means within 0.2 mm/day). At US-UTJ in 2024 the
  day-to-day series agrees (r = 0.956), but GO26's line is 0.7-1.2 mm/day lower in May-August.
* **Irrigated sites in dry climates.** OpenET puts annual ET over GO26's periods at
  589-1,045 mm/yr at US-UTJ, US-UTD, US-UTV and US-NSb, 3.3-4.8 times the mean annual
  precipitation in GO26's Table 1 (2.9-4.6 times gridMET precipitation over the periods),
  against 215-281 mm/yr in GO26's Table 1.
* **Current OpenET against the benchmark.** Against measured, closure-corrected EC (or
  weighing-lysimeter) ET at 36 benchmark cropland stations (550 months with at most five
  gap-filled days, 2016-2022),
  the collection 2.1 ensemble bias is -0.22 mm/day (-6.7 mm/month, R2 = 0.88): -0.05 in winter,
  -0.35 in spring, -0.39 in summer and +0.00 in fall, where GO26 report +0.08, +0.41, +1.34
  and +0.57. By class it is -0.34 (BSk), -0.13 (Cfa), -0.19 (Csa) and -0.21 mm/day (Dfa)
  against GO26's +0.48 to +1.10. SIMS is the only model without a negative pooled bias
  (+0.25 mm/month). On all measured days at the seven shared towers (2016-2020) the bias is -0.33
  to +0.01 mm/day (pooled -0.23); against unclosed ET it is +0.09 to +0.44.
* **Winter at US-UiC (GO26 Fig. 8).** GO26's January-February 2020 values (mean 0.75
  mm/day) are close to the benchmark's nearest cropland tower (US-Bo1, 10 km; 0.67 mm/day),
  so the winter check gives no evidence of inflated winter ET there.
* **ETo check.** The API's monthly ETo equals OpenET's bias-corrected gridMET collection on
  Earth Engine at all 75 boxes (median difference 0.00 mm/month, maximum 0.66).

## Files not tracked in git

* `data/raw/`: downloads (`scripts/01`, `02` and `11`; BASE files for `scripts/10`). Not
  archived, except `data/raw/ameriflux_public/`; the scripts recreate the rest (the OpenET
  extracts need your own API key). `data/README.md` is tracked and describes every folder.
* `config/`, `data/external/golkar_deoliveira_2026/`, `data/raw/ameriflux_public/`,
  `results/`, `figures/`, `benchmark_site_plots/`: in the Zenodo deposit, together with a
  snapshot of the code.
* `paper/`, `docs/internal/`: manuscript drafts and working notes, kept locally.
* `.env`: credentials.

## Data and licenses

* Code: MIT license (`LICENSE`).
* Derived data and figures in `results/`, `figures/` and `benchmark_site_plots/`, and the
  tables in `config/` (transcribed from GO26 and Volk et al., 2024, both CC BY 4.0):
  CC BY 4.0 (`LICENSE-data.md`), except the values derived from the OpenET API.
* OpenET API data (collection 2.1): subject to the OpenET Terms of Service
  (https://etdata.org/terms-of-service/); attribute "OpenET, https://openetdata.org"
  wherever values are shown. The raw extracts and per-station-month comparisons stay in
  `data/raw/openet_api/` and are not redistributed; the derived statistics
  (`results/openet_api_*`, `results/go26_reconstructed_openet_bias.csv`, including monthly means
  of GO26's two plotted series) and the figure panels that show them (Fig. 2b-d) are not
  covered by CC BY 4.0 (`LICENSE-data.md`).
* `data/external/golkar_deoliveira_2026/`: figures from GO26, CC BY 4.0, reproduced
  unchanged for re-analysis with attribution.
* Benchmark flux data: Volk et al. (2023c), doi:10.5281/zenodo.7636781, CC BY 4.0.
* OpenET model data: Volk et al. (2023d), doi:10.5281/zenodo.10119477, CC BY 4.0.
* AmeriFlux: BASE data and the public metadata used by `scripts/11` (amerifluxr's
  variable-availability summary, AmeriFlux web services and site pages) are CC BY 4.0
  under the AmeriFlux data policy. BASE files are not redistributed; cite each site's BASE
  DOI (`config/go26_sites.csv`) and acknowledge the U.S. Department of Energy Office of
  Science, which funds the AmeriFlux data portal.
* Koppen-Geiger grid: Rubel et al. (2017), via the `kgcpy` package.
* State boundaries: U.S. Census Bureau county layer, via `basemap-data`.

## Use of AI tools

The analysis code, figures and the first draft of the comment were prepared with the help
of Claude (Anthropic). The authors reviewed the code and its results and take
responsibility for them.

## Citation

Cite the archive (DOI 10.5281/zenodo.23383493) as given in `CITATION.cff`; on GitHub, use
"Cite this repository". Please also cite the comment once published, GO26, the works below
and the AmeriFlux site DOIs of any BASE data you use.

* Melton, F.S., et al., 2022. OpenET: filling a critical data gap in water management for
  the western United States. J. Am. Water Resour. Assoc. 58, 971-994.
  https://doi.org/10.1111/1752-1688.12956
* Rubel, F., Brugger, K., Haslinger, K., Auer, I., 2017. The climate of the European Alps:
  shift of very high resolution Köppen-Geiger climate zones 1800-2100. Meteorol. Z. 26,
  115-125. https://doi.org/10.1127/metz/2016/0816
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
