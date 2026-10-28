# Data

Only this README in `data/` is tracked in git. `raw/` is recreated by the scripts and is
not deposited, except `raw/ameriflux_public/` (cached public AmeriFlux metadata that change
over time); `external/golkar_deoliveira_2026/` is also in the Zenodo deposit
(https://doi.org/10.5281/zenodo.23383493).

## raw/ (downloaded; not redistributed)

* `flux_ET_dataset.zip` -> `flux_ET_dataset/`: Volk et al. (2023c), Zenodo 7636781,
  md5 99f6668ca439e9bae48c4e7e08b5405d, CC BY 4.0.
* `OpenET_PhaseII_model_ET_dataset.zip` -> `OpenET_PhaseII_model_ET_dataset/`: Volk et al.
  (2023d), Zenodo 10119477, md5 26b2af882c21dc56c634ac9f77a2ba6e, CC BY 4.0.

  `scripts/01_download_data.py` uses a zip already in `raw/` or `external/` whose md5
  matches the published sum, whatever its name (for example
  `external/flux_ET_dataset_Volk_2023.zip`), and otherwise downloads it from Zenodo into
  `raw/`; verified zips are extracted to `raw/`. `--verify` checks and extracts without
  downloading.
* `openet_api/`: OpenET API extracts (collection 2.1; 200 m x 200 m boxes centred on the
  towers, as GO26) written by `scripts/02_download_openet_api.py`, which needs an OpenET
  API key:
  - `sites.csv`, `sites_200m.geojson`: the 75 boxes (22 GO26 towers and 60 benchmark
    cropland stations; 7 are in both).
  - `<interval>_<variable>_<model>[_<SITE>]_<start>_<end>.csv`: one raw CSV per request.
    Months or days without data are absent rows, not zeros.
  - `openet_monthly.csv`: site, date (month start), `et_<model>` for the ensemble and the
    six models, `eto`, `pr`; mm/month, 2016-01 to 2025-12, all 75 boxes.
  - `openet_daily.csv`: site, date, `et_ensemble`, `eto`, `pr`; mm/d, 2016-01-01 to
    2025-07-31, at US-UiC, US-UTJ and the seven towers shared with the benchmark (ETo and
    precipitation at US-UiC and US-UTJ only).
  - `openet_daily_go26.csv`: site, date, `et_ensemble`; mm/d, 2016-01-01 to 2025-07-31, at
    all 22 GO26 towers (the nine above and 13 extracted on 28 October 2026, 14:49-15:18
    UTC), read by `scripts/10` for the daily check of Figure 2 of the comment. Kept apart
    so that `openet_daily.csv`, which `scripts/12` reads, is unchanged.
  - `manifest.json`: request log (endpoint, parameters including the collection version,
    tag, site or sites, rows, status and UTC time; upload entries name the file; requests
    made after the 15 October 2026 extraction also log the md5 of the boxes' GeoJSON,
    `geojson_md5`). The API key is never written.
  - `derived/` (OpenET values, not deposited), written by
    `scripts/12_openet_api_analysis.py`: `go26_sites_full.csv` (the full per-tower table,
    with the zero-fill predictions and daily-based means), `go26_figure_check_all.csv` and
    `go26_figure_check_monthly.csv` (GO26's plotted OpenET line against the API daily
    ensemble, by day set and by month), `benchmark_station_months.csv` (benchmark
    station-months: measured closed ET and OpenET minus measured, ensemble and six models)
    and `uic_winter_openet.csv` (the US-UiC winter comparison with OpenET, ETo and
    precipitation); by `scripts/10_reconstruct_go26_observed_et.py`:
    `go26_openet_sites_daily_check.csv` (the per-tower table of Figure 2 with the daily
    ensemble over the same days, `openet_daily_mm_d`); and by
    `scripts/14_check_openet_disclosure.py`: `disclosure_check.json`
    (the record of the joint OpenET disclosure check, which `scripts/13` requires before it
    builds the deposit).

  The extracts used for the published numbers were made on 15 October 2026, 03:38-04:09
  UTC. OpenET revises its collections, so a new extraction can differ slightly; if the md5
  sums of `openet_monthly.csv`, `openet_daily.csv`, `sites.csv` and `sites_200m.geojson`
  match those recorded in `results/openet_api_go26_summary.json` (`provenance`),
  `results/openet_api_*` (except `openet_api_eto_ee_check.csv`, which queries the live Earth
  Engine collection) and `figures/fig2_*` are reproduced exactly.

  ETo is OpenET's bias-corrected gridMET ETo (Volk et al., 2026); precipitation is
  gridMET. OpenET data are subject to the OpenET Terms of Service
  (https://etdata.org/terms-of-service/); attribute "OpenET, https://openetdata.org".
* `ameriflux_public/`: public AmeriFlux metadata cached by
  `scripts/11_ameriflux_availability.py` (no login; fetched on 15 October 2026,
  04:26-04:29 UTC; in the Zenodo deposit):
  - `AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_<sha>.csv`: amerifluxr's BASE
    variable-availability summary at the pinned commit (7319e632).
  - `api_v1_data_availability_AmeriFlux_BASE-BADM_CCBY4.0.json`,
    `api_v1_site_display_AmeriFlux.json`: AmeriFlux web-service responses.
  - `siteinfo_<SITE>.html`: site pages of the 22 GO26 towers (current BASE version and
    years available), with every e-mail address of the site teams, including `mailto:`
    links, replaced by `[email removed]` before the page is cached.
  - `provenance.json`: URL, access time and md5 of every cached file; for the site pages
    also the md5 and size of the page as served (`md5_original`, `bytes_original`; the
    `page_md5` in `results/ameriflux_public_provenance.json`) and the number of addresses
    removed (`emails_removed`).

  The metadata (the summary and the web-service responses) are CC BY 4.0 under the
  AmeriFlux data policy. The site pages are AmeriFlux website content (© LBNL / UC Regents;
  the photos on them carry their own terms), cached for provenance only; they are not
  covered by this repository's licences.
* `ameriflux_base/`: AmeriFlux BASE files of the 22 GO26 towers, read only by
  `scripts/10_reconstruct_go26_observed_et.py` (not deposited):
  - `AMF_<SITE>_BASE-BADM_<version>.zip`: the BASE-BADM product as downloaded (BASE CSV
    and BADM site metadata).
  - `AMF_<SITE>_BASE_<HH|HR>_<version>.csv`: the BASE CSV from the zip (hourly, HR, at
    US-Ne1, US-Ne2 and US-Ne3).
  - `provenance.json`: per site the version, URL, md5 and size of the zip, md5 of the CSV,
    access time (UTC), and the intended use and description sent with the request.

  The files behind `results/go26_*` are the versions GO26 cite at all 22 sites. They were
  downloaded on 15 October 2026, 08:02:54-08:03:22 UTC, in one request with the intended use
  "Research - Remote sensing" and the description "Re-analysis of the observed ET used by
  Golkar and de Oliveira (2026, Agric. For. Meteorol. 391, 111524) and evaluation of OpenET
  at AmeriFlux cropland sites", which AmeriFlux sends to the site teams. To get them you need
  an AmeriFlux account: download the versions in `config/go26_sites.csv` (column
  `base_version_go26`) from https://ameriflux.lbl.gov and extract the BASE CSV of each
  `AMF_<SITE>_BASE-BADM_<version>.zip` into this folder (script 10 reads only the CSV files),
  or run script 10 with `--download
  --agree-policy` and `AMERIFLUX_USER_ID` and `AMERIFLUX_USER_EMAIL` set in the environment
  or in `.env` (it fetches the current versions; AmeriFlux logs each download). CC BY 4.0
  under the AmeriFlux CC-BY-4.0 data policy, but not redistributed here; cite each site's
  BASE DOI (`config/go26_sites.csv`, column `base_doi`) and acknowledge the sites and the
  funding of AmeriFlux data resources as AmeriFlux recommends: "We acknowledge the following
  AmeriFlux sites for their data records: [the sites used]. In addition, funding for
  AmeriFlux data resources was provided by the U.S. Department of Energy's Office of
  Science."

## external/

* `golkar_deoliveira_2026/`: GO26 Fig. S2 (supplementary file mmc2, large version),
  Fig. 8 and Fig. 2, extracted unchanged from the CC BY 4.0 article. Figs. S2 and 8 are
  digitized by `scripts/04_digitize_afm_figures.py`; Fig. 2 is kept for the visual check
  of the zero stripe in its scatter plot. In the Zenodo deposit.
* Local copies of the two Zenodo zips may be kept here; `scripts/01` finds them by md5.
  They are not deposited (they are on Zenodo under their own DOIs).
* Anything else placed here (for example the article PDFs) is not read by the scripts and
  is neither tracked nor deposited.
