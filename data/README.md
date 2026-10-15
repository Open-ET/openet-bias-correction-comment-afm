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
  - `manifest.json`: request log (endpoint, file, status, UTC time). The API key is never
    written.

  ETo is OpenET's bias-corrected gridMET ETo (Volk et al., 2026); precipitation is
  gridMET. OpenET data are subject to the OpenET Terms of Service
  (https://etdata.org/terms-of-service/); attribute "OpenET, https://openetdata.org".
* `ameriflux_public/`: public AmeriFlux metadata cached by
  `scripts/11_ameriflux_availability.py` (no login):
  - `AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_<sha>.csv`: amerifluxr's BASE
    variable-availability summary at the pinned commit (7319e632).
  - `api_v1_data_availability_AmeriFlux_BASE-BADM_CCBY4.0.json`,
    `api_v1_site_display_AmeriFlux.json`: AmeriFlux web-service responses.
  - `siteinfo_<SITE>.html`: site pages of the 22 GO26 towers (current BASE version and
    years available).
  - `provenance.json`: URL, access time and md5 of every cached file.
* `ameriflux_base/` (optional): `AMF_<SITE>_BASE_<HH|HR>_<version>.csv` for the 22 GO26
  towers, used only by `scripts/10_reconstruct_go26_observed_et.py`. Needs an AmeriFlux
  account: download the files from https://ameriflux.lbl.gov (the versions GO26 cite are
  in `config/go26_sites.csv`, column `base_version_go26`), or run script 10 with
  `--download --agree-policy` and `AMERIFLUX_USER_ID` and `AMERIFLUX_USER_EMAIL` set in the
  environment or in `.env` (AmeriFlux logs each download and reports it to the site
  teams). Provenance goes to `ameriflux_base/provenance.json`. CC BY 4.0; cite each site's
  BASE DOI.

## external/

* `golkar_deoliveira_2026/`: GO26 Fig. S2 (supplementary file mmc2, large version),
  Fig. 8 and Fig. 2, extracted unchanged from the CC BY 4.0 article. Figs. S2 and 8 are
  digitized by `scripts/04_digitize_afm_figures.py`; Fig. 2 is kept for the visual check
  of the zero stripe in its scatter plot. In the Zenodo deposit.
* Local copies of the two Zenodo zips may be kept here; `scripts/01` finds them by md5.
  They are not deposited (they are on Zenodo under their own DOIs).
* Anything else placed here (for example the article PDFs) is not read by the scripts and
  is neither tracked nor deposited.
