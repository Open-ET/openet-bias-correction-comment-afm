# Data and figure license

Derived data in `results/`, figures in `figures/` and `benchmark_site_plots/static/`,
and the configuration tables in `config/` are released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/), except the values derived from the
OpenET API (see below). `config/` holds values transcribed from GO26 (Tables 1 and 3)
and the Volk et al. (2024) station list, both CC BY 4.0.

Third-party material keeps its own license and attribution:

* `data/external/golkar_deoliveira_2026/`: Figs. 2, 8 and S2 of Golkar, M.K.,
  de Oliveira, L.A., 2026, Agric. For. Meteorol. 391, 111524,
  https://doi.org/10.1016/j.agrformet.2026.111524, CC BY 4.0. Reproduced
  unchanged for re-analysis.
* Volk et al. (2023c) flux data, https://doi.org/10.5281/zenodo.7636781, CC BY 4.0
  (downloaded by `scripts/01_download_data.py`, not stored in the repository). The
  Bokeh pages in `benchmark_site_plots/html/` are the benchmark's own graphical files,
  copied unchanged (CC BY 4.0).
* Volk et al. (2023d) OpenET model data, https://doi.org/10.5281/zenodo.10119477,
  CC BY 4.0 (downloaded, not stored).
* OpenET API data (collection 2.1, accessed 15 October 2026 UTC by
  `scripts/02_download_openet_api.py`). The statistics in `results/openet_api_*`
  and `results/go26_reconstructed_openet_bias.csv`, and the figure panels that
  show them (Fig. 2b-d, `figures/fig2_zero_fill_bias_and_site_ET.*`), are derived
  from OpenET data: attribute "OpenET, https://openetdata.org". They are subject to the OpenET Terms of
  Service (https://etdata.org/terms-of-service/) and are not covered by the
  CC BY 4.0 license above. The raw extracts (`data/raw/openet_api/`, including the
  station-month comparison table and the US-UiC winter comparison in `derived/`) are not
  redistributed. `results/` holds statistics over study periods, seasons, climate classes
  and stations, and monthly means of the API daily ensemble only for GO26's two plotted
  series (US-UiC 2020, US-UTJ 2024); `scripts/12` withholds small stations and checks that
  no single monthly or daily OpenET value can be solved for from these files. Rows of
  `results/go26_reconstructed_openet_bias.csv` that would allow this are left blank.
* AmeriFlux LE availability (`results/ameriflux_le_availability*.csv`,
  `results/ameriflux_public_provenance.json`, from `scripts/11_ameriflux_availability.py`):
  derived from the AmeriFlux BASE variable-availability summary distributed with
  amerifluxr (https://github.com/chuhousen/amerifluxr, commit 7319e632) and from
  AmeriFlux site metadata; CC BY 4.0 via AmeriFlux (attribute AmeriFlux and the
  site teams). The cached AmeriFlux web-service responses and site pages in
  `data/raw/ameriflux_public/` are AmeriFlux content, included for provenance.
* AmeriFlux BASE data, CC BY 4.0 under the AmeriFlux data policy; cite each
  site's DOI (config/go26_sites.csv). Not redistributed. The daily series rebuilt
  from them by `scripts/10_reconstruct_go26_observed_et.py`
  (`results/go26_reconstructed_daily.csv`) lists the site DOIs in its header lines.
