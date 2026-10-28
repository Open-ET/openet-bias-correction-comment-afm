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
  unchanged for re-analysis. The series digitized from Figs. 8 and S2
  (`results/digitized_*`, CC BY 4.0) include GO26's plotted OpenET line: attribute
  GO26 and "OpenET, https://openetdata.org".
* Volk et al. (2023c) flux data, https://doi.org/10.5281/zenodo.7636781, CC BY 4.0
  (downloaded by `scripts/01_download_data.py`, not stored in the repository). The
  Bokeh pages in `benchmark_site_plots/html/` are the benchmark's own graphical files,
  copied unchanged (CC BY 4.0, Volk et al., 2023c); the summaries in
  `benchmark_site_plots/static/` are this work (CC BY 4.0, above). In the repository
  both are in `figures/site_plots/` (`scripts/03_site_plots.py`).
* Volk et al. (2023d) OpenET model data, https://doi.org/10.5281/zenodo.10119477,
  CC BY 4.0 (downloaded, not stored).
* OpenET API data (collection 2.1, extracted on 15 October 2026, 03:38-04:09 UTC, by
  `scripts/02_download_openet_api.py`; daily ensemble ET at 13 further GO26 towers on 28
  October 2026). The statistics in `results/openet_api_*`,
  `results/go26_reconstructed_openet_bias.csv` and
  `results/go26_reconstructed_openet_sites.csv`, and the figures that show them (Fig. 2b-d,
  `figures/fig2_zero_fill_bias_and_site_ET.*`; the OpenET values of Figure 2 of the comment,
  `figures/fig_openet_reconstruction.*`), are derived
  from OpenET data: attribute "OpenET, https://openetdata.org". They are subject to the
  OpenET Terms of Service (https://etdata.org/terms-of-service/) and are not covered by
  the CC BY 4.0 license above. The raw extracts and the tables with OpenET values
  (`data/raw/openet_api/`, including `derived/`) are not redistributed. The OpenET values in
  `results/` are statistics over study periods, seasons, climate classes and stations,
  apart from GO26's plotted OpenET line in `results/digitized_*` (above): before
  writing, `scripts/12` and `scripts/10` each check that no single monthly or daily OpenET
  value, and no sum or difference over fewer than six site-months, can be solved for from
  their outputs and the public data, and they withhold (leave blank) what would allow it.
* AmeriFlux public metadata (`data/raw/ameriflux_public/`, fetched on 15 October 2026,
  04:26-04:29 UTC by `scripts/11_ameriflux_availability.py`): AmeriFlux content. The
  metadata (the BASE variable-availability summary distributed with amerifluxr,
  https://github.com/chuhousen/amerifluxr, commit 7319e632, AmeriFlux web-service
  responses and site pages) are CC BY 4.0 under the AmeriFlux data policy; the site pages
  are included for provenance, with the site teams' e-mail addresses removed. The LE
  availability derived from them (`results/ameriflux_le_availability*.csv`,
  `results/ameriflux_public_provenance.json`) is CC BY 4.0 (attribute AmeriFlux and the
  site teams).
* AmeriFlux BASE data (downloaded on 15 October 2026, 08:02:54-08:03:22 UTC; the versions
  GO26 cite at all 22 sites), CC BY 4.0 under the AmeriFlux CC-BY-4.0 data policy; not
  redistributed. The files rebuilt from them by `scripts/10_reconstruct_go26_observed_et.py`
  are CC BY 4.0 AmeriFlux derivatives: `results/go26_observed_et_reconstruction.csv`,
  `results/go26_reconstructed_daily.csv`, `results/go26_reconstructed_winter.csv`,
  `results/go26_base_version_check.csv` (with the AmeriFlux availability summary above) and
  `results/go26_reconstructed_figure_check.csv` (with GO26's digitized Figs. 8 and S2).
  Cite each site's BASE DOI (`config/go26_sites.csv`; also in the `base_doi` column of
  `go26_observed_et_reconstruction.csv` and the header lines of
  `go26_reconstructed_daily.csv`). `results/go26_reconstructed_openet_bias.csv` and
  `results/go26_reconstructed_openet_sites.csv` hold OpenET-derived statistics (OpenET Terms
  of Service, above) next to the observed ET (CC BY 4.0, as above).

Acknowledgement asked for by the AmeriFlux CC-BY-4.0 data policy, for any use of the
BASE data or their derivatives above (the wording AmeriFlux recommends):
"We acknowledge the following AmeriFlux sites for their data records: [the sites used]. In
addition, funding for AmeriFlux data resources was provided by the U.S. Department of
Energy's Office of Science."
