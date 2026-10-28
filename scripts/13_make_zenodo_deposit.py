"""Build the Zenodo deposit (doi:10.5281/zenodo.23735189) in zenodo/.

The deposit holds a snapshot of the code and the inputs and outputs that git
does not track, one zip per folder with paths relative to the repository root
(unzip them there), plus README.md (the deposit description, with sizes and
the md5 sums of the data archives), description.html (README.md as HTML for
the Description field of the Zenodo record) and MD5SUMS (the data archives;
the code snapshot is not listed):

  openet-bias-correction-comment-afm-<version>.zip   the files git tracks: code, tests, docs, metadata
  config.zip                                         GO26 Tables 1 and 3, site notes, V24 stations
  data_external_golkar_deoliveira_2026.zip           GO26 Figs. 2, 8 and S2 (CC BY 4.0)
  data_raw_ameriflux_public.zip                      cached public AmeriFlux metadata (scripts/11)
  results.zip                                        derived results, with results/README.txt, a
                                                     guide to every file (FILE_NOTES below)
  figures.zip                                        Figures 1 and 2 of the comment, further figures, QC
                                                     overlays
  benchmark_site_plots.zip                           benchmark plots of the seven shared towers:
                                                     figures/site_plots/html/ and static/ (scripts/03)
                                                     as benchmark_site_plots/html/ and static/, with
                                                     benchmark_site_plots/README.txt

figures.zip leaves out figures/site_plots/ (html/, static/ and the Playwright
snapshots in png_bokeh/) and figures/.gitkeep.  Not deposited: the other
downloads in data/raw/ (the Volk et al. data are on Zenodo under their own
DOIs; the OpenET extracts need an API key and stay local under the OpenET Terms
of Service; AmeriFlux BASE files are not redistributed), docs/internal/,
paper/ and .env.

The inputs of every archive are checked first: folders present, the figures,
QC overlays and site plots complete, a FILE_NOTES entry for every results file
(and every non-optional entry present), no e-mail addresses in the text files
of the data archives, and the joint OpenET disclosure check of scripts/14
passed on the current files (its record, DISCLOSURE_RECORD, complete, passed,
covering every OpenET-derived results file, every file it read unchanged;
--no-disclosure-check skips this, e.g. to rebuild from an unzipped deposit
without the OpenET extracts).  A problem stops the script before anything is written.
The archives are then built in a temporary folder next to --out and moved into
place at the end, so a failure leaves an existing deposit as it was; zips of an
earlier build that the new one does not replace (another version) are deleted
then, other files in --out are kept.

The zips are deterministic (entries sorted by name, fixed timestamps,
permissions and creator fields, no .DS_Store or __pycache__): a rebuild from
unchanged files with the same Python and zlib versions (the README records both;
environment.yml pins Python, not zlib) gives the same md5 sums, since zlib's
deflate output can differ between versions.  Run it after the pipeline
(README.md, Pipeline).  In an unzipped deposit the site plots are read from
benchmark_site_plots/ when figures/site_plots/ is absent.

Usage:
  python scripts/13_make_zenodo_deposit.py [--out zenodo] [--version 1.1.0] [--no-disclosure-check]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import textwrap
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "openet-bias-correction-comment-afm"
DOI = "10.5281/zenodo.23735189"
PREPRINT_DOI = "10.31223/X5B51M"  # the comment on EarthArXiv (posted 21 October 2026)
REPO_URL = f"https://github.com/Open-ET/{NAME}"
COMMENT = ("Comment on Golkar and de Oliveira (2026): Missing eddy covariance observations undermine reported "
           "OpenET biases")
TITLE = f"Code and data for: {COMMENT}"
STAMP = (2026, 10, 15, 0, 0, 0)  # release date, used as every entry's timestamp
SKIP = {".DS_Store", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
AFFILIATION = "Desert Research Institute, Reno, Nevada, USA"
# author line without CITATION.cff (with it, the line, and any ORCID iDs, come from CITATION.cff)
AUTHORS = "Yeonuk Kim, Sayantan Majumdar, John M. Volk, Peter ReVelle, Todd G. Caldwell, Justin L. Huntington"
# archive name -> (folder in the archive, relative to ROOT; contents; licence)
FOLDERS = {
    "config.zip": ("config", "GO26 Table 1 (sites, periods, mean observed ET) with the BASE DOIs and versions from "
                             "GO26's reference list, GO26 Table 3 (BWk from its Sections 3.1-3.2), our site notes "
                             "(sources in the evidence column), and the 152 stations of Volk et al. (2024) from "
                             "Station_metadata.xlsx of Volk et al. (2023d)",
                   "CC BY 4.0 (GO26; Volk et al., 2023d; site notes: this work)"),
    "data_external_golkar_deoliveira_2026.zip": (
        "data/external/golkar_deoliveira_2026", "GO26 Figs. S2 and 8 (digitized by scripts/04) and Fig. 2, "
                                                "unchanged from the article", "CC BY 4.0 (Golkar and de Oliveira, 2026)"),
    "data_raw_ameriflux_public.zip": (
        "data/raw/ameriflux_public", "cached public AmeriFlux metadata read by scripts/11: BASE variable-availability "
                                     "summary (amerifluxr, commit 7319e63), web-service responses and site pages "
                                     "(e-mail addresses removed), with access times and md5 sums",
        "metadata (summary, web-service responses): CC BY 4.0 (AmeriFlux data policy); site pages: AmeriFlux "
        "website content (© LBNL / UC Regents), cached for provenance only, not licensed by this deposit"),
    "results.zip": ("results", "derived tables (CSV, JSON) of scripts 02 (Earth Engine ETo check) and 04-12, with "
                               "README.txt, a one-line guide to every file",
                    "CC BY 4.0, except OpenET-derived statistics (OpenET Terms of Service)"),
    "figures.zip": ("figures", "Figures 1 and 2 of the comment (fig_table1_reconstruction: GO26 Table 1 against "
                               "the site means rebuilt from BASE; fig_openet_reconstruction: OpenET against them over "
                               "the same days; scripts/08), further figures of the analysis (fig1-fig5) "
                               "(PNG 600 dpi, PDF), and QC overlays of the digitization",
                    "CC BY 4.0, except panels b-d of fig2 and the OpenET values of Figure 2 of the comment "
                    "(OpenET-derived: OpenET Terms of Service); Figures 1 and 2 of the comment and the observed ET "
                    "of fig2 panel d derive from AmeriFlux BASE data (cite the site BASE DOIs)"),
    "benchmark_site_plots.zip": (
        "benchmark_site_plots", "plots of the seven towers shared with GO26 from the Volk et al. (2023c) benchmark, "
                                "built from `figures/site_plots/` (scripts/03): the benchmark's Bokeh pages (`html/`) "
                                "and static summaries (`static/`), with README.txt",
        "CC BY 4.0 (html/: Volk et al., 2023c; static/: this work)"),
}

FIGURES = ("fig1_US-UTJ_zero_runs", "fig2_zero_fill_bias_and_site_ET", "fig3_site_map", "fig4_shared_site_daily",
           "fig5_shared_sites", "fig_table1_reconstruction",
           "fig_openet_reconstruction")  # the last two are Figures 1 and 2 of the comment
QC = ("figures/qc/digitized_US-UTJ_overlay.png", "figures/qc/digitized_US-UiC_overlay.png")
LEFT_OUT_OF_FIGURES = ("figures/site_plots/", "figures/.gitkeep")  # site plots: benchmark_site_plots.zip
SITE_PLOTS = "figures/site_plots"  # written by scripts/03_site_plots.py
SITE_PLOT_PARTS = ("html", "static")  # deposited as benchmark_site_plots/html/ and static/
SHARED = ("US-ARM", "US-Bi1", "US-Bi2", "US-Ne1", "US-Ne2", "US-Ne3", "US-xSL")
TEXT_SUFFIXES = {".csv", ".json", ".geojson", ".html", ".htm", ".txt", ".md", ".xml"}

BENCHMARK_README = """\
Site-specific plots from the Volk et al. (2023a) benchmark for the seven towers shared with
Golkar & de Oliveira (2026): US-ARM, US-Bi1, US-Bi2, US-Ne1, US-Ne2, US-Ne3, US-xSL.

html/    the benchmark's interactive Bokeh pages (graphical_files/<SITE>_plots.html),
         unchanged. They load BokehJS 2.1.1 from cdn.bokeh.org, so open them online.
static/  summaries from the daily data, 6.5 in wide (PNG, 600 dpi, and PDF): daily ET with
         gap-filled days and OpenET overpass values; daily closure ratio; monthly OpenET vs
         closed EC ET in months with <= 5 gap-filled days; LE + H vs Rn - G.

Full-page PNG snapshots of the Bokeh pages are not included; scripts/03_site_plots.py renders
them with Playwright (figures/site_plots/png_bokeh/).

Source data: Volk et al. (2023c), doi:10.5281/zenodo.7636781, and Volk et al. (2023d; the
OpenET values), doi:10.5281/zenodo.10119477, both CC BY 4.0. scripts/03_site_plots.py writes
html/ and static/ to figures/site_plots/; scripts/13_make_zenodo_deposit.py copies them into
this folder of the deposit and writes this README.

Licence: CC BY 4.0. html/: Volk et al. (2023c), the benchmark's own graphical files;
static/: this work (doi:10.5281/zenodo.23735189).
"""

# results/README.txt: one line per file of results/ (scripts/13 stops when a file has no entry)
FILE_NOTES = {
    "ameriflux_le_availability.csv":
        "scripts/11: one row per GO26 tower: study period, GO26 Table 1 mean, BASE versions (GO26's and current), "
        "publish years, LE variables, first LE year, LE availability over the period (le_avail_period, from "
        "amerifluxr's BASE variable-availability summary) and GO26 mean / availability (implied_mean_valid_mm_d)",
    "ameriflux_le_availability_by_year.csv":
        "scripts/11: site x year of each GO26 period: LE availability of the year, days in the year and in the "
        "period, valid-record days (capped at the period's days)",
    "ameriflux_public_provenance.json":
        "scripts/11: URL, access time (UTC) and md5 of every AmeriFlux file used (amerifluxr summary at commit "
        "7319e63, web services, site pages; page_md5 is the md5 of the page as served)",
    "benchmark_bowen_ratio_vs_ebr.csv":
        "scripts/05: per benchmark eddy covariance station: daily Bowen-ratio closed minus EBR-closed ET (median, "
        "mean, percentiles) and the shares of days with a negative, near-singular or out-of-range (0.5-2) "
        "Bowen-ratio factor",
    "benchmark_openet_bias_cropland.csv":
        "scripts/05: per benchmark cropland station: OpenET Phase II ensemble (Volk et al., 2023d) MBE and RMSE "
        "against EBR-closed, unclosed and Bowen-ratio closed ET on overpass days; monthly MBE in months with <= 5 "
        "gap-filled days",
    "benchmark_zero_fill_shift_cropland.csv":
        "scripts/05: per cropland station: share f of non-measured days in the calendar-year record, the "
        "benchmark's gap-fill ET on them (E_gapfill_mm_d) and the bias shift f x E that counting them as zero adds",
    "benchmark_zero_survey.csv":
        "scripts/05: per benchmark station (161): measured days, exact-zero days and days within 0.015 mm/d of "
        "zero of closed ET, negative days and the longest run of near-zero days",
    "digitized_US-UTJ.csv":
        "scripts/04: daily observed, OpenET and XGBoost ET digitized from GO26 Fig. S2 (US-UTJ, 2024), with flags "
        "for interpolated values and exact-zero observations; the openet column is GO26's plotted OpenET line "
        "(OpenET, https://openetdata.org)",
    "digitized_US-UiC.csv":
        "scripts/04: daily observed, OpenET and XGBoost ET digitized from GO26 Fig. 8 (US-UiC, 2020), same columns; "
        "the openet column is GO26's plotted OpenET line (OpenET, https://openetdata.org)",
    "digitized_summary.json":
        "scripts/04: per digitized site: calibration, zero days and runs, means, OpenET and XGBoost RMSE, MBE and "
        "R2 with and without the exact-zero days, the statistics printed on GO26's figure, and the sensitivity of "
        "the RMSE improvement to dots hidden under the plotted lines and to a doubled zero tolerance",
    "go26_base_version_check.csv":
        "scripts/10: per GO26 tower and year of the study period: share of records with LE in the cited BASE file "
        "and in AmeriFlux's variable-availability summary of October 2025, and in-period LE records of each; per "
        "tower: whether the cited version is newer than the summary, the month in which the summary's (earlier) "
        "version ends its LE against GO26's period end, and the cited file's last LE month (AmeriFlux BASE "
        "derivative)",
    "go26_observed_et_reconstruction.csv":
        "scripts/10: per GO26 tower: BASE file, version and DOI, columns used (one net radiometer; the other in "
        "*_netrad_alt), first and last valid LE and last record, valid and energy-balance days, site means of the "
        "zero-filled sums (over calendar days and over the days in the file), of valid-day ET (mean of valid "
        "records, and gap-filled by the mean diurnal variation, *_mdv), of Bowen-ratio closed ET as GO26 state it "
        "(flux-data-qaqc's 'br' correction; unfilled *_br_fdq, gap-filled *_br_fdq_mdv), of EBR (its 'ebr' "
        "correction) and of screened and unscreened Bowen-ratio closed ET, "
        "Bowen-ratio failure counts and extreme days, differences from GO26 Table 1 (diff_* = rebuilt value minus "
        "GO26's Table 1 mean, mm/d, the opposite sign to mean_diff_go26_minus_base in "
        "go26_reconstructed_figure_check.csv), implied zero fractions (f_implied, f_implied_unfilled = 1 - Table 1 "
        "mean / the gap-filled or unfilled valid-day mean); mean_et_hh_sum_zero_filled_unclosed is the Table 1 "
        "reproduction (zero-filled sub-daily sums over calendar days), mean_et_unclosed_zero_filled the valid-day ET "
        "with the other days set to 0; *_qc: LE flagged 2 in LE_SSITC_TEST removed (AmeriFlux BASE derivative)",
    "go26_reconstructed_daily.csv":
        "scripts/10: per tower and day: shares of records with LE and with all energy-balance terms, valid-day ET "
        "(unfilled and gap-filled), Bowen-ratio, screened and EBR closure factors, Bowen-ratio closed ET as GO26 "
        "state it (et_br_fdq; gap-filled et_br_fdq_mdv), EBR, screened and unscreened Bowen-ratio closed ET, "
        "zero-filled sums, flags (*_qc "
        "variants); '#' header with definitions and the site "
        "BASE DOIs (AmeriFlux BASE derivative)",
    "go26_reconstructed_figure_check.csv":
        "scripts/10: GO26's plotted daily observed ET (Figs. 8 and S2, digitized) against the BASE zero-filled daily "
        "sums: days compared, mean and largest differences, days within 0.02 and 0.1 mm/d, plotted days with fewer "
        "than half of the records or without LE, exact-zero days (AmeriFlux BASE derivative)",
    "go26_reconstructed_openet_bias.csv":
        "scripts/10: OpenET ensemble minus observed ET rebuilt from BASE by GO26 zone and season: zero-filled Table 1 "
        "sums over all site-months with OpenET (next to GO26's reported bias) and, on identical site-months (>= 20 "
        "days on which every valid-day variant has a value: all energy-balance terms, the day kept by flux-data-qaqc's "
        "Bowen-ratio limits and an EBR factor), zero-filled sums over all days of those months and, on identical "
        "days, GO26's processing as they state it (valid-day and Bowen-ratio closed ET), and gap-filled ET unclosed "
        "and closed with the Bowen ratio as GO26 state it, EBR or the screened Bowen ratio; OpenET values "
        "withheld where a single month or any sum or difference over fewer than 6 site-months could be solved for "
        "(OpenET-derived; observed ET: AmeriFlux BASE derivative)",
    "go26_reconstructed_openet_sites.csv":
        "scripts/10: per GO26 tower, OpenET ensemble and observed ET rebuilt from BASE averaged over the same days "
        "(Figure 2 of the comment): every day of the study period against the zero-filled Table 1 sums, and the days "
        "on which every valid-day variant has a value (as the common valid months of "
        "go26_reconstructed_openet_bias.csv) against those variants; OpenET is the monthly total weighted by the "
        "days in each month (OpenET-derived; observed ET: AmeriFlux BASE derivative)",
    "go26_reconstructed_winter.csv":
        "scripts/10: Dec-Feb means per tower (and January, February and January-February 2020 at US-UiC) of "
        "unclosed (unfilled and gap-filled), Bowen-ratio closed as GO26 state it, EBR, screened and unscreened "
        "Bowen-ratio closed and zero-filled ET from "
        "BASE and Bowen-ratio failure counts, with GO26 Fig. 8 (AmeriFlux BASE derivative)",
    "leakage_experiment.csv":
        "scripts/07: XGBoost correction of OpenET Phase II at 38 benchmark cropland stations: RMSE, MBE and gain "
        "over OpenET per scenario (clean, zero-filled + median-imputed), seed and validation design (random split, "
        "site-grouped, leave-one-site-out, leave-one-cluster-out)",
    "leakage_experiment_clusters.csv":
        "scripts/07: RMSE per multi-station cluster (stations within 10 km) under leave-one-site-out and "
        "leave-one-cluster-out",
    "leakage_experiment_met_unfilled.csv":
        "scripts/07: as leakage_experiment.csv, with missing tower meteorology left unfilled (sensitivity)",
    "leakage_experiment_met_unfilled_summary.csv":
        "scripts/07: mean, min and max over the seeds of leakage_experiment_met_unfilled.csv",
    "leakage_experiment_summary.csv":
        "scripts/07: mean, min and max over the seeds of leakage_experiment.csv",
    "leakage_experiment_versions.json":
        "scripts/07: Python and package versions, seeds, model settings, zero-filled sites, meteorology gaps, "
        "clusters and input md5 sums",
    "openet_api_benchmark_summary.csv":
        "scripts/12: current OpenET (API, collection 2.1; ensemble and six models) against the benchmark's closed "
        "ET in complete months at the cropland stations, by stratum (all, Koppen class and group, season, season x "
        "group, site), with GO26's zone and season biases; MBE per day only for the ensemble's pooled strata and "
        "each model's all-months row; small stations withheld (openet_withheld) (OpenET-derived)",
    "openet_api_eto_ee_check.csv":
        "scripts/02 --verify-eto-ee: monthly ETo of the OpenET API against OpenET's bias-corrected gridMET "
        "collection on Earth Engine at the 75 boxes, differences in mm/month (OpenET-derived)",
    "openet_api_go26_figure_check.csv":
        "scripts/12: GO26's plotted OpenET line (digitized Figs. 8 and S2) against the API daily ensemble over all "
        "plotted days of each figure (US-UiC 2020, US-UTJ 2024): days, means, difference, RMSE, r and least-squares "
        "scale (OpenET-derived; nothing by month)",
    "openet_api_go26_sites.csv":
        "scripts/12: per GO26 tower: OpenET over the study period (ensemble and models; mm/d, mm/yr), "
        "precipitation, ETo, GO26 Table 1 mean and ratios, LE availability (per-tower zero-fill predictions not "
        "deposited) (OpenET-derived)",
    "openet_api_go26_summary.json":
        "scripts/12: key numbers of the OpenET analysis: cross-tower statistics, the figure check over all plotted "
        "days, benchmark, shared-tower and US-UiC winter summaries, identifiability checks, definitions and "
        "provenance (collection, access times, md5 of the raw extracts) (OpenET-derived)",
    "openet_api_shared_daily.csv":
        "scripts/12: daily OpenET against measured closed and unclosed ET at the seven towers shared with GO26, by "
        "season, 2016 onward; seasonal rows of US-xSL and of the seven towers pooled withheld (openet_withheld) "
        "(OpenET-derived)",
    "openet_api_uic_winter.csv":
        "scripts/12: US-UiC winter: GO26 Fig. 8 observed ET (digitized, 2020) and the benchmark's measured winter "
        "ET at stations of the same Koppen class and the nearest stations, with the benchmark's gridMET ETo (no "
        "OpenET values)",
    "shared_site_comparison.csv":
        "scripts/06: per tower shared with GO26 (7): GO26 mean against the benchmark's long-term means and record "
        "coverage; OpenET Phase II bias, RMSE and R2 against closed, unclosed and Bowen-ratio closed ET (daily and "
        "monthly); GO26's zone results (Table 3)",
    "shared_site_monthly_pairs.csv":
        "scripts/06: monthly benchmark closed ET (et_corr, mm/month) with the days in the month (n_days), the days "
        "with closed ET, measured or gap-filled (n_et, equal to n_days in every month kept), the gap-filled days "
        "(n_gap; measured days = n_days - n_gap) and the "
        "OpenET Phase II ensemble (Volk et al., 2023d) at the six shared towers with qualifying months (all but "
        "US-xSL)",
    "site_koppen_check.csv":
        "scripts/09: GO26's Koppen class of each tower against the Rubel et al. (2017) grid, with its map category",
}
# entries that a complete pipeline may lack: Earth Engine check (scripts/02), files needing BASE files (scripts/10
# without BASE files writes only go26_observed_et_reconstruction.csv)
OPTIONAL_RESULTS = {"openet_api_eto_ee_check.csv", "go26_reconstructed_daily.csv", "go26_reconstructed_winter.csv",
                    "go26_reconstructed_openet_bias.csv", "go26_reconstructed_openet_sites.csv",
                    "go26_base_version_check.csv", "go26_reconstructed_figure_check.csv"}
# names every results/go26_* file (AmeriFlux BASE derivatives of scripts/10; tests/test_deposit.py checks this)
RESULTS_README_HEAD = """\
results/: derived tables of the comment on Golkar and de Oliveira (2026; GO26), written by the
numbered scripts (README.md of the code archive, Pipeline). One entry per file: the script that
writes it and what it holds. Read the CSV files with pandas.read_csv(path, comment="#"): the
go26_* files other than go26_observed_et_reconstruction.csv start with '#' lines (definitions,
licence, site DOIs).

Licences: CC BY 4.0 (LICENSE-data.md), except the OpenET-derived statistics in openet_api_*
(other than openet_api_uic_winter.csv, which holds no OpenET values),
go26_reconstructed_openet_bias.csv and go26_reconstructed_openet_sites.csv: OpenET data (OpenET,
https://openetdata.org), subject to the OpenET Terms of Service and not covered by CC BY 4.0;
attribute OpenET wherever they are shown.
The go26_* files (scripts/10) derive from AmeriFlux BASE data (CC BY 4.0, AmeriFlux CC-BY-4.0
data policy) and are CC BY 4.0 themselves, except the OpenET values noted: cite each site's BASE
DOI (config/go26_sites.csv; base_doi column of go26_observed_et_reconstruction.csv; header of
go26_reconstructed_daily.csv) with any of
  go26_base_version_check.csv
  go26_observed_et_reconstruction.csv
  go26_reconstructed_daily.csv
  go26_reconstructed_figure_check.csv
  go26_reconstructed_openet_bias.csv  (observed ET from BASE; its OpenET values are OpenET-derived)
  go26_reconstructed_openet_sites.csv  (observed ET from BASE; its OpenET values are OpenET-derived)
  go26_reconstructed_winter.csv
"""

# record of the joint OpenET disclosure check (scripts/14; local, not deposited) and the results it must cover
DISCLOSURE_RECORD = "data/raw/openet_api/derived/disclosure_check.json"
OPENET_RESULTS = ("results/openet_api_", "results/go26_reconstructed_openet_bias.csv",
                  "results/go26_reconstructed_openet_sites.csv")
NO_OPENET_VALUES = ("results/openet_api_uic_winter.csv",)  # named like them, but holds no OpenET values

# the code snapshot outside a git checkout (e.g. an unzipped deposit): what .gitignore lets git track
CODE_DIRS = (".github", "scripts", "src", "tests")
CODE_FILES = (".gitignore", "CITATION.cff", "LICENSE", "LICENSE-data.md", "README.md", "data/README.md",
              "environment.yml", "pyproject.toml")


class DepositError(RuntimeError):
    """The inputs of the deposit are incomplete; nothing was written."""


def _inside_root(paths) -> list[str]:
    """``paths`` that lie inside ROOT, as 'relative/path/' prefixes."""
    out = []
    for p in paths:
        try:
            out.append(Path(p).resolve().relative_to(ROOT.resolve()).as_posix().rstrip("/") + "/")
        except ValueError:
            pass
    return out


def code_files(exclude=()) -> list[str]:
    """Files git tracks or would track (respects .gitignore), excluding the deposit folder (zenodo/) and the
    folders in ``exclude`` (--out and the temporary build folder, when inside the repository)."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                             check=True, capture_output=True).stdout.decode()
        files = [f for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):  # no git, or not a git checkout
        files = [f for f in CODE_FILES if (ROOT / f).is_file()]
        files += [p.relative_to(ROOT).as_posix() for d in CODE_DIRS if (ROOT / d).is_dir()
                  for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix != ".pyc"
                  and not any(part.endswith(".egg-info") for part in p.parts)]
    prefixes = tuple(["zenodo/", *_inside_root(exclude)])
    return sorted(f for f in files if (ROOT / f).is_file() and not f.startswith(prefixes)
                  and not set(Path(f).parts) & SKIP)


def folder_files(folder: str) -> list[str]:
    """Files under ``folder`` (relative to ROOT), sorted; [] if the folder is missing."""
    base = ROOT / folder
    if not base.is_dir():
        return []
    return sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*")
                  if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP)


def results_readme(names: list[str]) -> str:
    """results/README.txt for the files ``names`` of results/ (every one needs a FILE_NOTES entry)."""
    missing = sorted(set(names) - set(FILE_NOTES))
    if missing:
        raise DepositError(f"no FILE_NOTES entry in scripts/13_make_zenodo_deposit.py for results/"
                           f"{', results/'.join(missing)}: add a one-line description of each")
    body = []
    for n in sorted(names):
        body += ["", n, *textwrap.wrap(FILE_NOTES[n], width=100, initial_indent="    ", subsequent_indent="    ",
                                       break_on_hyphens=False)]
    return RESULTS_README_HEAD + "\n".join(body) + "\n"


def site_plot_entries(problems: list[str]) -> list[tuple[str, Path | bytes]]:
    """figures/site_plots/html/ and static/ as benchmark_site_plots/html/ and static/, plus README.txt.

    An unzipped deposit has benchmark_site_plots/ instead of figures/site_plots/: when no part of figures/site_plots/
    has files, both parts are read from benchmark_site_plots/ (its README.txt is written anew).  A repository with
    both always uses figures/site_plots/, so fresh and stale plots are never mixed."""
    entries: list[tuple[str, Path | bytes]] = []
    base = SITE_PLOTS
    if not any(folder_files(f"{SITE_PLOTS}/{part}") for part in SITE_PLOT_PARTS) and \
            any(folder_files(f"benchmark_site_plots/{part}") for part in SITE_PLOT_PARTS):
        base = "benchmark_site_plots"  # an unzipped deposit
    for part in SITE_PLOT_PARTS:
        src = f"{base}/{part}"
        files = folder_files(src)
        if not files:
            problems.append(f"{src}/ is missing or empty: run scripts/03_site_plots.py (--no-render without Chromium)")
        entries += [(f"benchmark_site_plots/{part}/{f[len(src) + 1:]}", ROOT / f) for f in files]
    have = {a for a, _ in entries}
    need = [f"benchmark_site_plots/html/{s}_plots.html" for s in SHARED]
    need += [f"benchmark_site_plots/static/{s}_summary.{e}" for s in SHARED for e in ("pdf", "png")]
    lost = [n.replace("benchmark_site_plots/", f"{base}/", 1) for n in need if n not in have]
    if lost and entries:
        problems.append(f"{len(lost)} site plots of scripts/03 are missing: {', '.join(lost[:4])}"
                        + (" ..." if len(lost) > 4 else ""))
    return entries + [("benchmark_site_plots/README.txt", BENCHMARK_README.encode())]


def disclosure_problems() -> list[str]:
    """Why the record of scripts/14 does not cover the current files ([] if it does)."""
    rec = ROOT / DISCLOSURE_RECORD
    run = "run scripts/14_check_openet_disclosure.py (needs the OpenET extracts), or pass --no-disclosure-check"
    if not rec.is_file():
        return [f"{DISCLOSURE_RECORD} not found: {run}"]
    r = json.loads(rec.read_text())
    if not (r.get("passed") and r.get("complete")):
        return [f"the joint OpenET disclosure check in {DISCLOSURE_RECORD} "
                f"{'failed' if not r.get('passed') else 'was partial (--families or --no-controls)'}: {run}"]
    files = r.get("files", {})
    changed = [f for f, h in files.items() if not (ROOT / f).is_file() or md5(ROOT / f) != h]
    openet = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "results").glob("*")) if p.is_file()]
    uncovered = [f for f in openet if f.startswith(OPENET_RESULTS) and f not in NO_OPENET_VALUES and f not in files]
    out = []
    if changed:
        out.append(f"changed since the joint OpenET disclosure check: {', '.join(changed)}: {run}")
    if uncovered:
        out.append(f"OpenET-derived files the joint disclosure check did not read: {', '.join(uncovered)}: add "
                   f"them to scripts/14")
    return out


def collect(version: str, exclude=(), disclosure: bool = True) -> list[tuple[str, list[tuple[str, Path | bytes]], str, str]]:
    """Every archive as (name, entries, contents, licence), entries as (name in the zip, file or bytes).

    Checks all inputs first and raises DepositError listing every problem, so that nothing is written (and an
    existing deposit is kept) when one is missing or incomplete; ``disclosure`` includes the record of scripts/14.
    """
    problems: list[str] = disclosure_problems() if disclosure else []
    files = code_files(exclude)
    code = f"{NAME}-{version}.zip"
    if not any(f.startswith("scripts/") for f in files):
        problems.append("no scripts/ found for the code snapshot")
    archives = [(code, [(f"{NAME}-{version}/{f}", ROOT / f) for f in files],
                 f"snapshot of the GitHub repository ({len(files)} files: code, tests, README, licences, "
                 f"CITATION.cff, environment.yml)", "MIT (code)")]
    for name, (folder, what, lic) in FOLDERS.items():
        entries: list[tuple[str, Path | bytes]]
        if name == "benchmark_site_plots.zip":
            entries = site_plot_entries(problems)
        else:
            ff = folder_files(folder)
            if not ff:
                problems.append(f"{folder}/ is missing or empty: run the pipeline (README.md) or unzip the deposit")
            if name == "figures.zip":
                ff = [f for f in ff if not f.startswith(LEFT_OUT_OF_FIGURES)]
                need = [f"figures/{f}.{e}" for f in FIGURES for e in ("pdf", "png")] + list(QC)
                if ff and (lost := [f for f in need if f not in ff]):
                    problems.append(f"figures missing: {', '.join(lost)} (scripts/04, 06, 08 after 10, 09)")
            if name == "results.zip":
                ff = [f for f in ff if f != "results/README.txt"]  # written anew below
            entries = [(f, ROOT / f) for f in ff]
            if name == "results.zip" and ff:
                names = [f.split("/", 1)[1] for f in ff]
                if absent := sorted(set(FILE_NOTES) - OPTIONAL_RESULTS - set(names)):
                    problems.append(f"results missing: results/{', results/'.join(absent)} (README.md, Pipeline)")
                try:
                    entries.append(("results/README.txt", results_readme(names).encode()))
                except DepositError as err:
                    problems.append(str(err))
        for _, src in entries:  # third-party e-mail addresses must not be redistributed
            if isinstance(src, Path) and src.suffix.lower() in TEXT_SUFFIXES and _has_email(src.read_bytes()):
                problems.append(f"{src.relative_to(ROOT)} contains e-mail addresses: remove them (scripts/11 does "
                                f"for the AmeriFlux site pages)")
        n = len(entries)
        archives.append((name, entries, f"`{folder}/`: {what} ({n} files)", lic))
    if problems:
        raise DepositError("the deposit inputs are incomplete:\n  - " + "\n  - ".join(problems))
    return archives


def _has_email(data: bytes) -> bool:
    from etcomment import amf_public as amf  # the redaction rule of the cached AmeriFlux pages
    return amf.redact_emails(data)[1] > 0


def write_zip(path: Path, entries) -> None:
    """Deterministic zip of ``entries`` [(name in the zip, file path or bytes)]: sorted by name, fixed
    timestamp, permissions (0644) and creator fields (Unix, zip version 2.0 also when built on Windows)."""
    names = [a for a, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: duplicate entries")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arcname, src in sorted(entries, key=lambda e: e[0]):
            info = zipfile.ZipInfo(arcname, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            info.create_version = info.extract_version = 20
            z.writestr(info, src if isinstance(src, bytes) else Path(src).read_bytes(), compresslevel=9)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.1f} kB"


def citation_authors(path: Path) -> list[dict[str, str]]:
    """Authors of CITATION.cff (given-names, family-names, orcid, ...), read without a YAML parser."""
    authors: list[dict[str, str]] = []
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "-")):  # a top-level key
            inside = line.split(":", 1)[0].strip() == "authors"
            continue
        m = re.match(r"\s*(-\s+)?([\w-]+):\s*(.*?)\s*$", line) if inside else None
        if m:
            if m[1]:
                authors.append({})
            if authors:
                authors[-1][m[2]] = m[3].strip("'\"")
    return authors


def author_line(path: Path | None = None) -> str:
    """'Given Family, ...' from CITATION.cff, each with a link to its ORCID iD when CITATION.cff gives one."""
    path = path or ROOT / "CITATION.cff"
    if not path.is_file():
        return AUTHORS
    names = []
    for a in citation_authors(path):
        name = " ".join(a[k] for k in ("given-names", "family-names") if a.get(k))
        if orcid := a.get("orcid"):
            url = orcid if orcid.startswith("http") else f"https://orcid.org/{orcid}"
            name += f" ([{url.rsplit('/', 1)[-1]}]({url}))"
        names.append(name)
    return ", ".join(names)


def cited_authors(path: Path | None = None) -> str:
    """'Family, I.I., ..., & Family, I.' from CITATION.cff (AUTHORS without it)."""
    path = path or ROOT / "CITATION.cff"
    names = [f"{a['family-names']}, " + "".join(f"{w[0]}." for w in re.split(r"[\s.]+", a.get("given-names", "")) if w)
             for a in (citation_authors(path) if path.is_file() else []) if a.get("family-names")]
    return (", ".join(names[:-1]) + ", & " + names[-1] if len(names) > 1 else names[0]) if names else AUTHORS


def deposit_citation(version: str, path: Path | None = None) -> str:
    """This deposit as a reference: 'Family, I.I., ..., & Family, I. (year). Title (Version v) [dataset]. Zenodo. DOI',
    with the authors of CITATION.cff (AUTHORS without it)."""
    return f"{cited_authors(path)} ({STAMP[0]}). {TITLE} (Version {version}) [dataset]. Zenodo. https://doi.org/{DOI}"


def preprint_citation(path: Path | None = None) -> str:
    """The comment's EarthArXiv preprint as a reference, with the authors of the deposit (the same six)."""
    return f"{cited_authors(path)} (2026). {COMMENT} [preprint]. EarthArXiv. https://doi.org/{PREPRINT_DOI}"


def readme(version: str, rows: list[tuple[str, str, int, str, str]]) -> str:
    code = rows[0][0]  # the code snapshot: no md5 here or in MD5SUMS
    table = "\n".join(f"| `{n}` | {c} | {human(size)} | {lic} | " + ("–" if n == code else f"`{h}`") + " |"
                      for n, c, size, lic, h in rows)
    built = f"Python {platform.python_version()}, zlib {zlib.ZLIB_RUNTIME_VERSION}"
    return f"""# {TITLE}

Version {version} · DOI [{DOI}](https://doi.org/{DOI}) · Preprint of the comment (EarthArXiv):
[{PREPRINT_DOI}](https://doi.org/{PREPRINT_DOI}) · Code: {REPO_URL}

{author_line()}
({AFFILIATION})

## Description

Reproducible analysis for the comment *Comment on Golkar and de Oliveira (2026): Missing eddy
covariance observations undermine reported OpenET biases* (Kim et al., 2026, EarthArXiv preprint,
https://doi.org/{PREPRINT_DOI}; submitted to *Agricultural and Forest Meteorology*) on Golkar,
M.K., de Oliveira, L.A., 2026, Diagnosing and
correcting OpenET satellite evapotranspiration across US climate zones: Error decomposition,
explainable AI, and uncertainty quantification, *Agric. For. Meteorol.* 391, 111524,
https://doi.org/10.1016/j.agrformet.2026.111524 (GO26, CC BY 4.0).

The comment shows that missing eddy covariance observations were effectively treated as zero ET
in GO26. The deposit holds the code (a snapshot of the GitHub repository) and its inputs and
outputs: the digitized GO26 figures, the comparison with the Volk et al. (2023a) benchmark,
AmeriFlux's public data-availability records, observed ET rebuilt from the AmeriFlux BASE files
of GO26's 22 towers, statistics of the current OpenET collection (2.1) at GO26's towers and the
benchmark cropland stations (the OpenET extracts themselves are not deposited), a train/test
leakage experiment, Figures 1 and 2 of the comment and the further figures of the analysis.

Data dates (UTC, from the provenance files):

- OpenET API extraction (collection 2.1): 15 October 2026, 03:38-04:09 UTC (request log
  `data/raw/openet_api/manifest.json`, not deposited; `results/openet_api_go26_summary.json`,
  `provenance`); daily ensemble ET at 13 further GO26 towers, for the check of Figure 2 of the
  comment: 28 October 2026, 14:49-15:18 UTC.
- AmeriFlux public metadata: 15 October 2026, 04:26-04:29 UTC
  (`data/raw/ameriflux_public/provenance.json`).
- AmeriFlux BASE files: 15 October 2026, 08:02:54-08:03:22 UTC (`data/raw/ameriflux_base/provenance.json`,
  not deposited); the versions equal those GO26 cite at all 22
  sites (`results/go26_observed_et_reconstruction.csv`, `base_version`, `version_matches_go26`).

## Changes in version 1.1.0

Version 1.1.0 (28 October 2026) adds Figure 2 of the comment; nothing published in version 1.0.0
(15 October 2026, https://doi.org/10.5281/zenodo.23383493) changes.

- New figure: `figures/fig_openet_reconstruction.*` (Figure 2 of the comment), the EC site means
  of Figure 1 against the current OpenET ensemble averaged over the same days at GO26's towers:
  every day of the study period for the zero-filled means, and the days on which every
  processing variant has a value for the valid-day, gap-filled and closure-corrected means.
- New table: `results/go26_reconstructed_openet_sites.csv` (`scripts/10`), the per-tower OpenET
  and observed means behind Figure 2 (OpenET-derived; checked jointly with the other OpenET
  statistics by `scripts/14`).
- Daily OpenET ensemble ET extracted at the 13 further GO26 towers (28 October 2026; not
  deposited) to check the monthly averaging of Figure 2; `scripts/02` now makes 37 requests.
- Digitization sensitivity (`scripts/04`; new key `sensitivity` in
  `results/digitized_summary.json`, its other values unchanged): at US-UTJ the RMSE improvement
  without the exact zeros is -1.2% to +4.0% (0.5% as digitized) with the dots hidden under the
  plotted lines added and the zero tolerance doubled.
- Code: `scripts/02`, `04`, `08`, `10`, `13` and `14` and their tests; `python-docx` added to
  `environment.yml` (`docs` extra; no result depends on it).
- Unchanged: every other file in `results/` and `figures/`, and `config.zip`,
  `data_external_golkar_deoliveira_2026.zip` and `data_raw_ameriflux_public.zip` (identical md5
  sums); `benchmark_site_plots.zip` differs only in the DOI in its `README.txt`.

## Contents

| File | Contents | Size | Licence | MD5 |
|---|---|---|---|---|
{table}

Every archive except `{code}` holds paths relative to the repository root.
`results.zip` includes `results/README.txt`, a one-line guide to every file.

`MD5SUMS` lists the data archives (all but `{code}`). Verify them with `md5sum -c MD5SUMS`
(Linux) or `md5 -r *.zip` (macOS; compare with `MD5SUMS`). They are deterministic:
`scripts/13_make_zenodo_deposit.py` rebuilds them with the same md5 sums from unchanged files
with the same Python and zlib versions (built with
{built}; `environment.yml` pins Python, not zlib), since zlib's compressed output can differ
between versions. Without the OpenET
extracts, pass `--no-disclosure-check`: before building, the script requires the record of the
joint OpenET disclosure check (`scripts/14`), which needs them.

## How to use

1. Unzip `{code}`, then unzip the other archives inside the folder it creates
   (`{NAME}-{version}/`), so that `config/`, `data/`, `results/`, `figures/` and
   `benchmark_site_plots/` sit next to `scripts/` and `src/`.
2. Inside that folder, create the environment: `conda env create -f environment.yml && conda
   activate etcomment` (exact versions of the direct dependencies; it installs the code in the
   folder with `pip install -e .`). Tested with `environment.yml` (Python 3.12.15, macOS arm64);
   CI runs the tests on Ubuntu with Python 3.11 and 3.12. `python -m pytest -q` runs the offline
   tests.
3. Inputs that are not in this deposit (`README.md` in the code archive, Data streams):
   - the Volk et al. (2023c, 2023d) benchmark and OpenET Phase II data, public on Zenodo
     (doi:10.5281/zenodo.7636781, doi:10.5281/zenodo.10119477): `python scripts/01_download_data.py`;
   - the OpenET API extracts (collection 2.1, 200 m x 200 m boxes as in GO26): a free OpenET API
     key in `OPENET_API_KEY`, then `python scripts/02_download_openet_api.py` (37 requests).
     Without a key, skip scripts 02, 12 and 14 (their outputs are in `results/`) and build the
     deposit with `scripts/13 --no-disclosure-check`. OpenET revises its
     collections, so a new extraction can differ slightly; if the md5 sums of the raw extracts
     match those recorded in `results/openet_api_go26_summary.json` (`provenance`: `md5_openet_monthly.csv`,
     `md5_openet_daily.csv`, `md5_sites.csv`, `md5_sites_200m.geojson`), `results/openet_api_*`
     (except `openet_api_eto_ee_check.csv`, which queries the live Earth Engine collection) and
     `figures/fig2_*` are reproduced exactly;
   - optionally, AmeriFlux BASE files for `scripts/10` (AmeriFlux account; the versions GO26 cite
     are in `config/go26_sites.csv`; extract the BASE CSV of each
     `AMF_<SITE>_BASE-BADM_<version>.zip` into `data/raw/ameriflux_base/`, as `scripts/10` reads
     only the CSV files); without them skip `scripts/10`, which would otherwise
     overwrite `results/go26_observed_et_reconstruction.csv` with status rows.
4. Run the numbered scripts as listed in `README.md` (Pipeline). `scripts/03` renders PNG
   snapshots of the benchmark's Bokeh pages with Playwright (`python -m playwright install
   chromium`), or pass `--no-render`. The Bokeh pages (`benchmark_site_plots/html/`) load
   BokehJS 2.1.1 from cdn.bokeh.org, so open them online. `results/` and `figures/` in this
   deposit are the outputs used in the comment; a rerun with the same inputs and environment
   reproduces them (the figure scripts rerun themselves with PYTHONHASHSEED=0, which makes the
   PDFs byte-reproducible on the same platform and fonts; they were made on macOS arm64 with
   its system Helvetica).

## Licences

- Code: MIT (`LICENSE`).
- `config/`, `results/` and `figures/`: CC BY 4.0 (`LICENSE-data.md`), except the
  OpenET-derived statistics below.
- `benchmark_site_plots/`: CC BY 4.0; `html/` are the Bokeh pages of Volk et al. (2023c),
  unchanged; `static/` is this work.
- OpenET-derived statistics (`results/openet_api_*`, except `results/openet_api_uic_winter.csv`,
  which holds no OpenET values; `results/go26_reconstructed_openet_bias.csv` and
  `results/go26_reconstructed_openet_sites.csv`; panels b-d of `figures/fig2_*`; the OpenET
  values of Figure 2 of the comment, `figures/fig_openet_reconstruction.*`): OpenET data,
  subject to the OpenET Terms of Service (https://etdata.org/terms-of-service/); attribute
  "OpenET, https://openetdata.org" wherever they are shown. They are not covered by CC BY 4.0.
- AmeriFlux BASE derivatives (`results/go26_base_version_check.csv`,
  `results/go26_observed_et_reconstruction.csv`, `results/go26_reconstructed_daily.csv`,
  `results/go26_reconstructed_figure_check.csv`, `results/go26_reconstructed_winter.csv`, and the
  observed ET of `results/go26_reconstructed_openet_bias.csv` and
  `results/go26_reconstructed_openet_sites.csv`, whose OpenET values are OpenET-derived; the
  observed ET behind panel d of `figures/fig2_*`; Figure 1 of the comment,
  `figures/fig_table1_reconstruction.*`, and the observed ET of Figure 2,
  `figures/fig_openet_reconstruction.*`): CC BY 4.0 under
  the AmeriFlux CC-BY-4.0 data policy; cite each site's BASE DOI
  (`config/go26_sites.csv`; also in the `base_doi` column of
  `results/go26_observed_et_reconstruction.csv` and the header of the daily file).
- AmeriFlux public metadata (`data/raw/ameriflux_public/`): the metadata (the amerifluxr summary and
  the web-service responses) are CC BY 4.0 under the AmeriFlux data policy. The site pages
  (`siteinfo_*.html`) are AmeriFlux website content (© LBNL / UC Regents; the photos on them carry
  their own terms), cached unchanged except for the removed e-mail addresses, for provenance only;
  they are not covered by the licences of this deposit. `results/ameriflux_*` derive from the
  metadata (CC BY 4.0; attribute AmeriFlux and the site teams).
- GO26 figures (`data/external/golkar_deoliveira_2026/`): CC BY 4.0, Golkar and de Oliveira (2026).
- Map layers of `figures/fig3_*` and `results/site_koppen_check.csv`: the Koppen-Geiger grid of
  Rubel et al. (2017), via `kgcpy` (BSD); U.S. Census Bureau state boundaries, shipped with
  `basemap-data` (MIT); GSHHG version 2.3.6 lakes (Wessel and Smith, 1996), shipped with
  `basemap-data` (LGPL-3.0-or-later).
- `results/digitized_*` include GO26's plotted OpenET line (digitized from their CC BY 4.0
  figures): attribute GO26 and "OpenET, https://openetdata.org" wherever it is shown.
- The AmeriFlux sites and the funding of AmeriFlux data resources are acknowledged below
  (Acknowledgements), as the AmeriFlux CC-BY-4.0 data policy asks.

## Acknowledgements

Funding provided by U.S. Geological Survey Water Resources Research Institute, OpenET Planning
Phase III Grant G25AC00180-00. We acknowledge the following AmeriFlux sites for their data
records: US-ARM, US-Bi1, US-Bi2, US-IAB, US-MN1, US-MN2, US-MN3, US-Mo1, US-NSb, US-Ne1,
US-Ne2, US-Ne3, US-RGA, US-RGB, US-UTD, US-UTJ, US-UTV, US-UiC, US-UiG, US-VT1, US-WT1, and
US-xSL. In addition, funding for AmeriFlux data resources was provided by the U.S. Department
of Energy's Office of Science.

## Citation

Please cite this deposit as

{deposit_citation(version)}

and the comment (the EarthArXiv preprint; its journal version once published) as

{preprint_citation()}

The code is also on GitHub, {REPO_URL}, of which the code archive is a snapshot (`CITATION.cff`
there and in the archive). Please also cite GO26,
Volk et al. (2023a, 2023b, 2023c, 2023d, 2024), Volk et al. (2026) for the bias-corrected gridMET
reference ET, flux-data-qaqc (Volk et al., 2021; version 0.5.0) for the energy balance closure,
Twine et al. (2000) for the Bowen-ratio closure as GO26 describe it, OpenET (Melton et al.,
2022), Rubel et al. (2017) for the Koppen-Geiger grid, Wessel and Smith
(1996) for the lakes on the site map, and the AmeriFlux BASE DOIs of the sites
(`config/go26_sites.csv`). The full references are in `README.md` in the code archive.

## Related identifiers

- Supplement to (the comment, EarthArXiv preprint): https://doi.org/{PREPRINT_DOI}
- Comment on: https://doi.org/10.1016/j.agrformet.2026.111524
- Source code: {REPO_URL}
- Data used: https://doi.org/10.5281/zenodo.7636781, https://doi.org/10.5281/zenodo.10119477
- References: https://doi.org/10.1016/j.agrformet.2023.109307, https://doi.org/10.1016/j.dib.2023.109274,
  https://doi.org/10.1038/s44221-023-00181-7, https://doi.org/10.1016/j.agwat.2026.110647,
  https://doi.org/10.21105/joss.03418, https://doi.org/10.1111/1752-1688.12956,
  https://doi.org/10.1127/metz/2016/0816, https://doi.org/10.1029/96JB00104,
  https://doi.org/10.1016/S0168-1923%2800%2900123-4

## Use of AI tools

The authors identified potential methodological issues in the original study and used Claude
(Anthropic) to assist in developing and refining code to investigate these issues and
independently reproduce the reported results. All code, analyses, and interpretations were
reviewed and verified by the authors, who retain full responsibility for the findings and
conclusions. The analysis code is publicly available on GitHub to support transparency and
reproducibility.
"""


# HTML tags Zenodo keeps in a record's description (InvenioRDM's sanitizer); description_html uses no others
ZENODO_TAGS = ("a", "code", "em", "h2", "li", "ol", "p", "table", "tbody", "td", "th", "tr", "ul")
LIST_ITEM = re.compile(r"^( *)(-|\d+\.) (.*)$")


def _inline(text: str) -> str:
    """Inline Markdown of the deposit README as HTML: `code`, *emphasis*, [links](url) and bare URLs."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", text)):
        part = html.escape(part, quote=False)
        if i % 2:
            out.append(f"<code>{part}</code>")
            continue
        links: list[str] = []

        def keep(m: re.Match) -> str:  # written links, kept out of the bare-URL pass
            links.append(f'<a href="{m.group(2)}">{m.group(1)}</a>')
            return f"\0{len(links) - 1}\0"

        part = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", keep, part)
        url = r"""https?://[^\s<>()"']*[^\s<>().,;:"']"""  # a quote or closing punctuation ends the URL
        part = re.sub(url, lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', part)
        part = re.sub(r"\*([^*\s][^*]*?)\*", r"<em>\1</em>", part)
        out.append(re.sub(r"\0(\d+)\0", lambda m: links[int(m.group(1))], part))
    return "".join(out)


def _list_html(items: list[list]) -> str:
    """Nested ul/ol from (indent, kind, text) items; deeper items belong to the item above them."""
    parts, i = [], 0
    while i < len(items):
        indent, _, text = items[i]
        j = i + 1
        while j < len(items) and items[j][0] > indent:
            j += 1
        parts.append(f"<li>{_inline(text)}{_list_html(items[i + 1:j]) if j > i + 1 else ''}</li>")
        i = j
    kind = items[0][1]
    return f"<{kind}>{''.join(parts)}</{kind}>"


def description_html(readme_md: str) -> str:
    """The deposit README (Markdown, :func:`readme`) as HTML for the Description field of the Zenodo record.

    The title line is left out (Zenodo shows the record's title); sections become h2, the contents table a table,
    lists ul/ol (wrapped lines and nested items as in the README), other blocks paragraphs.  Only ZENODO_TAGS.
    """
    lines = readme_md.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out = []
    for block in re.split(r"\n\s*\n", "\n".join(lines).strip()):
        rows = block.splitlines()
        if block.startswith("## "):
            out.append(f"<h2>{_inline(block[3:].strip())}</h2>")
        elif all(r.startswith("|") for r in rows):
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows
                     if not re.fullmatch(r"\|[-| :]+\|", r.strip())]
            body = "".join("<tr>" + "".join(f"<{t}>{_inline(c)}</{t}>" for c in row) + "</tr>"
                           for t, row in [("th", cells[0])] + [("td", r) for r in cells[1:]])
            out.append(f"<table><tbody>{body}</tbody></table>")
        elif LIST_ITEM.match(rows[0]):
            items: list[list] = []
            for r in rows:
                m = LIST_ITEM.match(r)
                if m:
                    items.append([len(m.group(1)), "ol" if m.group(2)[0].isdigit() else "ul", m.group(3)])
                else:
                    items[-1][2] += " " + r.strip()
            out.append(_list_html(items))
        else:
            out.append(f"<p>{_inline(' '.join(r.strip() for r in rows))}</p>")
    return "\n".join(out) + "\n"


def install(tmp: Path, out: Path) -> None:
    """Move the files built in ``tmp`` into ``out``.  Zips of an earlier build that the new one does not replace
    (another version) are deleted; other files in ``out`` are kept."""
    out.mkdir(parents=True, exist_ok=True)
    new = sorted(p.name for p in tmp.iterdir())
    for old in out.glob("*.zip"):
        if old.name not in new:
            old.unlink()
    for name in new:
        os.replace(tmp / name, out / name)


def build(out: Path, version: str, disclosure: bool = True) -> list[tuple[str, str, int, str, str]]:
    """Check the inputs, build the deposit in a temporary folder next to ``out`` and move it into place.

    Returns the rows of the README table (name, contents, size, licence, md5).  Raises DepositError, with
    nothing written, when inputs are missing; any other failure leaves ``out`` as it was, too.
    """
    out = out.resolve()
    tmp = out.parent / f".{out.name}.building"
    archives = collect(version, exclude=(out, tmp), disclosure=disclosure)
    if tmp.exists():
        shutil.rmtree(tmp)  # left over from an interrupted build
    tmp.mkdir(parents=True)
    try:
        rows = []
        for name, entries, what, lic in archives:
            write_zip(tmp / name, entries)
            rows.append((name, what, (tmp / name).stat().st_size, lic, md5(tmp / name)))
        text = readme(version, rows)
        (tmp / "README.md").write_text(text, encoding="utf-8")
        (tmp / "description.html").write_text(description_html(text), encoding="utf-8")
        (tmp / "MD5SUMS").write_text("".join(f"{h}  {n}\n" for n, _, _, _, h in rows[1:]))  # not the code snapshot
        install(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", default=str(ROOT / "zenodo"))
    ap.add_argument("--version", default="1.1.0")
    ap.add_argument("--no-disclosure-check", action="store_true",
                    help="build without the record of scripts/14 (e.g. from an unzipped deposit); not for upload")
    args = ap.parse_args(argv)
    if args.no_disclosure_check:
        print("WARNING: --no-disclosure-check: the joint OpenET disclosure check (scripts/14) was not verified")
    try:
        rows = build(Path(args.out), args.version, disclosure=not args.no_disclosure_check)
    except DepositError as err:
        raise SystemExit(f"STOPPED: {err}\nNothing was written; {args.out} is unchanged.") from None
    for n, _, size, _, h in rows:
        print(f"{h}  {human(size):>9}  {n}")
    print(f"wrote {len(rows)} archives, README.md, description.html and MD5SUMS to {args.out}")


if __name__ == "__main__":
    main()
