"""Build the Zenodo deposit (doi:10.5281/zenodo.23383493) in zenodo/.

The deposit holds a snapshot of the code and the inputs and outputs that git
does not track, one zip per folder with paths relative to the repository root
(unzip them there), plus README.md (the deposit description, with sizes and
md5 sums) and MD5SUMS:

  openet-bias-correction-comment-afm-<version>.zip   the files git tracks: code, tests, docs, metadata
  config.zip                                         GO26 Tables 1 and 3, V24 station list
  data_external_golkar_deoliveira_2026.zip           GO26 Figs. 2, 8 and S2 (CC BY 4.0)
  data_raw_ameriflux_public.zip                      cached public AmeriFlux metadata (scripts/11)
  results.zip                                        derived results
  figures.zip                                        figures of the comment, QC overlays, site summaries
  benchmark_site_plots.zip                           benchmark plots of the seven shared towers

Not deposited: the other downloads in data/raw/ (the Volk et al. data are on
Zenodo under their own DOIs; the OpenET extracts need an API key and stay
local under the OpenET Terms of Service; AmeriFlux BASE files are not
redistributed), docs/internal/, paper/ and .env.

The zips are deterministic (sorted entries, fixed timestamps and permissions,
no .DS_Store or __pycache__), so a rebuild from unchanged files gives the same
md5 sums.  Run it after the pipeline (README.md, Pipeline).

Usage:
  python scripts/13_make_zenodo_deposit.py [--out zenodo] [--version 1.0.0]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "openet-bias-correction-comment-afm"
DOI = "10.5281/zenodo.23383493"
STAMP = (2026, 10, 15, 0, 0, 0)  # release date, used as every entry's timestamp
SKIP = {".DS_Store", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
# archive name -> (folder relative to ROOT, contents, licence)
FOLDERS = {
    "config.zip": ("config", "GO26 Tables 1 and 3 (sites, periods, mean observed ET, BASE DOIs and the versions GO26 "
                             "cite), Volk et al. (2024) station list", "CC BY 4.0"),
    "data_external_golkar_deoliveira_2026.zip": (
        "data/external/golkar_deoliveira_2026", "GO26 Figs. S2 and 8 (digitized by scripts/04) and Fig. 2, "
                                                "unchanged from the article", "CC BY 4.0 (Golkar and de Oliveira, 2026)"),
    "data_raw_ameriflux_public.zip": (
        "data/raw/ameriflux_public", "cached public AmeriFlux metadata read by scripts/11: BASE variable-availability "
                                     "summary (amerifluxr, commit 7319e63), web-service responses and site pages, "
                                     "with access times and md5 sums", "CC BY 4.0 (AmeriFlux data policy)"),
    "results.zip": ("results", "derived tables (CSV, JSON) of scripts 04-12", "CC BY 4.0, except OpenET-derived values"),
    "figures.zip": ("figures", "Figs. 1-5 of the comment (PNG 600 dpi, PDF), QC overlays of the digitization, "
                               "benchmark site summaries", "CC BY 4.0, except Fig. 2b-d (OpenET-derived)"),
    "benchmark_site_plots.zip": ("benchmark_site_plots", "benchmark plots of the seven towers shared with GO26: "
                                                          "Bokeh pages and static summaries", "CC BY 4.0 (Volk et al., 2023c)"),
}


# the code snapshot outside a git checkout (e.g. an unzipped deposit): what .gitignore lets git track
CODE_DIRS = (".github", "scripts", "src", "tests")
CODE_FILES = (".gitignore", "CITATION.cff", "LICENSE", "LICENSE-data.md", "README.md", "data/README.md",
              "environment.yml", "pyproject.toml")


def code_files() -> list[str]:
    """Files git tracks or would track (respects .gitignore), excluding the deposit folder itself."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                             check=True, capture_output=True).stdout.decode()
        files = [f for f in out.split("\0") if f and not f.startswith("zenodo/")]
    except (OSError, subprocess.CalledProcessError):  # no git, or not a git checkout
        files = [f for f in CODE_FILES if (ROOT / f).is_file()]
        files += [str(p.relative_to(ROOT)) for d in CODE_DIRS if (ROOT / d).is_dir()
                  for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix != ".pyc"
                  and not any(part.endswith(".egg-info") for part in p.parts)]
    return sorted(f for f in files if (ROOT / f).is_file() and not set(Path(f).parts) & SKIP)


def folder_files(folder: str) -> list[str]:
    base = ROOT / folder
    if not base.is_dir():
        raise FileNotFoundError(f"{folder}/ is missing: run the pipeline (README.md) before packaging")
    return sorted(str(p.relative_to(ROOT)) for p in base.rglob("*")
                  if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP)


def write_zip(path: Path, files: list[str], prefix: str = "") -> None:
    """Deterministic zip of ``files`` (paths relative to ROOT), stored under ``prefix``."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo(prefix + f, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (ROOT / f).read_bytes(), compresslevel=9)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.1f} kB"


def readme(version: str, rows: list[tuple[str, str, int, str, str]]) -> str:
    table = "\n".join(f"| `{n}` | {c} | {human(size)} | {lic} | `{h}` |" for n, c, size, lic, h in rows)
    code = rows[0][0]
    return f"""# Code and data for: Comment on Golkar and de Oliveira (2026): Exact-zero observed evapotranspiration undermines the diagnosis of OpenET bias

Version {version} · DOI [{DOI}](https://doi.org/{DOI}) · Code: https://github.com/montimaj/{NAME}

Yeonuk Kim, Sayantan Majumdar, John M. Volk, Peter ReVelle, Todd G. Caldwell, Justin L. Huntington
(Desert Research Institute, Reno, Nevada, USA)

## Description

Reproducible analysis for the comment *Comment on Golkar and de Oliveira (2026): Exact-zero
observed evapotranspiration undermines the diagnosis of OpenET bias* (Kim et al., submitted to
*Agricultural and Forest Meteorology*) on Golkar, M.K., de Oliveira, L.A., 2026, Diagnosing and
correcting OpenET satellite evapotranspiration across US climate zones, *Agric. For. Meteorol.*
391, 111524, https://doi.org/10.1016/j.agrformet.2026.111524 (GO26, CC BY 4.0).

The comment shows that the observed eddy covariance ET used by GO26 counts missing data as zero.
The deposit holds the code (a snapshot of the GitHub repository) and its inputs and outputs: the
digitized GO26 figures, the comparison with the Volk et al. (2023a) benchmark, AmeriFlux's public
data-availability records, the current OpenET collection at GO26's towers and the benchmark
cropland stations, a train/test leakage experiment, and the figures of the comment.

## Contents

| File | Contents | Size | Licence | MD5 |
|---|---|---|---|---|
{table}

Every archive except `{code}` holds paths relative to the repository root.

## How to use

1. Unzip `{code}`, then unzip the other archives inside the folder it creates
   (`{NAME}-{version}/`), so that `config/`, `data/`, `results/`, `figures/` and
   `benchmark_site_plots/` sit next to `scripts/` and `src/`.
2. Create the environment: `conda env create -f environment.yml && conda activate etcomment`
   (exact versions of the direct dependencies; `python -m pytest -q` runs the offline tests).
3. Inputs that are not in this deposit (`README.md` in the code archive, Data streams):
   - the Volk et al. (2023c, 2023d) benchmark and OpenET Phase II data, public on Zenodo
     (doi:10.5281/zenodo.7636781, doi:10.5281/zenodo.10119477): `python scripts/01_download_data.py`;
   - the OpenET API extracts (collection 2.1, 200 m x 200 m boxes as in GO26): a free OpenET API
     key in `OPENET_API_KEY`, then `python scripts/02_download_openet_api.py` (24 requests);
   - optionally, AmeriFlux BASE files for `scripts/10` (AmeriFlux account).
4. Run the numbered scripts as listed in `README.md` (Pipeline). `results/` and `figures/` in this
   deposit are the outputs used in the comment; a rerun with the same inputs reproduces them.

## Licences

- Code: MIT (`LICENSE`).
- `config/`, `results/`, `figures/` and `benchmark_site_plots/`: CC BY 4.0 (`LICENSE-data.md`), except
  values derived from OpenET.
- OpenET-derived statistics (`results/openet_api_*`, Fig. 2b-d): OpenET data, https://openetdata.org,
  subject to the OpenET Terms of Service; attribute OpenET wherever they are shown. They are not
  covered by CC BY 4.0.
- GO26 figures (`data/external/golkar_deoliveira_2026/`): CC BY 4.0, Golkar and de Oliveira (2026).
- AmeriFlux public metadata (`data/raw/ameriflux_public/`): AmeriFlux CC BY 4.0 data policy.
- Benchmark Bokeh pages (`benchmark_site_plots/html/`): Volk et al. (2023c), CC BY 4.0.

## Citation

Please cite this deposit (`CITATION.cff` in the code archive), the comment once published, GO26,
Volk et al. (2023a, 2023c, 2023d, 2024), Volk et al. (2026) for the bias-corrected gridMET reference
ET, OpenET (Melton et al., 2022), and the AmeriFlux BASE DOIs of the sites
(`config/go26_sites.csv`).

## Related identifiers

- Comment on: https://doi.org/10.1016/j.agrformet.2026.111524
- Source code: https://github.com/montimaj/{NAME}
- Data used: https://doi.org/10.5281/zenodo.7636781, https://doi.org/10.5281/zenodo.10119477
- References: https://doi.org/10.1038/s44221-023-00181-7, https://doi.org/10.1016/j.agwat.2026.110647

## Use of AI tools

The analysis code, figures and the first draft of the comment were prepared with the help of
Claude (Anthropic). The authors reviewed the code and its results and take responsibility for them.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "zenodo"))
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.zip"):
        old.unlink()
    code = f"{NAME}-{args.version}.zip"
    files = code_files()
    write_zip(out / code, files, prefix=f"{NAME}-{args.version}/")
    rows = [(code, f"snapshot of the GitHub repository ({len(files)} files: code, tests, README, licences, "
                   f"CITATION.cff, environment.yml)", "MIT (code)")]
    for name, (folder, what, lic) in FOLDERS.items():
        ff = folder_files(folder)
        write_zip(out / name, ff)
        rows.append((name, f"`{folder}/`: {what} ({len(ff)} files)", lic))
    table = [(n, c, (out / n).stat().st_size, lic, md5(out / n)) for n, c, lic in rows]
    (out / "README.md").write_text(readme(args.version, table))
    (out / "MD5SUMS").write_text("".join(f"{h}  {n}\n" for n, _, _, _, h in table))
    for n, _, size, _, h in table:
        print(f"{h}  {human(size):>9}  {n}")
    print(f"wrote {len(table)} archives, README.md and MD5SUMS to {out}")


if __name__ == "__main__":
    main()
