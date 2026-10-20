"""Tests for the Zenodo deposit (scripts/13) and the reproducible-PDF helpers of etcomment.style.

Synthetic files only: no network, no data downloads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from etcomment import style as st

REPO = Path(__file__).resolve().parents[1]
CFF = """cff-version: 1.2.0
title: "Code"
authors:
  - family-names: Doe
    given-names: Jane A.
    orcid: "https://orcid.org/0000-0001-2345-6789"
  - family-names: Roe
    given-names: Richard
references:
  - type: article
    authors:
      - family-names: Other
        given-names: Author
"""


def _load_s13():
    spec = importlib.util.spec_from_file_location("s13", REPO / "scripts" / "13_make_zenodo_deposit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def s13(tmp_path, monkeypatch):
    """scripts/13 working on a synthetic repository in tmp_path/repo, without git (file lists from CODE_DIRS)."""
    mod = _load_s13()
    monkeypatch.setattr(mod, "ROOT", tmp_path / "repo")

    def no_git(*args, **kwargs):
        raise OSError("no git in this test")

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run=no_git, CalledProcessError=subprocess.CalledProcessError))
    return mod


def _repo(s13) -> Path:
    """Every input of the deposit, as small synthetic files."""
    files: dict[str, str | bytes] = {
        "scripts/01_download_data.py": "print('download')\n",
        "src/etcomment/__init__.py": "",
        "README.md": "# readme\n",
        "CITATION.cff": CFF,
        "config/go26_sites.csv": "site\nUS-AAA\n",
        "data/external/golkar_deoliveira_2026/fig8_US-UiC.png": b"\x89PNG fig8",
        "data/raw/ameriflux_public/provenance.json": "{}\n",
        "data/raw/ameriflux_public/siteinfo_US-AAA.html":
            '<td>PI: </td><td>Jane Doe <a href="[email removed]">[email removed]</a></td>\n',
        "figures/.gitkeep": "",
        "figures/site_plots/png_bokeh/US-ARM_plots.png": b"snapshot",
        "figures/qc/.DS_Store": b"finder",
    }
    files.update({f"results/{n}": f"col\n{n}\n" for n in s13.FILE_NOTES if n not in s13.OPTIONAL_RESULTS})
    files.update({f"figures/{f}.{e}": f"{f}.{e}".encode() for f in s13.FIGURES for e in ("pdf", "png")})
    files.update({q: q.encode() for q in s13.QC})
    files.update({f"figures/site_plots/html/{s}_plots.html": f"<html>{s}</html>\n" for s in s13.SHARED})
    files.update({f"figures/site_plots/static/{s}_summary.{e}": f"{s}.{e}".encode()
                  for s in s13.SHARED for e in ("pdf", "png")})
    for rel, content in files.items():
        p = s13.ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
    _record(s13)
    return s13.ROOT


def _record(s13, **over) -> None:
    """A record of scripts/14 that covers the OpenET-derived results as they are now."""
    files = {p.relative_to(s13.ROOT).as_posix(): s13.md5(p) for p in sorted((s13.ROOT / "results").glob("openet_api_*"))}
    rec = s13.ROOT / s13.DISCLOSURE_RECORD
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({"passed": True, "complete": True, "files": files, **over}))


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_write_zip_fixes_order_time_permissions_and_creator(s13, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("from a file\n")
    entries = [("z/b.txt", b"bytes\n"), ("a.txt", src)]
    s13.write_zip(tmp_path / "one.zip", entries)
    s13.write_zip(tmp_path / "two.zip", entries[::-1])
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "one.zip") as z:
        assert z.namelist() == ["a.txt", "z/b.txt"] and z.read("a.txt") == b"from a file\n"
        for info in z.infolist():
            assert info.date_time == s13.STAMP and info.compress_type == zipfile.ZIP_DEFLATED
            assert (info.create_system, info.create_version, info.extract_version) == (3, 20, 20)
            assert info.external_attr >> 16 == 0o644
    with pytest.raises(ValueError, match="duplicate"):
        s13.write_zip(tmp_path / "dup.zip", [("a.txt", b"1"), ("a.txt", b"2")])


def test_build_assembles_every_archive(s13, tmp_path):
    root = _repo(s13)
    out = tmp_path / "deposit"
    rows = s13.build(out, "9.9.9")
    names = [r[0] for r in rows]
    assert names == [f"{s13.NAME}-9.9.9.zip", *s13.FOLDERS]
    assert not (tmp_path / ".deposit.building").exists()
    # benchmark_site_plots.zip: figures/site_plots/html/ and static/ under benchmark_site_plots/, plus README.txt
    with zipfile.ZipFile(out / "benchmark_site_plots.zip") as z:
        want = sorted(["benchmark_site_plots/README.txt"]
                      + [f"benchmark_site_plots/html/{s}_plots.html" for s in s13.SHARED]
                      + [f"benchmark_site_plots/static/{s}_summary.{e}" for s in s13.SHARED for e in ("pdf", "png")])
        assert z.namelist() == want
        assert z.read("benchmark_site_plots/README.txt").decode() == s13.BENCHMARK_README
        for n in want[1:]:
            assert z.read(n) == (root / "figures/site_plots" / n.split("/", 1)[1]).read_bytes()
    # figures.zip: no site plots, Playwright snapshots, .gitkeep or Finder files
    with zipfile.ZipFile(out / "figures.zip") as z:
        fig = z.namelist()
    assert len(fig) == 2 * len(s13.FIGURES) + len(s13.QC)
    assert not [n for n in fig if n.startswith("figures/site_plots/") or n.endswith((".gitkeep", ".DS_Store"))]
    # results.zip: the files plus README.txt with one entry per file
    with zipfile.ZipFile(out / "results.zip") as z:
        res = z.namelist()
        guide = z.read("results/README.txt").decode()
    files = sorted(n for n in s13.FILE_NOTES if n not in s13.OPTIONAL_RESULTS)
    assert res == ["results/README.txt", *(f"results/{n}" for n in files)]
    assert all(f"\n{n}\n    scripts/" in guide for n in files)
    # code snapshot under one top folder; MD5SUMS and README table match the archives
    with zipfile.ZipFile(out / f"{s13.NAME}-9.9.9.zip") as z:
        assert z.namelist() == [f"{s13.NAME}-9.9.9/{f}" for f in
                                ("CITATION.cff", "README.md", "scripts/01_download_data.py", "src/etcomment/__init__.py")]
    sums = dict(line.split("  ")[::-1] for line in (out / "MD5SUMS").read_text().splitlines())
    assert sums == {n: hashlib.md5((out / n).read_bytes()).hexdigest() for n in names}
    readme = (out / "README.md").read_text()
    assert all(f"`{h}`" in readme for h in sums.values())
    assert "Jane A. Doe ([0000-0001-2345-6789](https://orcid.org/0000-0001-2345-6789)), Richard Roe\n" in readme
    assert "as the AmeriFlux CC-BY-4.0 data policy asks" in readme
    ack = " ".join(readme.split("## Acknowledgements\n\n", 1)[1].split("\n\n", 1)[0].split())
    assert ack.startswith("Funding provided by U.S. Geological Survey Water Resources Research Institute, OpenET")
    assert "data records: US-ARM," in ack and ack.endswith("U.S. Department of Energy's Office of Science.")
    cite = f"Doe, J.A., & Roe, R. (2026). {s13.TITLE} (Version 9.9.9) [dataset]. Zenodo. https://doi.org/{s13.DOI}"
    assert s13.deposit_citation("9.9.9") == cite and f"\n{cite}\n" in readme and s13.REPO_URL in readme
    # description.html: the README as HTML for the Zenodo record, with only the tags Zenodo keeps
    desc = (out / "description.html").read_text()
    assert desc == s13.description_html(readme) and "<h2>Contents</h2>" in desc
    assert set(re.findall(r"</?([a-z0-9]+)", desc)) <= set(s13.ZENODO_TAGS)
    assert not re.search(r"`|\]\(|^#|\*", re.sub(r"<code>.*?</code>", "", desc), re.M)  # no Markdown left
    assert all(f"<td><code>{h}</code></td>" in desc for h in sums.values())
    assert '<a href="https://orcid.org/0000-0001-2345-6789">0000-0001-2345-6789</a>' in desc
    # deterministic: a second build elsewhere gives the same bytes
    s13.build(tmp_path / "again", "9.9.9")
    again = _snapshot(tmp_path / "again")
    assert {k: v for k, v in again.items()} == _snapshot(out)


def test_inputs_are_checked_before_the_old_deposit_is_touched(s13, tmp_path, monkeypatch):
    root = _repo(s13)
    out = tmp_path / "deposit"
    s13.build(out, "1.0.0")
    (out / "notes.txt").write_text("kept\n")  # not part of the deposit
    before = _snapshot(out)
    # three problems at once: a site plot missing, a results file without a FILE_NOTES entry, an e-mail address
    (root / "figures/site_plots/static/US-Ne1_summary.pdf").unlink()
    (root / "results/new_table.csv").write_text("x\n1\n")
    (root / "data/raw/ameriflux_public/siteinfo_US-BBB.html").write_text('<a href="mailto:pi@example.org">PI</a>')
    with pytest.raises(s13.DepositError) as err:
        s13.build(out, "2.0.0")
    msg = str(err.value)
    assert "US-Ne1_summary.pdf" in msg and "results/new_table.csv" in msg and "siteinfo_US-BBB.html" in msg
    assert _snapshot(out) == before and not (tmp_path / ".deposit.building").exists()
    with pytest.raises(SystemExit) as stop:  # the command line says so, too
        s13.main(["--out", str(out), "--version", "2.0.0"])
    assert "STOPPED" in str(stop.value.code) and "unchanged" in str(stop.value.code)
    assert _snapshot(out) == before
    # a failure while writing (inputs complete) also leaves the old deposit as it was
    (root / "figures/site_plots/static/US-Ne1_summary.pdf").write_bytes(b"pdf")
    (root / "results/new_table.csv").unlink()
    (root / "data/raw/ameriflux_public/siteinfo_US-BBB.html").unlink()
    write = s13.write_zip

    def failing(path, entries):
        if path.name == "figures.zip":
            raise OSError("disk full")
        write(path, entries)

    monkeypatch.setattr(s13, "write_zip", failing)
    with pytest.raises(OSError, match="disk full"):
        s13.build(out, "2.0.0")
    assert _snapshot(out) == before and not (tmp_path / ".deposit.building").exists()
    # a complete build of another version replaces the old archives and keeps other files
    monkeypatch.setattr(s13, "write_zip", write)
    s13.build(out, "2.0.0")
    now = _snapshot(out)
    assert f"{s13.NAME}-2.0.0.zip" in now and f"{s13.NAME}-1.0.0.zip" not in now
    assert now["notes.txt"] == b"kept\n" and now["config.zip"] == before["config.zip"]


def test_deposit_needs_a_current_joint_disclosure_check(s13, tmp_path, capsys):
    root = _repo(s13)
    out = tmp_path / "deposit"
    s13.build(out, "1.0.0")
    before = _snapshot(out)
    (root / "results/openet_api_go26_sites.csv").write_text("col\nchanged after the check\n")
    with pytest.raises(s13.DepositError, match=r"changed since the joint OpenET disclosure check: "
                                               r"results/openet_api_go26_sites.csv: run scripts/14"):
        s13.build(out, "1.0.0")
    _record(s13)
    (root / "results/openet_api_extra.csv").write_text("x\n1\n")  # an OpenET-derived file the check never read
    with pytest.raises(s13.DepositError, match="did not read: results/openet_api_extra.csv"):
        s13.build(out, "1.0.0")
    (root / "results/openet_api_extra.csv").unlink()
    for over, msg in (({"passed": False}, "failed"), ({"complete": False}, "was partial")):
        _record(s13, **over)
        with pytest.raises(s13.DepositError, match=msg):
            s13.build(out, "1.0.0")
    (root / s13.DISCLOSURE_RECORD).unlink()
    with pytest.raises(s13.DepositError, match="disclosure_check.json not found"):
        s13.build(out, "1.0.0")
    assert _snapshot(out) == before
    s13.main(["--out", str(out), "--version", "1.0.0", "--no-disclosure-check"])  # e.g. from an unzipped deposit
    assert "WARNING: --no-disclosure-check" in capsys.readouterr().out


def test_missing_site_plot_folders_and_figures_are_reported(s13, tmp_path):
    root = _repo(s13)
    for f in (root / "figures/site_plots/html").iterdir():
        f.unlink()
    (root / "figures/fig3_site_map.pdf").unlink()
    with pytest.raises(s13.DepositError) as err:
        s13.collect("1.0.0")
    msg = str(err.value)
    assert "figures/site_plots/html/ is missing or empty" in msg and "scripts/03" in msg
    assert "figures/fig3_site_map.pdf" in msg


def test_file_notes_cover_the_results(s13):
    with pytest.raises(s13.DepositError, match="FILE_NOTES.*results/extra.csv"):
        s13.results_readme(["leakage_experiment.csv", "extra.csv"])
    assert all("\n" not in v and re.match(r"scripts/\d\d", v) for v in s13.FILE_NOTES.values())
    assert s13.OPTIONAL_RESULTS <= set(s13.FILE_NOTES)
    # every results file a script names has an entry
    named = {n for p in (REPO / "scripts").glob("[01][0-9]_*.py") if not p.name.startswith("13_")
             for n in re.findall(r"results/([A-Za-z0-9_.-]+\.(?:csv|json))", p.read_text())}
    assert named and named <= set(s13.FILE_NOTES)
    # and, where the pipeline has run (not in CI), every file in results/
    res = REPO / "results"
    if res.is_dir():
        have = {p.name for p in res.iterdir() if p.is_file() and p.name not in s13.SKIP | {"README.txt"}}
        assert have <= set(s13.FILE_NOTES), sorted(have - set(s13.FILE_NOTES))


def test_results_readme_lists_the_base_derivatives_and_openet_sources(s13):
    # every go26_* file scripts/10 writes has an entry, is named in the licence paragraph (cite the BASE DOIs) and,
    # except the table written without BASE files, is optional, so a pipeline without BASE files still builds
    s10 = (REPO / "scripts" / "10_reconstruct_go26_observed_et.py").read_text()
    written = set(re.findall(r'out_dir / "(go26_[a-z0-9_]+\.csv)"', s10))
    go26 = {n for n in s13.FILE_NOTES if n.startswith("go26_")}
    assert written and written == go26
    assert all(f"\n  {n}" in s13.RESULTS_README_HEAD for n in go26)
    assert go26 - s13.OPTIONAL_RESULTS == {"go26_observed_et_reconstruction.csv"}
    # GO26's plotted OpenET line is in the digitized series: attribute OpenET
    assert all("(OpenET, https://openetdata.org)" in s13.FILE_NOTES[f"digitized_{s}.csv"] for s in ("US-UiC", "US-UTJ"))
    assert not [n for n in s13.FILE_NOTES if "monthly" in n and n.startswith("openet_api_")]  # no monthly OpenET files


def test_author_line_shows_orcid_ids_from_citation_cff(tmp_path):
    s13 = _load_s13()
    cff = tmp_path / "CITATION.cff"
    cff.write_text(CFF)
    assert [a["family-names"] for a in s13.citation_authors(cff)] == ["Doe", "Roe"]
    assert s13.author_line(cff) == ("Jane A. Doe ([0000-0001-2345-6789](https://orcid.org/0000-0001-2345-6789)), "
                                    "Richard Roe")
    real = re.sub(r" \(\[[^]]*\]\([^)]*\)\)", "", s13.author_line(REPO / "CITATION.cff"))  # without ORCID links
    assert real == s13.AUTHORS
    assert s13.author_line(tmp_path / "missing.cff") == s13.AUTHORS


# -- reproducible PDFs (etcomment.style) ---------------------------------------------------------------------
def test_rerun_with_fixed_hash_seed_only_for_scripts_run_without_it(monkeypatch):
    calls = []

    class Replaced(Exception):
        pass

    def fake_execve(path, argv, env):
        calls.append((path, argv, env))
        raise Replaced

    monkeypatch.setattr(st.os, "execve", fake_execve)
    monkeypatch.setattr(st.os, "name", "posix")
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    st.rerun_with_fixed_hash_seed("figure_script")  # imported (tests): nothing happens
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    st.rerun_with_fixed_hash_seed("__main__")  # already fixed: nothing happens
    assert calls == []
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(Replaced):
        st.rerun_with_fixed_hash_seed("__main__")
    path, argv, env = calls[0]
    assert path == sys.executable and argv == [sys.executable, *sys.orig_argv[1:]]
    assert env["PYTHONHASHSEED"] == "0" and env.get("PATH") == os.environ.get("PATH")


def test_figure_script_reruns_with_hash_seed_zero_and_writes_pdf_without_date(tmp_path):
    script = tmp_path / "fig.py"
    script.write_text(
        "import os, sys\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from etcomment import style as st\n"
        "def main():\n"
        "    st.rerun_with_fixed_hash_seed(__name__)\n"
        "    print(os.environ.get('PYTHONHASHSEED'), sys.flags.hash_randomization, sys.argv[1:])\n"
        "    fig, ax = plt.subplots(figsize=(2, 1))\n"
        "    ax.plot([0, 1], [0, 1])\n"
        "    for ext in ('pdf', 'png'):\n"
        "        fig.savefig(sys.argv[1] + '.' + ext, metadata=st.PDF_METADATA if ext == 'pdf' else None)\n"
        "if __name__ == '__main__':\n"
        "    main()\n")
    env = {**os.environ, "PYTHONHASHSEED": "7", "MPLBACKEND": "Agg"}
    env.pop("SOURCE_DATE_EPOCH", None)
    run = subprocess.run([sys.executable, str(script), str(tmp_path / "out")], env=env, capture_output=True,
                         text=True, timeout=120, check=True)
    assert run.stdout.split("\n")[0] == f"0 0 {[str(tmp_path / 'out')]!r}"
    pdf = (tmp_path / "out.pdf").read_bytes()
    assert b"/CreationDate" not in pdf and b"/Producer" in pdf
    assert (tmp_path / "out.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_description_html_converts_the_readme_markdown(s13):
    md = ("# Title\n\nVersion 1 · DOI [10.1/x](https://doi.org/10.1/x) · Code: https://example.org/repo\n\n"
          "## Section\n\nA *journal* name, `a_*` code & more\n(see https://doi.org/10.2/y).\n\n"
          "| File | MD5 |\n|---|---|\n| `a.zip` | `abc` |\n\n"
          "1. First, wrapped\n   onto two lines:\n   - nested one\n     wrapped;\n   - nested two\n2. Second\n\n"
          "- bullet\n")
    assert s13.description_html(md) == (
        '<p>Version 1 · DOI <a href="https://doi.org/10.1/x">10.1/x</a> · Code: '
        '<a href="https://example.org/repo">https://example.org/repo</a></p>\n'
        "<h2>Section</h2>\n"
        '<p>A <em>journal</em> name, <code>a_*</code> code &amp; more (see '
        '<a href="https://doi.org/10.2/y">https://doi.org/10.2/y</a>).</p>\n'
        "<table><tbody><tr><th>File</th><th>MD5</th></tr><tr><td><code>a.zip</code></td><td><code>abc</code></td></tr>"
        "</tbody></table>\n"
        "<ol><li>First, wrapped onto two lines:<ul><li>nested one wrapped;</li><li>nested two</li></ul></li>"
        "<li>Second</li></ol>\n"
        "<ul><li>bullet</li></ul>\n")
    # a quote or closing punctuation right after a bare URL is not part of the link
    assert s13._inline('attribute "OpenET, https://openetdata.org" wherever') == (
        'attribute "OpenET, <a href="https://openetdata.org">https://openetdata.org</a>" wherever')


def test_figure_1_of_the_comment_is_required(s13, tmp_path):
    root = _repo(s13)
    out = tmp_path / "deposit"
    s13.build(out, "1.0.0")
    before = _snapshot(out)
    (root / "figures/fig_table1_reconstruction.png").unlink()
    with pytest.raises(s13.DepositError, match="figures/fig_table1_reconstruction.png"):
        s13.build(out, "1.0.1")
    assert _snapshot(out) == before


def test_rebuild_from_an_unzipped_deposit_reads_benchmark_site_plots(s13, tmp_path):
    root = _repo(s13)
    s13.build(tmp_path / "first", "1.0.0")
    with zipfile.ZipFile(tmp_path / "first" / "benchmark_site_plots.zip") as z:
        z.extractall(root)  # the deposit layout: benchmark_site_plots/ at the root ...
    for part in s13.SITE_PLOT_PARTS:  # ... and no figures/site_plots/
        for f in (root / s13.SITE_PLOTS / part).iterdir():
            f.unlink()
    s13.build(tmp_path / "again", "1.0.0")
    first, again = (zipfile.ZipFile(tmp_path / d / "benchmark_site_plots.zip") for d in ("first", "again"))
    assert sorted(again.namelist()) == sorted(first.namelist())
    assert all(again.read(n) == first.read(n) for n in first.namelist())


def test_site_plots_come_from_figures_when_both_folders_exist(s13, tmp_path):
    root = _repo(s13)
    s13.build(tmp_path / "first", "1.0.0")
    with zipfile.ZipFile(tmp_path / "first" / "benchmark_site_plots.zip") as z:
        z.extractall(root)  # a stale copy of the deposit layout next to figures/site_plots/
    (root / "benchmark_site_plots/static/US-ARM_summary.png").write_bytes(b"stale")
    (root / "benchmark_site_plots/html/US-ARM_plots.html").write_text("stale")
    s13.build(tmp_path / "again", "1.0.0")
    with zipfile.ZipFile(tmp_path / "again" / "benchmark_site_plots.zip") as z:
        assert z.read("benchmark_site_plots/static/US-ARM_summary.png") == b"US-ARM.png"
        assert z.read("benchmark_site_plots/html/US-ARM_plots.html") == b"<html>US-ARM</html>\n"
