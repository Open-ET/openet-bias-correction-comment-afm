"""Download and verify the public datasets used by the analysis.

  * Volk et al. (2023c) flux_ET_dataset.zip, Zenodo record 7636781 (183 MB)
  * Volk et al. (2023d) OpenET_PhaseII_model_ET_dataset.zip, Zenodo record 10119477

Both are CC BY 4.0.  A zip that is already on disk is used instead of a
download if its md5 matches the sum published on Zenodo, whatever its name,
in data/raw/ or data/external/ (e.g. data/external/flux_ET_dataset_Volk_2023.zip).
Verified zips are extracted to data/raw/.

Usage:
  python scripts/01_download_data.py            # find or download, verify, extract
  python scripts/01_download_data.py --verify   # only verify and extract zips already present
"""
from __future__ import annotations

import argparse
import http.client
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from etcomment import benchmark as bm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEARCH = [RAW, ROOT / "data" / "external"]


def url(rec: dict) -> str:
    return f"https://zenodo.org/records/{rec['record']}/files/{rec['file']}?download=1"


def find_local(rec: dict) -> Path | None:
    """A zip on disk with the published md5 (the expected name is checked first)."""
    cands = [d / rec["file"] for d in SEARCH] + sorted(p for d in SEARCH if d.is_dir() for p in d.glob("*.zip"))
    seen = set()
    for p in cands:
        p = p.resolve() if p.exists() else p
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        if bm.md5(p) == rec["md5"]:
            return p
    return None


def download(rec: dict, dst: Path, tries: int = 4, timeout_s: float = 120.0) -> bool:
    """Stream the file to ``<dst>.part`` and rename it only if the md5 matches."""
    part = dst.with_name(dst.name + ".part")
    try:
        for k in range(1, tries + 1):
            try:
                print(f"downloading {url(rec)} (attempt {k})", flush=True)
                with urllib.request.urlopen(url(rec), timeout=timeout_s) as r, open(part, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                if bm.md5(part) == rec["md5"]:
                    part.replace(dst)
                    return True
                print(f"{rec['file']}: md5 mismatch after download")
            except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets, truncated reads
                print(f"{rec['file']}: download failed ({e!r})")
            if k < tries:
                time.sleep(5 * k)
        return False
    finally:
        part.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for rec in bm.ZENODO.values():
        src = find_local(rec)
        if src is None and not args.verify:
            src = RAW / rec["file"]
            if not download(rec, src):
                print(f"{rec['file']}: no verified copy (expected md5 {rec['md5']})")
                ok = False
                continue
        if src is None:
            print(f"{rec['file']}: no zip with md5 {rec['md5']} in {', '.join(str(d) for d in SEARCH)}")
            ok = False
            continue
        print(f"{rec['file']}: md5 {rec['md5']} OK ({src})")
        bm.extract(src, RAW)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
