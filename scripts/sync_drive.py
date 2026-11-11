#!/usr/bin/env python
"""Pull the lab's tensile data from Google Drive into a new raw_data/<MM_DD_YY>/extracted/ when it changed.

Run daily by cron (see the crontab) or by hand: /data/zhq7531/envs/ml_gp_env/bin/python scripts/sync_drive.py
One-way and additive: nothing on Drive is touched and earlier snapshots are never changed. The whole folder is
downloaded to a temporary place, one file at a time with a pause (gentle on Drive), and compared by hash with the newest
snapshot, so edited files are caught as well as new ones; with no change the download is thrown away. The folder is
link-shared, so no sign-in is needed. Prints one
status line (cron appends it to raw_data/drive_sync.log); each snapshot has a CHANGES.txt.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import gdown

FOLDER_ID = "1Eb14X95O8uSdWR6x_c9M-NI8rsgYSfZ7"   # UIC-NU NASA Data / nu-formulations-tensile test
EXTRA_FILES = {"printability/Sample Warping.xlsx": "1A6PrB7DS0xX4-DAJ4qUCYe_N9FfPgxHm"}   # single files elsewhere on the Drive
RAW = Path(__file__).resolve().parents[1] / "raw_data"
DATED = re.compile(r"^(\d\d)_(\d\d)_(\d\d)(?:_(\d+))?$")
GDOWN_FOLDER_LIMIT = 50   # gdown lists at most 50 files per Drive folder
PAUSE_S = 1.0             # between file downloads


def fetch_file(file_id: str, dest: Path) -> None:
    """Download one link-shared Drive file (gdown's own download gets refused after a few runs; this does not);
    an HTML answer means Drive refused it."""
    with urllib.request.urlopen(f"https://drive.google.com/uc?export=download&id={file_id}", timeout=60) as r:
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"Drive did not serve file {dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.read())


def manifest(root: Path) -> dict[str, str]:
    """Relative path -> sha256 of every file under ``root``."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def latest_snapshot() -> Path | None:
    """The newest raw_data/<MM_DD_YY>[_k] folder holding an extracted/ tree."""
    def key(d: Path):
        m = DATED.match(d.name)
        return (int(m[3]), int(m[1]), int(m[2]), int(m[4] or 1))
    found = [d for d in RAW.iterdir() if DATED.match(d.name) and (d / "extracted").is_dir()]
    return max(found, key=key) if found else None


def new_snapshot_dir(now: datetime) -> Path:
    base, k = now.strftime("%m_%d_%y"), 2
    out = RAW / base
    while out.exists():
        out, k = RAW / f"{base}_{k}", k + 1
    return out


def main() -> int:
    now = datetime.now()
    stamp = now.isoformat(timespec="seconds")
    tmp = RAW / ".drive_sync_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        files = gdown.download_folder(id=FOLDER_ID, output=str(tmp), quiet=True, use_cookies=False, remaining_ok=True,
                                      skip_download=True)
        if not files:
            print(f"{stamp} FAILED: the folder listing came back empty; nothing changed")
            return 1
        # Any failed download raises, so a partial download never becomes a snapshot.
        for rel, file_id in [(f.path, f.id) for f in files] + list(EXTRA_FILES.items()):
            fetch_file(file_id, tmp / rel)
            time.sleep(PAUSE_S)
        new = manifest(tmp)
        full = [k for k, n in Counter(Path(p).parent.as_posix() for p in new).items() if n >= GDOWN_FOLDER_LIMIT]
        note = f"; WARNING: may be cut off at {GDOWN_FOLDER_LIMIT} files: {full}" if full else ""

        last = latest_snapshot()
        old = manifest(last / "extracted") if last else {}
        added = sorted(set(new) - set(old))
        changed = sorted(k for k in set(new) & set(old) if new[k] != old[k])
        removed = sorted(set(old) - set(new))
        if not (added or changed or removed):
            print(f"{stamp} no change ({len(new)} files, same as {last.name}){note}")
            return 0

        out = new_snapshot_dir(now)
        out.mkdir()
        tmp.rename(out / "extracted")
        lines = [f"Compared with {last.name if last else '(nothing)'} on {stamp}"]
        for label, items in (("added", added), ("changed", changed), ("removed from Drive", removed)):
            lines += [f"\n{label} ({len(items)}):"] + [f"  {k}" for k in items]
        (out / "CHANGES.txt").write_text("\n".join(lines) + "\n")
        print(f"{stamp} NEW DATA in raw_data/{out.name}: {len(added)} added, {len(changed)} changed, "
              f"{len(removed)} removed{note}")
        return 0
    except Exception as e:                  # e.g. Drive rate limit, no network: report and try again tomorrow
        print(f"{stamp} FAILED: {type(e).__name__}: {str(e).strip().splitlines()[0]}; nothing changed")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
