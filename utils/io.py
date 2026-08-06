"""Unpack the raw delivery zip."""
from __future__ import annotations

import zipfile
from pathlib import Path


def extract_raw(raw_dir: Path) -> tuple[Path, Path]:
    """Unzip the delivery (skipping ``~$`` Excel lock files) into ``raw_dir/extracted``
    and return ``(tensile_xlsx, viscosity_xlsx)``.
    """
    extract = raw_dir / "extracted"
    extract.mkdir(exist_ok=True)

    zips = list(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip found in {raw_dir}")
    with zipfile.ZipFile(zips[0]) as zf:
        for name in zf.namelist():
            if not Path(name).name.startswith("~$"):
                zf.extract(name, extract)

    xlsx = sorted(p for p in extract.glob("*.xlsx") if not p.name.startswith("~$"))
    tensile = next(p for p in xlsx if "Trial Data" in p.name)
    viscosity = next(p for p in xlsx if "viscosity" in p.name.lower())
    return tensile, viscosity
