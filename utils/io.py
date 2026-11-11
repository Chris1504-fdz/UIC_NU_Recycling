"""Unpack the raw delivery zip; resolve the (optional) output paths of the notebooks."""
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


def extract_zip(raw_dir: Path) -> Path:
    """Unzip the (single) delivery archive in ``raw_dir`` into ``raw_dir/extracted`` and
    return that folder. Generic version of :func:`extract_raw` for deliveries that are a
    folder tree of CSVs rather than two workbooks. Excel lock files (``~$``) are skipped.
    A snapshot from ``scripts/sync_drive.py`` has no zip: its ``extracted`` folder is returned as is.
    """
    extract = raw_dir / "extracted"
    zips = sorted(raw_dir.glob("*.zip"))
    if not zips and extract.is_dir():
        return extract
    extract.mkdir(exist_ok=True)
    if not zips:
        raise FileNotFoundError(f"No .zip found in {raw_dir}")
    with zipfile.ZipFile(zips[0]) as zf:
        for name in zf.namelist():
            if not Path(name).name.startswith("~$"):
                zf.extract(name, extract)
    return extract


def save_path(folder: Path, name: str, save: bool = True, ext: str = "") -> Path | None:
    """Where a notebook cell writes an output: ``folder / name`` when ``save`` is set, else
    None -- the plot helpers (and the cells) then write nothing. ``ext`` is appended when
    ``name`` carries no extension of its own, so ``"fig"`` -> ``fig.png`` but ``"fig.pdf"`` stays.
    """
    if not save:
        return None
    path = Path(folder) / name
    if not path.suffix:
        path = path.with_name(path.name + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
