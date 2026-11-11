"""Reusable data-processing utilities for the multi-task characterization project.

Submodules:
    config    project paths + shared constants
    io        unzip the raw delivery, optional output paths (save switches of the notebooks)
    tensile   parse tensile workbook, derive metrics / summaries
    instron   parse Instron raw CSV exports (DoE delivery), derive metrics, compare to DoE
    rheology  parse rheometer workbook, power-law fit, window stats
    doe       optimized Latin-hypercube design of experiments
    plots     figure helpers (each returns a Matplotlib Figure)
"""
from . import config, io, tensile, instron, rheology, doe, plots

__all__ = ["config", "io", "tensile", "instron", "rheology", "doe", "plots"]
