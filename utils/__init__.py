"""Reusable data-processing utilities for the multi-task characterization project.

Submodules:
    config    project paths + shared constants
    io        unzip the raw delivery
    tensile   parse tensile workbook, derive metrics / summaries
    rheology  parse rheometer workbook, power-law fit, window stats
    doe       optimized Latin-hypercube design of experiments
    plots     figure helpers (each returns a Matplotlib Figure)
"""
from . import config, io, tensile, rheology, doe, plots

__all__ = ["config", "io", "tensile", "rheology", "doe", "plots"]
