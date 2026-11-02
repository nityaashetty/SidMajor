"""Shared core for the Phase-2 approach folders.

This package is BYTE-IDENTICAL in Approach1_EEGNet/, Approach4_TemporalMIL/ and
Approach6_STFTCNN/. Only model.py and config.yaml differ. If you change anything here,
copy it to the other two folders before comparing results, otherwise the three devices
are no longer measuring the same thing.

Verify with:  python scripts/00_verify.py   (prints a core/ checksum)
"""
from . import blocks, cache, checkpoint, config, data, engine, metrics, splits  # noqa: F401

__all__ = ["blocks", "cache", "checkpoint", "config", "data", "engine",
           "metrics", "splits"]
