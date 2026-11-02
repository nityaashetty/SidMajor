"""Put the approach folder on sys.path so `core` and `model` import cleanly.

Phase-1 defect this fixes:
    every script began with sys.path.append(r"D:\\Major\\src\\datasets") and
    eeg_dataset.py used a bare `from feature_extraction import ...`, so the code only
    worked from one absolute location on one machine. Here the path is derived from
    __file__, so the folder can be copied anywhere on any of the three devices.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
