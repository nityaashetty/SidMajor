"""Raw EDF reading: the 19 scalp EEG channels only (EKG and Photic channels are dropped)."""
import numpy as np
import pyedflib

EEG_CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2",
                "F7", "T3", "T5", "F8", "T4", "T6", "FZ", "CZ", "PZ"]


def read_eeg(path):
    """Return (signal [19, n] float32 in uV, sampling rate)."""
    f = pyedflib.EdfReader(str(path))
    try:
        labels = [l.strip().upper() for l in f.getSignalLabels()]
        idx = [labels.index(f"{c.upper()}-AVG") for c in EEG_CHANNELS]
        rates = {float(f.getSampleFrequency(i)) for i in idx}
        if len(rates) != 1:
            raise ValueError(f"{path}: mixed sampling rates {rates}")
        x = np.stack([f.readSignal(i) for i in idx]).astype(np.float32)
    finally:
        f.close()
    return x, rates.pop()
