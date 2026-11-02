"""Spectral primitives on windowed EEG: PSDs, band powers, coherence."""
import numpy as np
from scipy.signal import welch

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
TOTAL = (1.0, 30.0)  # 30-45 Hz left out of relative power: dominated by muscle artifact

CH = ["Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2",
      "F7", "T3", "T5", "F8", "T4", "T6", "FZ", "CZ", "PZ"]
CI = {c: i for i, c in enumerate(CH)}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ"],
    "central": ["C3", "C4", "CZ"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "PZ"],
    "occipital": ["O1", "O2"],
}
POSTERIOR = ["P3", "P4", "PZ", "O1", "O2"]
HOMOLOGOUS = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("C3", "C4"),
              ("T3", "T4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]  # (left, right)
FRONTO_POSTERIOR = [("F3", "P3"), ("F4", "P4"), ("FZ", "PZ"), ("F3", "O1"), ("F4", "O2")]


def idx(names):
    return [CI[n] for n in names]


def windows(x, fs, starts_sec, win_sec):
    """x [C, n] -> [W, C, win*fs] for the given start seconds."""
    L = int(win_sec * fs)
    starts = [int(s * fs) for s in starts_sec if int(s * fs) + L <= x.shape[1]]
    if not starts:
        return np.zeros((0, x.shape[0], L), np.float32)
    return np.stack([np.asarray(x[:, s:s + L], dtype=np.float32) for s in starts])


def mean_psd(w, fs):
    """Average PSD over windows: [W, C, T] -> freqs [F], psd [C, F] (uV^2/Hz)."""
    f, p = welch(w, fs=fs, nperseg=w.shape[-1], window="hann", axis=-1, detrend="constant")
    return f, p.mean(axis=0)


def band_power(f, psd, band):
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return psd[..., m].sum(axis=-1) * (f[1] - f[0])


def coherence(w, fs, pairs, band):
    """Magnitude-squared coherence averaged over the band, per channel pair, from windows [W, C, T]."""
    T = w.shape[-1]
    spec = np.fft.rfft(w * np.hanning(T), axis=-1)  # [W, C, F]
    f = np.fft.rfftfreq(T, 1.0 / fs)
    m = (f >= band[0]) & (f < band[1])
    spec = spec[..., m]
    out = []
    for a, b in pairs:
        sa, sb = spec[:, CI[a]], spec[:, CI[b]]
        sxy = (sa * np.conj(sb)).mean(axis=0)
        sxx = (np.abs(sa) ** 2).mean(axis=0)
        syy = (np.abs(sb) ** 2).mean(axis=0)
        out.append(np.mean(np.abs(sxy) ** 2 / (sxx * syy + 1e-12)))
    return float(np.mean(out))
