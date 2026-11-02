import numpy as np
from scipy import signal as scipy_signal

SAMPLE_RATE = 200

DELTA = (0.5, 4)
THETA = (4, 8)
ALPHA = (8, 13)
BETA  = (13, 30)
BANDS = [DELTA, THETA, ALPHA, BETA]

def bandpower(signal, fs, band):
    low, high = band
    freqs, psd = scipy_signal.welch(signal, fs, nperseg=256)
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapezoid(psd[idx], freqs[idx])

def compute_stream1(signal):
    """
    S1 — Frequency slowing.
    Theta/alpha ratio + absolute band powers (delta, theta, alpha, beta)
    per channel. Output shape: (95,) — 19 channels × 5 features
    """
    features = []
    for ch in range(signal.shape[0]):
        delta = bandpower(signal[ch], SAMPLE_RATE, DELTA)
        theta = bandpower(signal[ch], SAMPLE_RATE, THETA)
        alpha = bandpower(signal[ch], SAMPLE_RATE, ALPHA)
        beta  = bandpower(signal[ch], SAMPLE_RATE, BETA)
        ratio = theta / (alpha + 1e-8)
        features.extend([ratio, delta, theta, alpha, beta])
    return np.array(features, dtype=np.float32)

def compute_stream2(signal):
    """
    Multi-band coherence for all 171 electrode pairs
    across 4 bands = 684 features — shape (684,)
    """
    n_channels = signal.shape[0]
    coherences = []

    for i in range(n_channels):
        for j in range(i+1, n_channels):
            f, coh = scipy_signal.coherence(
                signal[i], signal[j],
                fs=SAMPLE_RATE, nperseg=256
            )
            for band in BANDS:
                idx = np.logical_and(f >= band[0], f <= band[1])
                coherences.append(np.mean(coh[idx]))

    return np.array(coherences, dtype=np.float32)

def compute_stream3(signal):
    """
    Spectral entropy per channel — shape (19,)
    """
    entropies = []
    for ch in range(signal.shape[0]):
        freqs, psd = scipy_signal.welch(
            signal[ch], SAMPLE_RATE, nperseg=256
        )
        psd_norm = psd / (psd.sum() + 1e-8)
        ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
        entropies.append(ent)
    return np.array(entropies, dtype=np.float32)

def compute_stream4(signal):
    """
    Multi-band hemispheric asymmetry for 8 symmetric pairs
    across 4 bands = 32 features — shape (32,)
    """
    pairs = [(0,5),(1,6),(2,7),(3,8),(4,9),(10,13),(11,14),(12,15)]

    asymmetries = []
    for left, right in pairs:
        for band in BANDS:
            left_power  = bandpower(signal[left],  SAMPLE_RATE, band)
            right_power = bandpower(signal[right], SAMPLE_RATE, band)
            asym = ((left_power - right_power) /
                    (left_power + right_power + 1e-8))
            asymmetries.append(asym)

    return np.array(asymmetries, dtype=np.float32)

def extract_all_features(signal):
    return {
        "s1_freq_slowing": compute_stream1(signal),
        "s2_coherence":    compute_stream2(signal),
        "s3_complexity":   compute_stream3(signal),
        "s4_asymmetry":    compute_stream4(signal)
    }

if __name__ == "__main__":
    dummy = np.random.randn(19, 2000).astype(np.float32)
    features = extract_all_features(dummy)
    for name, vec in features.items():
        print(f"{name}: shape={vec.shape}")