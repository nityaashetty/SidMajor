"""Recording-level clinical EEG biomarkers, computed per behavioural state.

Eyes-closed (EC) resting EEG is where clinical slowing is read, so most markers come from EC.
Two families need the event annotations and are what earlier CAUEEG work could not use:
  * alpha reactivity  - posterior alpha suppression from EC to eyes-open (EO); reduced in AD
  * photic driving    - occipital power at the flash frequency vs neighbouring bins
CEFA-Net's 8 clinical ratios are included (TAR, DAR, slow/fast, fronto-posterior alpha,
alpha/beta asymmetry, frontal theta, posterior alpha), plus alpha peak frequency, entropy,
Hjorth parameters and inter-hemispheric / fronto-posterior coherence.
Missing states give NaN; the matching has_* indicator records it.
"""
import numpy as np

from ..data.events import STATE_EC, STATE_EO, STATE_UNKNOWN, window_starts
from ..data.store import load_signal, load_timeline
from .spectral import (BANDS, FRONTO_POSTERIOR, HOMOLOGOUS, POSTERIOR, REGIONS, TOTAL, band_power,
                       coherence, idx, mean_psd, windows)

EPS = 1e-12


def _pick(starts, step, max_n):
    """Non-overlapping-ish subset (spaced by `step` s), evenly thinned to at most max_n."""
    out, last = [], -1e9
    for s in starts:
        if s - last >= step:
            out.append(s)
            last = s
    if len(out) > max_n:
        out = [out[i] for i in np.linspace(0, len(out) - 1, max_n).round().astype(int)]
    return out


def _state_windows(x, fs, state, clean, st, cfg):
    starts = window_starts(state, clean, st, cfg["window_sec"])
    return windows(x, fs, _pick(starts, cfg["step_sec"], cfg["max_windows"]), cfg["window_sec"])


def _rel(f, psd, band, chans):
    tot = band_power(f, psd[chans], TOTAL)
    return band_power(f, psd[chans], band) / (tot + EPS)


def _alpha_cog(f, p):
    m = (f >= 7.0) & (f <= 13.0)
    return float((f[m] * p[m]).sum() / (p[m].sum() + EPS))


def _hjorth(w):
    d1 = np.diff(w, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = w.var(axis=-1), d1.var(axis=-1), d2.var(axis=-1)
    mob = np.sqrt(v1 / (v0 + EPS))
    comp = np.sqrt(v2 / (v1 + EPS)) / (mob + EPS)
    return float(np.mean(mob)), float(np.mean(comp))


def ec_features(w, fs):
    f, psd = mean_psd(w, fs)
    allc = list(range(psd.shape[0]))
    post = idx(POSTERIOR)
    out = {}
    for b, rng in BANDS.items():
        for r, chans in REGIONS.items():
            out[f"ec_rel_{b}_{r}"] = float(np.log(_rel(f, psd, rng, idx(chans)).mean() + EPS))
    bp = {b: band_power(f, psd, rng) for b, rng in BANDS.items()}  # [C] each
    tot = {b: v.sum() for b, v in bp.items()}
    out["ec_theta_alpha_ratio"] = float(np.log(tot["theta"] / (tot["alpha"] + EPS)))
    out["ec_delta_alpha_ratio"] = float(np.log(tot["delta"] / (tot["alpha"] + EPS)))
    out["ec_slow_fast_ratio"] = float(np.log((tot["delta"] + tot["theta"]) / (tot["alpha"] + tot["beta"] + EPS)))
    out["ec_fronto_posterior_alpha"] = float(np.log(bp["alpha"][idx(REGIONS["frontal"])].mean()
                                                    / (bp["alpha"][post].mean() + EPS)))
    for b in ("alpha", "beta"):
        out[f"ec_{b}_asymmetry"] = float(np.mean([np.log(bp[b][idx([r])[0]] + EPS) - np.log(bp[b][idx([l])[0]] + EPS)
                                                  for l, r in HOMOLOGOUS]))
    p_post = psd[post].mean(axis=0)
    out["ec_alpha_cog_hz"] = _alpha_cog(f, p_post)
    m = (f >= 6.0) & (f <= 13.0)
    out["ec_alpha_peak_hz"] = float(f[m][np.argmax(p_post[m])])
    m = (f >= TOTAL[0]) & (f < TOTAL[1])
    p = psd[allc][:, m].mean(axis=0)
    pn = p / (p.sum() + EPS)
    out["ec_spectral_entropy"] = float(-(pn * np.log(pn + EPS)).sum() / np.log(len(pn)))
    out["ec_median_freq_hz"] = float(f[m][np.searchsorted(np.cumsum(pn), 0.5)])
    out["ec_log_total_power"] = float(np.log(band_power(f, psd, TOTAL).mean() + EPS))
    out["ec_hjorth_mobility"], out["ec_hjorth_complexity"] = _hjorth(w)
    for b in ("theta", "alpha", "beta"):
        out[f"ec_coh_interhemi_{b}"] = coherence(w, fs, HOMOLOGOUS, BANDS[b])
        out[f"ec_coh_frontopost_{b}"] = coherence(w, fs, FRONTO_POSTERIOR, BANDS[b])
    return out, f, psd


def eo_features(w, fs):
    f, psd = mean_psd(w, fs)
    allc = list(range(psd.shape[0]))
    out = {f"eo_rel_{b}": float(np.log(_rel(f, psd, rng, allc).mean() + EPS)) for b, rng in BANDS.items()}
    out["eo_alpha_cog_hz"] = _alpha_cog(f, psd[idx(POSTERIOR)].mean(axis=0))
    return out, f, psd


def photic_driving(x, fs, clean, blocks):
    """Mean log SNR of occipital power at the flash frequency (blocks <= 25 Hz, >= 4 s usable)."""
    occ = idx(["O1", "O2"])
    snrs = []
    for t0, t1, hz in blocks:
        if hz > 25 or hz < 1:
            continue
        a, b = int(np.ceil(t0)) + 2, int(np.floor(t1)) - 1
        if b - a < 4:
            continue
        seg = np.asarray(x[occ, a * fs:b * fs], dtype=np.float32)
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[-1]), axis=-1)) ** 2
        f = np.fft.rfftfreq(seg.shape[-1], 1.0 / fs)
        p = spec.mean(axis=0)
        on = (np.abs(f - hz) <= 0.5)
        near = (np.abs(f - hz) > 1.0) & (np.abs(f - hz) <= 3.0)
        if on.any() and near.any():
            snrs.append(np.log(p[on].mean() / (p[near].mean() + EPS) + EPS))
    return (float(np.mean(snrs)), len(snrs)) if snrs else (np.nan, 0)


def recording_features(serial, store_root, fs, cfg):
    """All biomarkers for one recording -> dict (NaN where a state is missing)."""
    x = load_signal(store_root, serial)
    state, clean, blocks = load_timeline(store_root, serial)
    feats = {"serial": serial}

    w_ec = _state_windows(x, fs, state, clean, STATE_EC, cfg)
    ec_source = "eyes_closed"
    if len(w_ec) < 8:  # too little labelled EC: fall back to unlabelled clean resting EEG
        w_ec = _state_windows(x, fs, state, clean, STATE_UNKNOWN, cfg)
        ec_source = "unknown"
    if len(w_ec) < 8:
        starts = window_starts(state, clean, None, cfg["window_sec"])
        w_ec = windows(x, fs, _pick(starts, cfg["step_sec"], cfg["max_windows"]), cfg["window_sec"])
        ec_source = "any"
    ec, f, psd_ec = ec_features(w_ec, fs)
    feats.update(ec)

    w_eo = _state_windows(x, fs, state, clean, STATE_EO, cfg)
    if len(w_eo) >= 4:
        eo, _, psd_eo = eo_features(w_eo, fs)
        feats.update(eo)
        post = idx(POSTERIOR)
        a_ec = band_power(f, psd_ec[post], BANDS["alpha"]).mean()
        a_eo = band_power(f, psd_eo[post], BANDS["alpha"]).mean()
        feats["alpha_reactivity"] = float(np.log((a_ec + EPS) / (a_eo + EPS)))
        feats["has_eo"] = 1.0
    else:
        feats.update({k: np.nan for k in ("eo_rel_delta", "eo_rel_theta", "eo_rel_alpha", "eo_rel_beta",
                                          "eo_alpha_cog_hz", "alpha_reactivity")})
        feats["has_eo"] = 0.0

    snr, n_blocks = photic_driving(x, fs, clean, blocks)
    feats["photic_driving_snr"] = snr
    feats["has_photic"] = float(n_blocks > 0)

    feats["_ec_source"] = ec_source
    feats["_n_ec_windows"] = int(len(w_ec))
    feats["_n_eo_windows"] = int(len(w_eo))
    feats["_n_photic_blocks"] = int(n_blocks)
    return feats
