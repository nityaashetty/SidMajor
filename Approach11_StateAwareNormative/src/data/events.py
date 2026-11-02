"""Event annotations -> per-second behavioural-state timeline and clean mask.

Two CAUEEG facts this module handles (both verified on all 1187 dementia-task recordings):
  * Event sample indices (200 Hz) count wall-clock time INCLUDING pauses, while the EDF
    signal has the paused stretches cut out. Index -> signal position therefore needs the
    cumulative paused duration subtracted. After correction, the final 'Paused' event lands
    within 0-6 s of the signal end (median 3 s), so state boundaries get a guard band.
  * The EDF joins the pieces on either side of a pause, leaving a discontinuity that is
    masked out (filter ringing).
"""
import re

import numpy as np

STATE_UNKNOWN, STATE_EC, STATE_EO, STATE_PHOTIC, STATE_HV = 0, 1, 2, 3, 4
STATE_NAMES = ["unknown", "eyes_closed", "eyes_open", "photic", "hyperventilation"]
STATE_IDS = {n: i for i, n in enumerate(STATE_NAMES)}

_LONG_ARTIFACT = re.compile(r"sleep|drows|snor")
_ELECTRODE = re.compile(r"check")
_ARTIFACT = re.compile(r"artifact|move|swallow|talk|cough|chew|couch|mouth|sweat|passive eye|montage|filter")
_PHOTIC_ON = re.compile(r"photic on\s*-\s*([\d.]+)\s*hz")
_HV_ON = re.compile(r"^hv\b.*\bon$")
_HV_OFF = re.compile(r"^hv\s*-\s*off")


def map_events(events, n_samples):
    """Event list [[idx, text], ...] -> [(signal_idx, lowercase text)] with paused time removed."""
    out, removed, pause_start = [], 0, None
    for idx, text in events:
        t = " ".join(str(text).strip().lower().split())
        if t == "paused":
            if pause_start is None:
                pause_start = idx
            out.append((idx - removed, "paused"))
        elif t == "recording resumed":
            if pause_start is not None:
                removed += idx - pause_start
                pause_start = None
            out.append((idx - removed, "resumed"))
        elif pause_start is not None:  # annotated while paused: belongs at the join
            out.append((pause_start - removed, t))
        else:
            out.append((idx - removed, t))
    return [(int(min(max(i, 0), n_samples - 1)), t) for i, t in out]


def build_timeline(events, n_samples, fs_event, cfg):
    """Per-second arrays: state (int8), clean (bool), plus photic blocks [(start_s, end_s, freq)].

    cfg: the `data` config section.
    """
    n_sec = n_samples // int(fs_event)
    state = np.zeros(n_sec, np.int8)
    clean = np.ones(n_sec, bool)
    mapped = map_events(events, n_samples)

    def mask(t0, t1):
        a, b = max(int(np.floor(t0)), 0), min(int(np.ceil(t1)), n_sec)
        if b > a:
            clean[a:b] = False

    pre, post = cfg["transition_guard_sec"]
    art_pre, art_post = cfg["artifact_guard_sec"]
    join = cfg["join_guard_sec"]

    eyes, photic_freq, hv = STATE_UNKNOWN, None, False
    hv_off_until = -1.0
    photic_blocks = []
    photic_start = 0.0
    prev_t = 0.0

    def effective():
        if hv:
            return STATE_HV
        if photic_freq is not None:
            return STATE_PHOTIC
        return eyes

    def fill(t0, t1):
        a, b = int(t0), min(int(np.ceil(t1)), n_sec)
        if b <= a:
            return
        state[a:b] = effective()
        if hv_off_until > t0:  # recovery period after hyperventilation still counts as HV
            c = min(int(np.ceil(hv_off_until)), b)
            state[a:c] = STATE_HV

    for idx, text in mapped:
        t = idx / fs_event
        fill(prev_t, t)
        prev_t = t
        before = effective()
        if text == "eyes open" or text == "passive eye open":
            eyes = STATE_EO
        elif text == "eyes closed" or text == "passive eye close":
            eyes = STATE_EC
        elif (m := _PHOTIC_ON.search(text)):
            photic_freq = float(m.group(1))
            photic_start = t
        elif text.startswith("photic off"):
            if photic_freq is not None:
                photic_blocks.append((photic_start, t, photic_freq))
            photic_freq = None
        elif _HV_ON.search(text):
            hv = True
        elif _HV_OFF.search(text):
            hv = False
            hv_off_until = t + cfg["post_hv_sec"]
        elif text in ("paused", "resumed"):
            mask(t - join, t + join)
        elif _LONG_ARTIFACT.search(text):
            mask(t - art_pre, t + cfg["long_artifact_sec"])
        elif _ELECTRODE.search(text):
            mask(t - art_pre, t + art_post + 2)
        elif _ARTIFACT.search(text):
            mask(t - art_pre, t + art_post)
        if effective() != before:
            mask(t - pre, t + post)
    fill(prev_t, n_sec)
    if photic_freq is not None:
        photic_blocks.append((photic_start, n_sec, photic_freq))

    clean[: int(cfg["skip_start_sec"])] = False
    blocks = np.array(photic_blocks, np.float32).reshape(-1, 3)
    return state, clean, blocks


def window_starts(state, clean, target_state, win_sec, require_clean=True):
    """Start seconds s such that seconds [s, s+win) are all in target_state (and clean)."""
    ok = state == target_state if target_state is not None else np.ones_like(clean)
    if require_clean:
        ok = ok & clean
    if len(ok) < win_sec:
        return np.zeros(0, np.int64)
    c = np.concatenate([[0], np.cumsum(ok, dtype=np.int64)])
    return np.flatnonzero(c[win_sec:] - c[:-win_sec] == win_sec).astype(np.int64)
