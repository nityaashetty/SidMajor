"""Step 1: original EDFs + event JSONs -> artifacts/store (signals, state timelines, manifest).

Resumable: recordings already in the store are skipped unless --rebuild is given or the
`data` config section changed.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap  # noqa: F401
import numpy as np

from src.common.config import add_common_args, load_config, store_fingerprint
from src.common.logging_utils import log, read_json, write_json
from src.data.events import STATE_NAMES
from src.data.splits import all_task_serials, load_splits
from src.data.store import CHANNELS, build_one, store_dirs


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N recordings (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dcfg = cfg["data"]
    fp = store_fingerprint(cfg)

    splits = load_splits(cfg["paths"]["dataset_root"])
    serials = all_task_serials(splits)[: args.limit]
    sig_dir, tl_dir, mf_path = store_dirs(cfg["paths"]["store"])
    sig_dir.mkdir(parents=True, exist_ok=True)
    tl_dir.mkdir(parents=True, exist_ok=True)

    old = {}
    if mf_path.exists() and not args.rebuild:
        m = read_json(mf_path)
        if m.get("fingerprint") == fp:
            old = m["recordings"]
        else:
            log("store: data config changed - rebuilding everything")
    todo = [s for s in serials if s not in old or not (sig_dir / f"{s}.npy").exists()]
    log(f"store: {len(serials)} recordings, {len(serials) - len(todo)} already built, {len(todo)} to build "
        f"({dcfg['workers']} workers, {dcfg['fs_in']}->{dcfg['fs']} Hz, band-pass {dcfg['bandpass']})")

    recs = dict(old)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=dcfg["workers"]) as ex:
        futs = {ex.submit(build_one, s, cfg["paths"]["dataset_root"], cfg["paths"]["store"], dcfg): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            recs[futs[fut]] = fut.result()
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                log(f"store: {i}/{len(todo)} built ({rate:.1f} rec/s, ~{(len(todo) - i) / rate / 60:.1f} min left)")

    # channel scale from TRAIN recordings only (median over recordings of clean-signal std)
    train = [r["serial"] for r in splits["train"] if r["serial"] in recs]
    ch_scale = np.median(np.array([recs[s]["ch_std"] for s in train]), axis=0)
    write_json(mf_path, {
        "fingerprint": fp,
        "fs": dcfg["fs"],
        "channels": CHANNELS,
        "states": STATE_NAMES,
        "channel_scale_uv": ch_scale.tolist(),
        "channel_scale_source": f"median clean-signal std over {len(train)} train recordings",
        "recordings": recs,
    })

    by_state = np.array([recs[s]["clean_sec_by_state"] for s in serials])
    med = np.median(by_state, axis=0)
    no_ec = int((by_state[:, 1] < 60).sum())
    log("store: done. median clean seconds per recording: " +
        ", ".join(f"{n}={int(v)}" for n, v in zip(STATE_NAMES, med)) +
        f" | recordings with <60 s clean eyes-closed: {no_ec}")


if __name__ == "__main__":
    main()
