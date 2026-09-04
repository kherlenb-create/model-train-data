"""Precompute analog *features* as float16, halving the mmap working set.

Why: a_pattern.npy (3.2G) + a_future.npy (1.6G) exceed what a 16GB box keeps in
page cache once the model and dataloader workers are resident. Training runs
fast until the cache fills, then every batch's random analog reads become SSD
faults -- throughput fell ~45x mid-epoch (37 -> 0.8 samp/s) at the *same* K and
batch size, which is the signature of paging, not compute.

What is stored: the 4 log-space features (gap, body, upper, lower) per bar,
already vol-scaled by the analog's own scale, exactly as __getitem__ would build
them. Two reasons this is the right representation for fp16:

  * Raw prices are hopeless in fp16 -- a $40,000 level quantizes to steps of 32.
  * Log prices are *also* hopeless: fp16 spacing at log(40000)=10.6 is 7.8e-3,
    while a typical 1-bar log return is 1.3e-3. The step is 6x the signal, so
    consecutive-bar differences -- which is what every feature is -- get
    obliterated (measured error/signal ratio: 2.4).

Features are differences of logs: small, centred near zero, where fp16 has ~1e-4
absolute resolution. Measured round-trip error after this change is ~2e-4, far
below the 1e-2 scale the model resolves.

Precomputing also moves ohlc_to_features + vol_scale off the training hot path.

Corrupt bars (non-finite or non-positive) and padding become NaN; the loader
masks any slot whose features are not finite.

Usage:
  python shrink_dataset.py candles_7y candles_7y_fp16
"""

import os
import shutil
import sys

import numpy as np

from data import is_valid_ohlc, ohlc_to_features, valid_prefix_len, vol_scale
from config import Config

CHUNK = 64


def convert(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    cfg = Config()

    # queries and targets: copy verbatim. They are small, the target is the
    # training signal, and __getitem__ needs raw prices for last_close chaining.
    for name in ("q_pattern.npy", "q_future.npy"):
        print(f"  copying {name} (float32, unchanged)", flush=True)
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    shutil.copyfile(os.path.join(src, "meta.npz"), os.path.join(dst, "meta.npz"))

    ap = np.load(os.path.join(src, "a_pattern.npy"), mmap_mode="r")
    af = np.load(os.path.join(src, "a_future.npy"), mmap_mode="r")
    n, _, K, Lp, _ = ap.shape
    Lf = af.shape[3]

    # one array: [pattern features ; future features] per analog, plus the
    # pattern length so the loader knows where the future starts
    feat = np.lib.format.open_memmap(
        os.path.join(dst, "a_feat.npy"), mode="w+", dtype=np.float16,
        shape=(n, K, Lp + Lf, 4))
    plen = np.lib.format.open_memmap(
        os.path.join(dst, "a_plen.npy"), mode="w+", dtype=np.int16, shape=(n, K))

    n_corrupt = 0
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        blk_p = np.asarray(ap[s:e], dtype=np.float64)
        blk_f = np.asarray(af[s:e], dtype=np.float64)
        out = np.full((e - s, K, Lp + Lf, 4), np.nan, dtype=np.float32)
        lens = np.zeros((e - s, K), dtype=np.int16)
        for bi in range(e - s):
            for j in range(K):
                pat = blk_p[bi, 0, j]
                aw = valid_prefix_len(pat)
                if aw < 1:
                    continue
                fut = blk_f[bi, 0, j]
                if not is_valid_ohlc(fut).all():
                    n_corrupt += 1
                    continue
                pf = ohlc_to_features(pat[:aw])
                # vol scale from the analog's own pattern, as __getitem__ does
                sa = vol_scale(pf, cfg.vol_floor)
                ff = ohlc_to_features(fut, prev_close=float(pat[aw - 1, 3]))
                out[bi, j, :aw] = pf / sa
                out[bi, j, Lp:Lp + Lf] = ff / sa
                lens[bi, j] = aw
        feat[s:e] = out.astype(np.float16)
        plen[s:e] = lens
        print(f"  a_feat: {e / n * 100:5.1f}%  (unusable analog slots: {n_corrupt})",
              end="\r", flush=True)
    feat.flush(); plen.flush()
    del feat, plen

    before = (os.path.getsize(os.path.join(src, "a_pattern.npy"))
              + os.path.getsize(os.path.join(src, "a_future.npy"))) / 1e9
    after = (os.path.getsize(os.path.join(dst, "a_feat.npy"))
             + os.path.getsize(os.path.join(dst, "a_plen.npy"))) / 1e9
    print(f"\n  analogs: {before:.2f}G -> {after:.2f}G "
          f"({n_corrupt} unusable slots)", flush=True)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("refusing to convert a dataset onto itself")
    print(f"converting {src} -> {dst}")
    convert(src, dst)
    print("\ndone. Set Config.data_dir to the new directory "
          "(analog_feats is auto-detected), then delete feature_stats.json.")


if __name__ == "__main__":
    main()
