"""
build_candles.py
================
Materialise the ACTUAL candles behind analogs_1y.npz.

analogs_1y.npz stores only compact references (symbol id + bar index) so it
stays small. This script rebuilds the raw OHLC candles from the PriceStore
(./data/*.csv) for BOTH the query windows and every retrieved analog, and
writes them as memory-mapped .npy files so peak RAM stays tiny even though the
analog tensor is ~2.4 GB on disk (float16).

For each window we store:
    * pattern : the `N` bars of the matched window (N = q_window)
    * future  : the next HORIZON (=50) bars that came after the anchor bar

Layout (raw absolute OHLC, columns = open, high, low, close):
    q_pattern (Q,       MAXW, 4)   f16   query pattern, bars [0:N], rest = NaN
    q_future  (Q,       H,    4)   f16   query's next 50 bars
    a_pattern (Q, F, K, MAXW, 4)   f16   analog patterns, [0:N] valid, rest NaN
    a_future  (Q, F, K, H,    4)   f16   analog next 50 bars
    (F = n filters = 1 here, K = MAXK = 200)

Padding / missing analogs (a_sym_id == -1) and any window that runs past the
edge of its series are left as NaN.

meta.npz carries the references + labels copied straight from analogs_1y.npz
(symbols, q_symbol, q_window, a_sym_id, a_end_idx, scores, n_analogs, ...).

Load with:
    import numpy as np
    a_pat = np.load("candles_1y/a_pattern.npy", mmap_mode="r")   # lazy, no RAM blowup
    meta  = np.load("candles_1y/meta.npz", allow_pickle=True)
"""
import os
import argparse
import numpy as np
import test_retrieval as T

IN_PATH  = "analogs_7y.npz"   # leak-free training analogs (candles_all/ was built
DATA_DIR = "./data"              # from the prod-parity analogs_all.npz -> eval only)
OUT_DIR  = "candles_7y"
HORIZON  = T.HORIZON            # 50


def ts_ns(store, sym):
    """int64 nanosecond timestamps for a symbol (source dtype is datetime64[s])."""
    return store.ts[sym].astype("datetime64[ns]").astype(np.int64)


def main(n_queries=None, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    store = T.PriceStore(DATA_DIR)
    d = np.load(IN_PATH, allow_pickle=True)

    Q_total = d["a_sym_id"].shape[0]
    Q = Q_total if n_queries is None else min(int(n_queries), Q_total)
    print(f"extracting {Q}/{Q_total} queries")

    symbols  = d["symbols"]
    q_symbol = d["q_symbol"][:Q].astype(str)
    q_window = d["q_window"][:Q].astype(int)
    q_end_ts = d["q_end_ts"][:Q].astype(np.int64)
    a_sym_id = d["a_sym_id"][:Q]             # (Q, F, K) int16, -1 = padding
    a_end_idx = d["a_end_idx"][:Q]           # (Q, F, K) int64, -1 = padding

    _, F, K = a_sym_id.shape
    MAXW = int(q_window.max())               # widest pattern (prod: 100)
    H = HORIZON
    print(f"Q={Q}  F={F}  K={K}  MAXW={MAXW}  HORIZON={H}")
    print(f"analog tensor ~ {Q*F*K*MAXW*4*2/1e9:.2f} GB (pattern) "
          f"+ {Q*F*K*H*4*2/1e9:.2f} GB (future) on disk")

    # memory-mapped outputs (written straight to disk, not held in RAM).
    # float16: candle log-features don't need f32 precision, and halving the
    # analog tensor keeps the working set cache-resident on a 16 GB machine.
    q_pat = np.lib.format.open_memmap(f"{out_dir}/q_pattern.npy", mode="w+",
                                      dtype=np.float16, shape=(Q, MAXW, 4))
    q_fut = np.lib.format.open_memmap(f"{out_dir}/q_future.npy", mode="w+",
                                      dtype=np.float16, shape=(Q, H, 4))
    a_pat = np.lib.format.open_memmap(f"{out_dir}/a_pattern.npy", mode="w+",
                                      dtype=np.float16, shape=(Q, F, K, MAXW, 4))
    a_fut = np.lib.format.open_memmap(f"{out_dir}/a_future.npy", mode="w+",
                                      dtype=np.float16, shape=(Q, F, K, H, 4))

    # cache int64-ns timestamps per symbol so we don't reconvert per query
    ts_cache = {}
    def get_ts(sym):
        if sym not in ts_cache:
            ts_cache[sym] = ts_ns(store, sym)
        return ts_cache[sym]

    def window(sym, end_idx, N):
        """(pattern[N,4], future[H,4]) ending at (inclusive) end_idx; None if OOB."""
        arr = store.ohlc[sym]
        if end_idx - N + 1 < 0 or end_idx + 1 + H > len(arr):
            return None
        pat = arr[end_idx - N + 1: end_idx + 1]           # (N, 4)
        fut = arr[end_idx + 1: end_idx + 1 + H]           # (H, 4)
        return pat.astype(np.float16), fut.astype(np.float16)

    q_missing = a_missing = a_pad = 0
    for qi in range(Q):
        N = int(q_window[qi])
        sym = q_symbol[qi]

        # ---- query window ----
        q_pat[qi].fill(np.nan)
        q_fut[qi].fill(np.nan)
        if sym:                                            # unfilled queries have ""
            ts = get_ts(sym)
            idx = int(np.searchsorted(ts, q_end_ts[qi]))
            if idx < len(ts) and ts[idx] == q_end_ts[qi]:
                w = window(sym, idx, N)
                if w:
                    q_pat[qi, :N] = w[0]
                    q_fut[qi] = w[1]
                else:
                    q_missing += 1
            else:
                q_missing += 1

        # ---- analogs ----
        a_pat[qi].fill(np.nan)
        a_fut[qi].fill(np.nan)
        for f in range(F):
            for k in range(K):
                sid = int(a_sym_id[qi, f, k])
                if sid < 0:                                # padding slot
                    a_pad += 1
                    continue
                e = int(a_end_idx[qi, f, k])
                w = window(symbols[sid], e, N)
                if w is None:
                    a_missing += 1
                    continue
                a_pat[qi, f, k, :N] = w[0]
                a_fut[qi, f, k] = w[1]

        if (qi + 1) % 500 == 0:
            print(f"  {qi+1}/{Q} queries", flush=True)

    for m in (q_pat, q_fut, a_pat, a_fut):
        m.flush()

    # references + labels, copied through for convenience (sliced to Q)
    np.savez_compressed(
        f"{out_dir}/meta.npz",
        symbols=symbols, filters=d["filters"], ks=d["ks"],
        q_symbol=q_symbol, q_window=q_window, q_end_ts=q_end_ts,
        a_sym_id=a_sym_id, a_end_idx=a_end_idx, a_end_ts=d["a_end_ts"][:Q],
        scores=d["scores"][:Q], n_analogs=d["n_analogs"][:Q],
        ohlc_cols=np.array(["open", "high", "low", "close"]),
        horizon=np.array(H), maxw=np.array(MAXW),
    )

    print(f"\nDONE -> {out_dir}/")
    print(f"  q_pattern (Q,MAXW,4)     query patterns  ({q_missing} skipped)")
    print(f"  q_future  (Q,H,4)        query futures")
    print(f"  a_pattern (Q,F,K,MAXW,4) analog patterns ({a_missing} OOB, {a_pad} padding)")
    print(f"  a_future  (Q,F,K,H,4)    analog futures")
    print(f"  meta.npz                 refs + labels")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--queries", type=int, default=10000,
                   help="number of queries to extract (default 1000; max 10000)")
    p.add_argument("-o", "--out-dir", default=OUT_DIR,
                   help=f"output directory (default {OUT_DIR})")
    args = p.parse_args()
    main(n_queries=args.queries, out_dir=args.out_dir)
