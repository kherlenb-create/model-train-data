"""
build_analogs.py
================
Generate a training set of `N_QUERIES` random (symbol, anchor, window_size)
patterns and their analogs, using the batched, prod-parity retrieval in
test_retrieval.retrieve_batch.

Covers BOTH free axes the scanner exposes, without redundant work:
  * VARIABLE top_k  -> retrieve ONCE at max(KS); the top-10 list is exactly the
                       first 10 of top-200, so every smaller K is a free slice.
  * TIME FILTERS    -> the scanner's last-1y/3y/5y/15y/all-time options. All
                       filters share ONE matmul per length and are stored as a
                       (Q, n_filters, MAXK) matrix. Filters are relative to each
                       query's own timestamp -> leak-free for historical anchors.

Continuations are NOT materialised (would be many GB). We store compact analog
REFERENCES (sym_id + end_idx + score); rebuild the (H,4) log-relative OHLC on
demand with continuations_for() -- the raw store is ~50 MB and always in memory.

RESUMABLE: work is done one length at a time with progress/ETA prints, and the
whole array is checkpointed to OUT_PATH every CHECKPOINT_EVERY lengths. Re-running
loads the checkpoint and skips lengths already done (queries are deterministic
from SEED/N_QUERIES/LENGTHS, so a resume lines up exactly). Delete OUT_PATH to
start fresh.

Output: analogs.npz
    symbols    (S,)                str    id -> instrument name (for sym_id)
    filters    (F,)                str    ['1y']  (see FILTERS; scanner date filter)
    ks         (len KS,)           int64  the K levels this set supports
    done_lengths (?,)             int64  lengths already computed (for resume)
    q_symbol   (Q,)                str    query instrument
    q_end_ts   (Q,)                int64  query window's newest bar (ns UTC)
    q_window   (Q,)                int16  window_size used
    scores     (Q, F, MAXK)        f32    cosine sim per analog (nan = padding)
    a_sym_id   (Q, F, MAXK)        int16  analog instrument id (-1 = padding)
    a_end_idx  (Q, F, MAXK)        int64  analog last-bar index into store (-1 pad)
    a_end_ts   (Q, F, MAXK)        int64  analog last bar (ns UTC)
    n_analogs  (Q, F)              int16  real analogs available (<= MAXK)

Slice at train time:  scores[:, f, :K]  for filter index f and any K in KS.
"""
import os
import time
import numpy as np
import pandas as pd
import test_retrieval as T

# ---------------- CONFIG ----------------
DATA_DIR   = "./data"
N_QUERIES  = 10000
KS         = [10, 20, 50, 80, 100, 150, 200]          # top_k levels available
FILTERS    = {"7y": 7}             # scanner "last 1y" date filter (see network.json)
STRIDE     = 1                       # 1 = exact prod coverage
HORIZON    = T.HORIZON
QUERY_EDGE = "right"                 # anchor = window's LAST bar (matches the GBP_CAD
                                     # network.json cache_key, whose ts is the newest bar).
PROD_PARITY = False                  # True  -> reproduce live prod (~90-98% analog match
                                     #          vs network.json); keeps FUTURE + self-overlap
                                     #          analogs. LEAKS THE FUTURE -> eval/parity only.
                                     # False -> leak-free training set (future-excluded).
                                     # See retrieve()/retrieve_batch prod_parity + docstring.
                                     # NOTE: flipped to False for the model training set.
                                     # The old prod-parity artifact lives on as
                                     # analogs_all.npz; this run writes analogs_train.npz.
SEED       = 0

LENGTHS      = list(range(10, 51)) + list(range(60, 101, 10))  # 10..50 every int, then 60,70,80,90,100 # cover every pattern length 10..100 (prod range)  #  130,160,200,(30-aar shift hiih), 250,300 candle windows (50-aar shift hiih )
OUT_PATH     = "analogs_7y.npz"   # leak-free training set (analogs_all.npz = prod-parity/eval)
CHECKPOINT_EVERY = 5                 # save every N lengths (also always at the end)
MAXK         = max(KS)
# -----------------------------------------

# One matmul per length shared across all 5 filters, and we process one length at
# a time -> only the current bank needs to be resident. Keep the LRU small to
# bound RAM (biggest bank ~2-5 GB fp32 during scoring).
T.LRU_BANKS = 2


def make_queries(store, n, lengths, seed=SEED, query_edge=QUERY_EDGE):
    rng = np.random.default_rng(seed)
    syms = [s for s in store.symbols if len(store.ohlc[s]) > max(lengths) + HORIZON + 2]
    out = []
    while len(out) < n:
        s = rng.choice(syms)
        N = int(rng.choice(lengths))
        ts = store.ts[s]
        # anchor index range so the whole window fits AND a full HORIZON
        # continuation exists after the window's newest bar (leak-free label).
        if query_edge == "right":                  # anchor = window's LAST bar
            lo, hi = N - 1, len(ts) - HORIZON - 1
        else:                                      # anchor = window's FIRST bar
            lo, hi = 0, len(ts) - N - HORIZON
        if hi <= lo:
            continue
        out.append({"symbol": s,
                    "anchor_ts": pd.Timestamp(ts[rng.integers(lo, hi)]),
                    "window_size": N})
    return out


def blank_arrays(Q, F, maxk):
    return dict(
        q_symbol=np.array([""] * Q, dtype=object),
        q_end_ts=np.zeros(Q, np.int64),
        q_window=np.zeros(Q, np.int16),
        scores=np.full((Q, F, maxk), np.nan, np.float32),
        a_sym_id=np.full((Q, F, maxk), -1, np.int16),
        a_end_idx=np.full((Q, F, maxk), -1, np.int64),
        a_end_ts=np.zeros((Q, F, maxk), np.int64),
        n_analogs=np.zeros((Q, F), np.int16),
    )


def fill(arr, raw, idxs, filters, sym2id):
    """Write one length's retrieval `raw` (dict years->list over the subset) into
    the global arrays at the original query positions `idxs`."""
    for f, (name, years) in enumerate(filters.items()):
        reslist = raw[years]
        for local, gi in enumerate(idxs):
            r = reslist[local]
            if not r:
                continue
            if f == 0:
                arr["q_symbol"][gi] = r["query"]["symbol"]
                arr["q_end_ts"][gi] = r["query"]["end_ts"].value
                arr["q_window"][gi] = r["query"]["window_size"]
            ms = r["matches"]
            arr["n_analogs"][gi, f] = len(ms)
            for k, m in enumerate(ms):
                arr["scores"][gi, f, k]    = m["score"]
                arr["a_sym_id"][gi, f, k]  = sym2id[m["symbol"]]
                arr["a_end_idx"][gi, f, k] = m["end_idx"]
                arr["a_end_ts"][gi, f, k]  = m["end_ts"].value


def save_checkpoint(path, arr, store, filters, done):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, symbols=np.array(store.symbols), filters=np.array(list(filters)),
            ks=np.array(KS, np.int64), done_lengths=np.array(sorted(done), np.int64),
            q_symbol=arr["q_symbol"].astype(str), q_end_ts=arr["q_end_ts"],
            q_window=arr["q_window"], scores=arr["scores"], a_sym_id=arr["a_sym_id"],
            a_end_idx=arr["a_end_idx"], a_end_ts=arr["a_end_ts"],
            n_analogs=arr["n_analogs"])
    os.replace(tmp, path)                          # atomic swap


def load_checkpoint(path, Q, F, maxk):
    d = np.load(path, allow_pickle=True)
    if d["scores"].shape != (Q, F, maxk):
        raise SystemExit(f"checkpoint {path} shape {d['scores'].shape} != "
                         f"({Q},{F},{maxk}); config changed -> delete it to rebuild")
    arr = dict(q_symbol=d["q_symbol"].astype(object), q_end_ts=d["q_end_ts"].copy(),
               q_window=d["q_window"].copy(), scores=d["scores"].copy(),
               a_sym_id=d["a_sym_id"].copy(), a_end_idx=d["a_end_idx"].copy(),
               a_end_ts=d["a_end_ts"].copy(), n_analogs=d["n_analogs"].copy())
    done = set(int(x) for x in d["done_lengths"])
    return arr, done


def continuations_for(store, symbols, a_sym_id, a_end_idx, horizon=HORIZON):
    """
    Rebuild (..., horizon, 4) log-relative OHLC from stored analog refs.
    Pass sliced views, e.g. a_sym_id[:, f, :K] / a_end_idx[:, f, :K].
    Padding entries (sym_id < 0) come back as zeros.
    """
    shape = a_end_idx.shape
    out = np.zeros(shape + (horizon, 4), np.float32)
    flat_s = a_sym_id.reshape(-1)
    flat_e = a_end_idx.reshape(-1)
    flat_o = out.reshape(-1, horizon, 4)
    for j in range(flat_s.size):
        sid, e = int(flat_s[j]), int(flat_e[j])
        if sid < 0:
            continue
        arr = store.ohlc[symbols[sid]]
        flat_o[j] = np.log(arr[e + 1: e + 1 + horizon] / arr[e, 3])
    return out


if __name__ == "__main__":
    store = T.PriceStore(DATA_DIR)
    sym2id = {s: i for i, s in enumerate(store.symbols)}
    print(f"{len(store.symbols)} symbols loaded")

    queries = make_queries(store, N_QUERIES, LENGTHS)
    Q, F = len(queries), len(FILTERS)
    by_len = {}                                    # window_size -> [global idx]
    for gi, q in enumerate(queries):
        by_len.setdefault(int(q["window_size"]), []).append(gi)
    all_lengths = sorted(by_len)
    print(f"{Q} queries over {len(all_lengths)} distinct lengths; "
          f"top-{MAXK}, {F} filters {list(FILTERS)}")

    # ---- resume or start fresh ----
    if os.path.exists(OUT_PATH):
        arr, done = load_checkpoint(OUT_PATH, Q, F, MAXK)
        print(f"resuming from {OUT_PATH}: {len(done)}/{len(all_lengths)} lengths done")
    else:
        arr, done = blank_arrays(Q, F, MAXK), set()

    todo = [N for N in all_lengths if N not in done]
    t_start = time.time()
    for i, N in enumerate(todo, 1):
        idxs = by_len[N]
        t = time.time()
        raw = T.retrieve_batch(store, [queries[g] for g in idxs], top_k=MAXK,
                               stride=STRIDE, horizon=HORIZON, query_edge=QUERY_EDGE,
                               lookback_years=list(FILTERS.values()),
                               with_continuations=False, prod_parity=PROD_PARITY)
        fill(arr, raw, idxs, FILTERS, sym2id)
        done.add(N)

        got = [len(r["matches"]) for r in raw[next(iter(FILTERS.values()))] if r]
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i:2}/{len(todo)}] len={N:3}  nq={len(idxs):4}  "
              f"{time.time()-t:5.1f}s  med analogs {int(np.median(got)) if got else 0:3}/{MAXK}"
              f"  | elapsed {elapsed/60:4.1f}m  eta {eta/60:4.1f}m", flush=True)

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(OUT_PATH, arr, store, FILTERS, done)
            print(f"      checkpoint saved ({len(done)}/{len(all_lengths)} lengths)",
                  flush=True)

    save_checkpoint(OUT_PATH, arr, store, FILTERS, done)
    print(f"DONE -> {OUT_PATH}  ({len(done)}/{len(all_lengths)} lengths, "
          f"{time.time()-t_start:.0f}s total).  Slice scores[:, f, :K]; "
          f"f in 0..{F-1}, K in {KS}")
