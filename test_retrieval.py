"""
local_retrieval.py
==================
A local, in-process mirror of the prod Pinecone search path, for training.

Mirrors SearchService.executeNewSearch step by step:
  1. vectorize query with createCandleVectors logic (exact length, no resample)
  2. similarity search against the length-N bank  (== `index-${window_size}`)
  3. self-match exclusion                          (== buildExcludeSelfMatchFilter)
  4. TIME FILTER  <-- training-only addition: no future analogs
  5. overfetch top_k + 300, dedup by (symbol, timeframe, utc-day), slice top_k
  6. fetch each analog's H-bar continuation from raw arrays

Banks are NOT pre-stored on disk. A length-N bank is computed on demand from
the raw price arrays (fully vectorized) and LRU-cached, so any query length
10..100 works without building 91 index files.

METRIC NOTE: prod Pinecone uses Euclidean distance; this mirror scores with
dot product (cosine). Because createCandleVectors L2-normalizes, the two are
monotone transforms of each other (||a-b||^2 = 2 - 2 a.b) -> identical top-K,
identical order. Scores here are COSINE (higher = better); the similarity
feature fed to the model must use this same convention at train AND serve
time (convert prod's distance at inference: cos = 1 - d/2 for squared
distance). parity_check() handles the conversion when comparing to prod.

Usage:
    store = PriceStore("./data")                      # once
    result = retrieve(store, symbol="EUR_USD",         # anchor_ts = window's FIRST
                      anchor_ts=pd.Timestamp("2023-05-10 14:00", tz="UTC"),
                      window_size=37, top_k=50)         # bar (prod convention)
    result["matches"][0] -> {symbol, end_ts, score, ...}
    result["continuations"] -> (K, H, 4) log-relative OHLC
"""

import os, glob
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- CONFIG ----------------
TS_CANDIDATES = ["ts_local", "timestamp", "time", "datetime", "date"]
DATA_FIXES = {"META": "2022-06-10"}     # trim wrong-instrument head
HORIZON = 50                            # continuation bars, must match model
BANK_STRIDE = 1                        # 1 = exact prod coverage; 2-5 for laptop
BANK_DTYPE = np.float16                # halves memory, ranking-safe
LRU_BANKS = 4                           # how many length-banks to keep in RAM
OVERFETCH = 300                         # prod: new_top_k = top_k + 300
# -----------------------------------------


# ---------------------------------------------------------------
# ingestion (same as verified earlier: epoch->UTC, META trim)
# ---------------------------------------------------------------
def _parse_ts(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.abs().median() > 1e12 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True)


class PriceStore:
    """Holds raw OHLC + timestamps per symbol. ~50 MB total. Built once."""

    def __init__(self, data_dir: str):
        self.ohlc, self.ts = {}, {}
        for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
            sym = os.path.basename(path).replace(".csv", "")
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = next(c for c in TS_CANDIDATES if c in df.columns)
            df["ts"] = _parse_ts(df[ts_col])
            df = (df[["ts", "open", "high", "low", "close"]]
                  .dropna().drop_duplicates("ts").sort_values("ts"))
            if sym in DATA_FIXES:
                df = df[df["ts"] >= pd.Timestamp(DATA_FIXES[sym], tz="UTC")]
            self.ohlc[sym] = df[["open", "high", "low", "close"]].to_numpy(np.float64)
            # store tz-naive UTC datetime64[ns] so numpy comparisons work
            self.ts[sym] = df["ts"].dt.tz_localize(None).to_numpy()
        self.symbols = list(self.ohlc)
        self._bank_cache: OrderedDict = OrderedDict()   # (N, stride) -> bank dict

    # -----------------------------------------------------------
    # prod vectorizer, batched over M windows at once
    # -----------------------------------------------------------
    @staticmethod
    def vectorize_batch(windows: np.ndarray) -> np.ndarray:
        """
        windows: (M, N, 4) OHLC  ->  (M, 4*N) unit vectors.
        Exact port of createCandleVectors, vectorized:
          per-window P_ref = mean(close); per-channel z-score (population std);
          flatten in column_stack-then-ravel order (o,h,l,c interleaved per bar);
          L2 normalize.
        """
        w = windows.astype(np.float64)
        p_ref = w[:, :, 3].mean(axis=1)[:, None, None]           # (M,1,1)
        f = w / p_ref                                            # (M,N,4)
        mu = f.mean(axis=1, keepdims=True)                       # per channel
        sd = f.std(axis=1, keepdims=True, ddof=0) + 1e-8
        z = (f - mu) / sd                                        # (M,N,4)
        v = z.reshape(len(w), -1)                                # ravel of (N,4) rows
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (v / n)

    def vectorize_one(self, window_ohlc: np.ndarray) -> np.ndarray:
        return self.vectorize_batch(window_ohlc[None])[0]

    # -----------------------------------------------------------
    # length-N bank, computed on demand, LRU cached
    # -----------------------------------------------------------
    def get_bank(self, N: int, stride: int = BANK_STRIDE) -> dict:
        key = (N, stride)
        if key in self._bank_cache:
            self._bank_cache.move_to_end(key)
            return self._bank_cache[key]

        vec_parts, sym_ids, end_idx_parts, end_ts_parts = [], [], [], []
        for si, sym in enumerate(self.symbols):
            arr = self.ohlc[sym]
            if len(arr) < N + HORIZON + 1:
                continue
            # windows with a FULL horizon after them, right edges strided
            wins = sliding_window_view(arr, (N, 4)).squeeze(axis=1)  # (n-N+1, N, 4)
            last_valid = len(arr) - HORIZON - 1                      # last legal end index
            ends = np.arange(N - 1, last_valid + 1, stride)
            wins = wins[ends - (N - 1)]
            good = ~np.any(wins[:, :, :] <= 0, axis=(1, 2))
            ends, wins = ends[good], wins[good]
            if len(ends) == 0:
                continue
            vec_parts.append(self.vectorize_batch(wins).astype(BANK_DTYPE))
            sym_ids.append(np.full(len(ends), si, dtype=np.int32))
            end_idx_parts.append(ends.astype(np.int64))
            end_ts_parts.append(self.ts[sym][ends])

        bank = {
            "vecs": np.concatenate(vec_parts),                  # (M, 4N) fp16
            "sym_id": np.concatenate(sym_ids),                  # (M,)
            "end_idx": np.concatenate(end_idx_parts),           # (M,) index into raw arr
            "end_ts": np.concatenate(end_ts_parts),             # (M,) datetime64
        }
        bank["day_key"] = bank["end_ts"].astype("datetime64[D]")  # utc-day for dedup

        self._bank_cache[key] = bank
        if len(self._bank_cache) > LRU_BANKS:
            self._bank_cache.popitem(last=False)
        return bank


# ---------------------------------------------------------------
# the prod-parity retrieve()
# ---------------------------------------------------------------
def retrieve(store: PriceStore,
             symbol: str,
             anchor_ts: pd.Timestamp,
             window_size: int,
             top_k: int,
             lam: float = 0.0,
             horizon: int = HORIZON,
             stride: int = BANK_STRIDE,
             query_edge: str = "left",
             lookback_years: float = None,
             prod_parity: bool = False) -> dict:
    """
    Mirrors SearchService.executeNewSearch, plus the training-only time filter
    and optional recency weighting (lam=0 reproduces prod scoring exactly).

    lookback_years mirrors the scanner's date filter (last 1/3/5/15y, all-time):
    only analogs whose window ends within `lookback_years` BEFORE the query's
    newest bar are eligible. None = all-time (prod cache_key "no-filter").
    Relative to the query's own timestamp, so it is leak-free for historical
    anchors (a 2018 query's "last 3y" = 2015-2018, never real-now).

    anchor_ts + query_edge define the query window (window_size bars):
      query_edge="left"  (prod / default): the bars STARTING at anchor_ts.
          This matches prod, whose cache_key anchors on the window's FIRST bar
          (e.g. cache_key ...:1776326400:... -> 2026-04-14 05:00 is the first
          of 50 query bars; the window ends at 2026-04-16 08:00). Pass a prod
          cache_key timestamp here to reproduce that search exactly.
      query_edge="right": the bars ENDING at anchor_ts (look-back convention).

    Either convention is fine for training AS LONG AS it is used at both train
    and serve time; "left" reproduces prod from a cache_key timestamp. NOTE: the
    older behaviour of this function was an implicit "right" edge, which does NOT
    match prod and yields ~0% analog overlap -- see git history / parity notes.

    prod_parity=False (default): TRAINING mode. Applies the leak-free future
    filter (no analog whose window+continuation reaches the query's newest bar)
    and excludes ALL self-windows overlapping the query.

    prod_parity=True: reproduce live prod. Drops the future filter (prod serves
    at real-now and returns analogs dated after a historical query) and keeps
    overlapping self-windows (prod's top hit is the query symbol shifted a bar),
    excluding only the exact query window. With lookback_years=1 + query_edge and
    the correct UTC anchor this matches the GBP_CAD network.json at ~96% (residual
    = fp16 bank + local CSVs newer than prod's index snapshot). Do NOT use for
    training: it leaks the future.
    """
    assert 10 <= top_k <= 300, "prod bounds"
    if query_edge not in ("left", "right"):
        raise ValueError("query_edge must be 'left' or 'right'")

    # --- 1. cut + vectorize the query at its exact length ---
    ts_arr = store.ts[symbol]
    anchor_ts = pd.Timestamp(anchor_ts)
    q_ts = np.datetime64(anchor_ts.tz_convert("UTC").tz_localize(None)
                         if anchor_ts.tzinfo else anchor_ts)
    qi = np.searchsorted(ts_arr, q_ts)
    if qi >= len(ts_arr) or ts_arr[qi] != q_ts:
        raise ValueError(f"{symbol}: no bar at {anchor_ts}")
    # window bounds [lo, hi) of length window_size; anchor is left or right edge
    lo, hi = (qi, qi + window_size) if query_edge == "left" \
             else (qi - window_size + 1, qi + 1)
    if lo < 0 or hi > len(ts_arr):
        raise ValueError("not enough bars for the query window at this anchor")
    q_ohlc = store.ohlc[symbol][lo:hi]
    q_vec = store.vectorize_one(q_ohlc)

    q_end_idx = hi - 1                    # index of the query window's newest bar
    q_end_ts = ts_arr[q_end_idx]          # reference "now" for filters / recency

    bank = store.get_bank(window_size, stride)

    # --- 2. similarity: vectors are unit -> cosine == dot ---
    scores = bank["vecs"].astype(np.float32) @ q_vec.astype(np.float32)

    # --- 3+4. filters as masks ---
    q_sym = store.symbols.index(symbol)
    valid = np.ones(len(scores), dtype=bool)
    # time filter (TRAINING ONLY): analog + its full continuation strictly before
    # the query's newest bar -> leak-free. prod_parity skips it: live serving uses
    # all data up to real-now, so prod returns analogs "after" a historical query.
    if not prod_parity:
        bar = ts_arr[1] - ts_arr[0]                   # H1 bar step
        cutoff = q_end_ts - horizon * bar
        valid &= bank["end_ts"] < cutoff
    # scanner date filter: analog must be within lookback_years of the query bar
    if lookback_years is not None:
        lower = q_end_ts - np.timedelta64(int(round(lookback_years * 365.25)), "D")
        valid &= bank["end_ts"] >= lower
    # self-match exclusion: same symbol, windows overlapping the query window.
    # prod keeps these (its top hit is the query symbol shifted a bar), so under
    # prod_parity we only drop the exact query window (self, distance 0).
    if prod_parity:
        valid &= ~((bank["sym_id"] == q_sym) & (bank["end_idx"] == q_end_idx))
    else:
        overlap = (bank["sym_id"] == q_sym) & \
                  (np.abs(bank["end_idx"] - q_end_idx) < window_size)
        valid &= ~overlap
    scores[~valid] = -np.inf

    # --- optional recency weighting (training knob; lam=0 == prod) ---
    if lam > 0:
        age_days = (q_end_ts - bank["end_ts"]) / np.timedelta64(1, "D")
        scores = np.where(np.isfinite(scores),
                          scores * np.exp(-lam * np.maximum(age_days, 0) / 365.0),
                          scores)

    # --- 5. overfetch, dedup by (symbol, utc-day), slice ---
    n_fetch = min(top_k + OVERFETCH, len(scores))
    cand = np.argpartition(-scores, n_fetch - 1)[:n_fetch]
    cand = cand[np.argsort(-scores[cand])]            # sorted best-first

    seen, keep = set(), []
    for i in cand:
        if not np.isfinite(scores[i]):
            break
        k = (bank["sym_id"][i], bank["day_key"][i])   # prod key: symbol|timeframe|day
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
        if len(keep) == top_k:
            break

    # --- 6. continuations from raw arrays, log-relative to analog's last close ---
    conts, matches = [], []
    for i in keep:
        sym = store.symbols[bank["sym_id"][i]]
        e = int(bank["end_idx"][i])
        anchor = store.ohlc[sym][e, 3]
        cont = np.log(store.ohlc[sym][e + 1: e + 1 + horizon] / anchor)
        conts.append(cont.astype(np.float32))
        matches.append({
            "symbol": sym,
            "end_ts": pd.Timestamp(bank["end_ts"][i]),
            "score": float(scores[i]),
            "end_idx": e,
            "age_days": float((q_end_ts - bank["end_ts"][i])
                              / np.timedelta64(1, "D")),
        })

    return {
        "query": {"symbol": symbol, "anchor_ts": anchor_ts,
                  "query_edge": query_edge,
                  "end_ts": pd.Timestamp(q_end_ts), "window_size": window_size},
        "matches": matches,
        "continuations": np.stack(conts) if conts else np.zeros((0, horizon, 4)),
    }


# ---------------------------------------------------------------
# batched many-query retrieval  (same results as retrieve(), ~10-20x faster)
# ---------------------------------------------------------------
def retrieve_batch(store: PriceStore,
                   queries: list,
                   top_k: int = 50,
                   lam: float = 0.0,
                   horizon: int = HORIZON,
                   stride: int = BANK_STRIDE,
                   query_edge: str = "left",
                   lookback_years: float = None,
                   with_continuations: bool = True,
                   chunk_target_gb: float = 0.75,
                   prod_parity: bool = False) -> list:
    """
    Vectorized version of retrieve() for many queries of MIXED lengths.

    Buckets queries by window_size so each length-bank is built + fp32-cast ONCE,
    scores every query in a bucket with a single batched matmul (chunked to bound
    RAM), then applies the identical self-match + time filters and (symbol, utc-day)
    dedup per query. Output for each query is byte-identical to retrieve().

    queries: list of dicts, each:
        {"symbol": str, "anchor_ts": pd.Timestamp/str, "window_size": int,
         "query_edge": "left"|"right"  (optional, defaults to `query_edge`)}

    lookback_years: None/float -> returns a LIST aligned to `queries` (each item a
        retrieve()-style dict, or None for an invalid query).
        list/tuple of values (e.g. [1,3,5,15,None]) -> runs ALL those scanner date
        filters in a single pass (one matmul per length, shared across filters) and
        returns a DICT {lookback_value: list_aligned_to_queries}. Use this to cover
        many lengths x filters without rebuilding banks per filter.

    chunk_target_gb caps the (M x chunk) score matrix; lower it if RAM is tight.
    """
    if query_edge not in ("left", "right"):
        raise ValueError("query_edge must be 'left' or 'right'")
    assert 10 <= top_k <= 300, "prod bounds"

    # lookback_years may be a single value (list output) or a list of values
    # (dict output keyed by value) -- the list form shares ONE matmul per length
    # across all filters, so covering many lengths x filters stays cheap.
    multi = isinstance(lookback_years, (list, tuple))
    lookbacks = list(lookback_years) if multi else [lookback_years]

    sym2id = {s: i for i, s in enumerate(store.symbols)}
    results = {lb: [None] * len(queries) for lb in lookbacks}

    buckets = defaultdict(list)                       # window_size -> [query idx]
    for gi, q in enumerate(queries):
        buckets[int(q["window_size"])].append(gi)

    for N, gis in buckets.items():
        bank = store.get_bank(N, stride)
        V = bank["vecs"].astype(np.float32)           # cast ONCE per length
        Vsym, Vidx = bank["sym_id"], bank["end_idx"]
        Vts, Vday = bank["end_ts"], bank["day_key"]
        M = len(Vsym)
        n_fetch = min(top_k + OVERFETCH, M)

        # ---- build query windows + per-query meta for this bucket ----
        wins, gkeep = [], []
        q_endidx, q_endts, q_symid, cutoff = [], [], [], []
        for gi in gis:
            q = queries[gi]
            sym = q["symbol"]
            edge = q.get("query_edge", query_edge)
            if sym not in sym2id:
                continue
            ts_arr = store.ts[sym]
            a = pd.Timestamp(q["anchor_ts"])
            qts = np.datetime64(a.tz_convert("UTC").tz_localize(None)
                                if a.tzinfo else a)
            qi = np.searchsorted(ts_arr, qts)
            if qi >= len(ts_arr) or ts_arr[qi] != qts:
                continue
            lo, hi = (qi, qi + N) if edge == "left" else (qi - N + 1, qi + 1)
            if lo < 0 or hi > len(ts_arr):
                continue
            eidx = hi - 1
            wins.append(store.ohlc[sym][lo:hi])
            gkeep.append(gi)
            q_endidx.append(eidx)
            q_endts.append(ts_arr[eidx])
            q_symid.append(sym2id[sym])
            cutoff.append(ts_arr[eidx] - horizon * (ts_arr[1] - ts_arr[0]))
        if not wins:
            continue

        Q = store.vectorize_batch(np.stack(wins)).astype(np.float32)   # (nv, 4N)
        q_endidx = np.asarray(q_endidx)
        q_symid = np.asarray(q_symid)
        q_endts = np.asarray(q_endts, dtype=Vts.dtype)
        cutoff = np.asarray(cutoff, dtype=Vts.dtype)

        C = max(1, int(chunk_target_gb * 1e9 / (M * 4)))   # queries per chunk
        for s0 in range(0, len(gkeep), C):
            sl = slice(s0, s0 + C)
            Qc = Q[sl]
            symc, idxc, cutc = q_symid[sl], q_endidx[sl], cutoff[sl]
            S = V @ Qc.T                                    # (M, c) cosine scores
            # SHARED masks (same for every filter). prod_parity mirrors
            # retrieve(prod_parity=True): drop the leak-free future cutoff and the
            # self-OVERLAP exclusion, keeping only the exact-self-window drop, so
            # the batch reproduces live prod (LEAKS the future -- eval only).
            if prod_parity:
                base_bad = (Vsym[:, None] == symc[None, :]) & \
                           (Vidx[:, None] == idxc[None, :])
            else:
                base_bad = ~(Vts[:, None] < cutc[None, :])       # future-exclusion (leak-free)
                base_bad |= (Vsym[:, None] == symc[None, :]) & \
                            (np.abs(Vidx[:, None] - idxc[None, :]) < N)
            S[base_bad] = -np.inf

            for lb in lookbacks:                            # cheap per-filter pass
                if lb is None:                              # all-time
                    Sf = S if lam == 0 else S.copy()
                else:                                       # scanner date floor
                    lower = q_endts[sl] - np.timedelta64(
                        int(round(lb * 365.25)), "D")
                    Sf = np.where(Vts[:, None] >= lower[None, :], S, -np.inf)
                if lam > 0:
                    age = (q_endts[sl][None, :] - Vts[:, None]) / np.timedelta64(1, "D")
                    np.multiply(Sf, np.exp(-lam * np.maximum(age, 0) / 365.0),
                                out=Sf, where=np.isfinite(Sf))

                part = np.argpartition(-Sf, n_fetch - 1, axis=0)[:n_fetch]
                for j in range(Qc.shape[0]):
                    col = part[:, j]
                    sc = Sf[col, j]
                    order = np.argsort(-sc)
                    col, sc = col[order], sc[order]

                    seen, keep = set(), []
                    for c_idx, sval in zip(col, sc):
                        if not np.isfinite(sval):
                            break
                        k = (Vsym[c_idx], Vday[c_idx])
                        if k in seen:
                            continue
                        seen.add(k)
                        keep.append((int(c_idx), float(sval)))
                        if len(keep) == top_k:
                            break

                    matches, conts = [], []
                    for c_idx, sval in keep:
                        ms = store.symbols[Vsym[c_idx]]
                        e = int(Vidx[c_idx])
                        matches.append({
                            "symbol": ms,
                            "end_ts": pd.Timestamp(Vts[c_idx]),
                            "score": sval,
                            "end_idx": e,
                            "age_days": float((q_endts[s0 + j] - Vts[c_idx])
                                              / np.timedelta64(1, "D")),
                        })
                        if with_continuations:
                            anc = store.ohlc[ms][e, 3]
                            conts.append(np.log(store.ohlc[ms][e + 1: e + 1 + horizon]
                                                / anc).astype(np.float32))

                    gi = gkeep[s0 + j]
                    q = queries[gi]
                    results[lb][gi] = {
                        "query": {"symbol": q["symbol"],
                                  "anchor_ts": pd.Timestamp(q["anchor_ts"]),
                                  "query_edge": q.get("query_edge", query_edge),
                                  "end_ts": pd.Timestamp(q_endts[s0 + j]),
                                  "window_size": N},
                        "matches": matches,
                        "continuations": (np.stack(conts) if conts
                                          else np.zeros((0, horizon, 4), np.float32))
                                         if with_continuations else None,
                    }
    return results if multi else results[lookbacks[0]]


# ---------------------------------------------------------------
# parity check against prod (feed it exported Pinecone results)
# ---------------------------------------------------------------
# Prod's Pinecone index uses EUCLIDEAN distance; the local mirror scores with
# cosine (dot product on unit vectors). For L2-normalized vectors these give
# the IDENTICAL ranking:  ||a-b||^2 = 2 - 2*(a.b).  Only the score number
# differs, so prod scores are converted to cosine before comparison.
def _prod_score_to_cosine(d: float, squared: bool) -> float:
    return 1.0 - (d / 2.0 if squared else (d * d) / 2.0)


def parity_check(local_result: dict, prod_matches: list, tol: float = 1e-3):
    """
    prod_matches: list of {"symbol", "end_timestamp"(ms epoch), "score"}
    where "score" is the raw Pinecone Euclidean score.
    Tries both distance conventions (squared / non-squared) and reports the
    better fit -- the parity run itself tells you which one your index uses.
    """
    def _day(ts) -> pd.Timestamp:      # normalize to tz-naive UTC day
        ts = pd.Timestamp(ts)
        if ts.tzinfo:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.floor("D")

    loc = {(m["symbol"], _day(m["end_ts"])): m["score"]
           for m in local_result["matches"]}

    best = None
    for squared in (True, False):
        prd = {(m["symbol"],
                _day(pd.Timestamp(m["end_timestamp"], unit="ms", tz="UTC")))
               : _prod_score_to_cosine(m["score"], squared)
               for m in prod_matches}
        inter = set(loc) & set(prd)
        overlap = len(inter) / max(len(prd), 1)
        max_dev = max((abs(loc[k] - prd[k]) for k in inter), default=float("inf"))
        if best is None or max_dev < best[2]:
            best = (squared, overlap, max_dev)

    squared, overlap, max_dev = best
    conv = "squared" if squared else "non-squared"
    print(f"top-K overlap: {overlap:.1%}   max score deviation: {max_dev:.2e}"
          f"   (tolerance {tol}, prod distance convention: {conv})")
    if max_dev > tol:
        print("  !! score deviation above tolerance -> check vectorizer port")
    if overlap < 0.999:
        print("  !! missing matches -> run local bank with stride=1 for this test")
    return overlap, max_dev


# ---------------------------------------------------------------
if __name__ == "__main__":
    store = PriceStore("./data")
    print(f"{len(store.symbols)} symbols loaded")

    # anchor_ts is the query window's FIRST bar (prod cache_key convention);
    # this reproduces cache_key search:XAU_USD:H1:50:1776326400:50 exactly.
    r = retrieve(store, symbol="GBP_CAD",
                 anchor_ts=pd.Timestamp("2025-12-03 13:00", tz="UTC"),
                 window_size=50, top_k=50,
                 query_edge="right",        # cache_key ts is the window's LAST bar
                 lookback_years=1,          # scanner "last 1y" date filter
                 prod_parity=True)          # reproduce live prod (leaks future!)
    print(f"{len(r['matches'])} analogs, continuations {r['continuations'].shape}")
    for m in r["matches"][:5]:
        print(f"  {m['symbol']:>12}  {m['end_ts']}  sim={m['score']:.4f}"
              f"  age={m['age_days']:.0f}d")