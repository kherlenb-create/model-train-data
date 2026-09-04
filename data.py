"""Dataset for retrieval-conditioned trajectory generation.

Each sample from candles_train contains:
  q_pattern (100, 4)         query pattern, NaN-padded at the tail, raw OHLC
  q_future  (50, 4)          the actual historical future (training target)
  a_pattern (1, 200, 100, 4) analog patterns (similar shapes from other symbols/times)
  a_future  (1, 200, 50, 4)  what actually followed each analog
  scores    (1, 200)         similarity, sorted descending; invalid slots are NaN

At inference the scanner hands the model a variable number of analogs
K in {10, 20, 50, 80, 100, 150, 200} and a query of variable length
(1-50 or 60/70/80/90/100 candles). Training mirrors that: K is sampled
per *batch* (see KVariableBatchSampler) and the query pattern is randomly
tail-cropped per sample; analog patterns are cropped to the same length so
they stay aligned with what the query shows.

Candles are re-parameterized so any generated vector maps back to a *valid*
OHLC bar (high >= max(open, close), low <= min(open, close)):

  gap   g = log(O_t / C_{t-1})        (first bar of a pattern uses g = 0)
  body  r = log(C_t / O_t)
  upper u = log(H_t / max(O_t, C_t))  >= 0
  lower l = log(min(O_t, C_t) / L_t)  >= 0

All four channels are divided by a per-sample volatility scale s (robust std
of the pattern's close-to-close log returns; range-based fallback for very
short windows) so the model is scale/vol invariant, then standardized with
global per-channel stats computed on the train split.
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from config import Config

EPS = 1e-12


def is_valid_ohlc(ohlc: np.ndarray) -> np.ndarray:
    """(T, 4) raw OHLC -> (T,) bool, True where the bar is usable.

    Checks `isfinite`, not just `isnan`: the source arrays contain +inf bars
    (bad ticks from dataset generation), and inf passes every isnan-based
    guard, survives np.maximum(x, EPS), then turns into NaN at log(inf/inf) --
    which silently wipes the model weights on the first backward pass."""
    return np.isfinite(ohlc).all(axis=1) & (ohlc > 0).all(axis=1)


def valid_prefix_len(ohlc: np.ndarray) -> int:
    """Length of the leading run of usable bars in a (T, 4) window.

    Patterns are stored with valid bars contiguous from index 0 and padding at
    the tail, so a prefix length is the honest measure of usable data; a bare
    count would happily accept a window with a hole in the middle."""
    ok = is_valid_ohlc(ohlc)
    bad = np.flatnonzero(~ok)
    return int(bad[0]) if bad.size else int(ohlc.shape[0])


def ohlc_to_features(ohlc: np.ndarray, prev_close: float | None = None) -> np.ndarray:
    """(T, 4) raw OHLC -> (T, 4) [gap, body, upper, lower] in log space.

    Callers must pass only bars that `is_valid_ohlc` accepts; the assert below
    is a backstop so corrupt data fails loudly here instead of surfacing as a
    NaN loss many steps later."""
    assert np.isfinite(ohlc).all() and (ohlc > 0).all(), (
        "ohlc_to_features got non-finite or non-positive prices; "
        "filter with is_valid_ohlc/valid_prefix_len first")
    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    pc = np.empty_like(c)
    pc[0] = o[0] if prev_close is None else prev_close
    pc[1:] = c[:-1]
    gap = np.log(np.maximum(o, EPS) / np.maximum(pc, EPS))
    body = np.log(np.maximum(c, EPS) / np.maximum(o, EPS))
    top = np.maximum(o, c)
    bot = np.minimum(o, c)
    upper = np.clip(np.log(np.maximum(h, EPS) / np.maximum(top, EPS)), 0.0, None)
    lower = np.clip(np.log(np.maximum(bot, EPS) / np.maximum(l, EPS)), 0.0, None)
    return np.stack([gap, body, upper, lower], axis=1).astype(np.float32)


def features_to_ohlc(feat: np.ndarray, last_close: float) -> np.ndarray:
    """(T, 4) features -> (T, 4) raw OHLC, chained from last_close. Inverse of
    ohlc_to_features up to the >=0 clamp on wicks (enforced here too)."""
    out = np.empty((feat.shape[0], 4), dtype=np.float64)
    pc = float(last_close)
    for t in range(feat.shape[0]):
        g, r, u, lo = feat[t]
        o = pc * np.exp(g)
        c = o * np.exp(r)
        h = max(o, c) * np.exp(max(u, 0.0))
        l = min(o, c) * np.exp(-max(lo, 0.0))
        out[t] = (o, h, l, c)
        pc = c
    return out


def vol_scale(pattern_feat: np.ndarray, floor: float) -> float:
    """Per-pattern volatility scale: the larger of (a) robust std of
    close-to-close log returns and (b) mean candle range (|body| + wicks).
    (b) keeps the scale honest for flat-close patterns whose MAD collapses to
    ~0, which would otherwise blow normalized targets up by orders of
    magnitude; it is also the only estimate available for 1-4 bar queries."""
    rng_proxy = float(np.mean(np.abs(pattern_feat[:, 1]) + pattern_feat[:, 2]
                              + pattern_feat[:, 3]))
    cc = pattern_feat[1:, 0] + pattern_feat[1:, 1]
    mad = 0.0
    if cc.size >= 4:
        mad = 1.4826 * float(np.median(np.abs(cc - np.median(cc))))
    s = float(max(mad, rng_proxy, floor))
    # a non-finite scale divides straight into the targets and NaNs the loss
    return s if np.isfinite(s) and s > 0 else float(floor)


class CandleTrajectoryDataset(Dataset):
    """One sample: query pattern + K analogs (pattern & future) -> target future.

    Indexing: dataset[(pos, k, crop_len)] — the batch sampler supplies k per
    batch and crop_len per sample (crop_len=None keeps the full window).
    Plain dataset[pos] uses cfg.val_k and no crop (deterministic, for val)."""

    def __init__(self, cfg: Config, indices: np.ndarray, stats: dict | None = None):
        self.cfg = cfg
        self.indices = indices
        self.stats = stats
        self._arrays = None  # opened lazily per worker (mmaps don't pickle)
        self._precomputed = False   # set by _open based on what data_dir holds

    def _open(self):
        d = self.cfg.data_dir
        self._arrays = {
            "qp": np.load(os.path.join(d, "q_pattern.npy"), mmap_mode="r"),
            "qf": np.load(os.path.join(d, "q_future.npy"), mmap_mode="r"),
        }
        # Datasets built by shrink_dataset.py carry precomputed float16 analog
        # features instead of raw OHLC: half the bytes, so the working set fits
        # in page cache, and no per-batch feature math.
        self._precomputed = os.path.exists(os.path.join(d, "a_feat.npy"))
        if self._precomputed:
            self._arrays["a_feat"] = np.load(os.path.join(d, "a_feat.npy"),
                                             mmap_mode="r")
            self._arrays["a_plen"] = np.load(os.path.join(d, "a_plen.npy"),
                                             mmap_mode="r")
        else:
            self._arrays["ap"] = np.load(os.path.join(d, "a_pattern.npy"),
                                         mmap_mode="r")
            self._arrays["af"] = np.load(os.path.join(d, "a_future.npy"),
                                         mmap_mode="r")
        meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        self._scores = meta["scores"][:, 0, :]      # (N, 200)
        self._n_analogs = meta["n_analogs"][:, 0]   # (N,)
        self._q_window = meta["q_window"]           # (N,)

    def __len__(self):
        return len(self.indices)

    def raw(self, pos: int):
        """Raw (un-normalized) pattern and future for plotting."""
        if self._arrays is None:
            self._open()
        i = int(self.indices[pos])
        qp = np.asarray(self._arrays["qp"][i])
        return qp[:valid_prefix_len(qp)], np.asarray(self._arrays["qf"][i])

    def __getitem__(self, key):
        if self._arrays is None:
            self._open()
        cfg, a = self.cfg, self._arrays
        if isinstance(key, tuple):
            pos, k_req, crop_len = key
        else:
            pos, k_req, crop_len = key, cfg.val_k, None
        i = int(self.indices[pos])
        # arrays are sized k_req so every sample in a batch collates; slots
        # beyond this sample's analog count stay masked out
        k = min(k_req, int(self._n_analogs[i]))
        # clamp to the finite prefix: q_window can overstate the usable length
        # when the stored bars contain inf (chrono_split drops such samples from
        # training, but raw/eval callers can still land here)
        w_full = min(int(self._q_window[i]),
                     valid_prefix_len(np.asarray(a["qp"][i])))
        assert w_full > 0, f"sample {i} has no valid query bars"
        w = w_full if crop_len is None else min(crop_len, w_full)

        # ---- query pattern (last w candles of the window) ----
        qp_raw = np.asarray(a["qp"][i, w_full - w:w_full])
        qp_feat_valid = ohlc_to_features(qp_raw)
        s = vol_scale(qp_feat_valid, cfg.vol_floor)
        last_close = float(qp_raw[-1, 3])

        qp_feat = np.zeros((cfg.max_pattern_len, 4), dtype=np.float32)
        qp_feat[:w] = qp_feat_valid / s
        qp_mask = np.zeros(cfg.max_pattern_len, dtype=bool)
        qp_mask[:w] = True

        # ---- target future (the actual historical path) ----
        qf_raw = np.asarray(a["qf"][i])                       # (50, 4)
        tgt = ohlc_to_features(qf_raw, prev_close=last_close) / s

        # ---- K analogs, cropped to the same window length ----
        L = cfg.max_pattern_len + cfg.horizon
        an_feat = np.zeros((k_req, L, 4), dtype=np.float32)
        an_mask = np.zeros((k_req, L), dtype=bool)
        an_seg = np.zeros((k_req, L), dtype=np.int64)         # 0=pattern 1=future
        an_score = np.zeros(k_req, dtype=np.float32)
        for j in range(k):
            score = self._scores[i, j]
            if not np.isfinite(score):
                continue        # unscored slot; masked out

            if self._precomputed:
                aw_full = int(a["a_plen"][i, j])
                aw = min(w, aw_full)
                if aw < 1:
                    continue
                Lp = cfg.max_pattern_len
                # stored pattern features are the analog's *full* window; take
                # its last `aw` bars so the crop aligns with the query's
                row = np.asarray(a["a_feat"][i, j], dtype=np.float32)
                ap_feat = row[aw_full - aw:aw_full]
                af_feat = row[Lp:Lp + cfg.horizon]
                if not (np.isfinite(ap_feat).all() and np.isfinite(af_feat).all()):
                    continue
            else:
                ap_full = np.asarray(a["ap"][i, 0, j], dtype=np.float64)
                aw_full = valid_prefix_len(ap_full)
                aw = min(w, aw_full)
                if aw < 1:
                    continue    # no usable pattern bars; slot stays masked out
                ap_raw = ap_full[aw_full - aw:aw_full]
                af_raw = np.asarray(a["af"][i, 0, j], dtype=np.float64)
                # the future must be valid end-to-end -- it is a teacher-forcing
                # conditioning signal, so a hole cannot be masked around
                if not is_valid_ohlc(af_raw).all():
                    continue
                ap_feat = ohlc_to_features(ap_raw)
                sa = vol_scale(ap_feat, cfg.vol_floor)
                ap_feat = ap_feat / sa
                af_feat = ohlc_to_features(
                    af_raw, prev_close=float(ap_raw[-1, 3])) / sa

            an_feat[j, :aw] = ap_feat
            an_feat[j, aw:aw + cfg.horizon] = af_feat
            an_mask[j, :aw + cfg.horizon] = True
            an_seg[j, aw:aw + cfg.horizon] = 1
            an_score[j] = score

        # ---- global standardization + winsorization ----
        if self.stats is not None:
            mu = np.asarray(self.stats["mean"], dtype=np.float32)
            sd = np.asarray(self.stats["std"], dtype=np.float32)
            c = cfg.feat_clip
            qp_feat[qp_mask] = np.clip((qp_feat[qp_mask] - mu) / sd, -c, c)
            an_feat[an_mask] = np.clip((an_feat[an_mask] - mu) / sd, -c, c)
            tgt = np.clip((tgt - mu) / sd, -c, c)

        # backstop: a single non-finite value here poisons every parameter on
        # the next backward pass, and the symptom (loss nan) shows up far from
        # the cause. Fail loudly, naming the sample.
        for nm, arr in (("qp_feat", qp_feat), ("an_feat", an_feat),
                        ("tgt", tgt), ("an_score", an_score)):
            assert np.isfinite(arr).all(), (
                f"non-finite {nm} for dataset index {i} "
                f"(pos={pos}, k_req={k_req}, crop_len={crop_len})")

        return {
            "qp_feat": torch.from_numpy(qp_feat),
            "qp_mask": torch.from_numpy(qp_mask),
            "an_feat": torch.from_numpy(an_feat),
            "an_mask": torch.from_numpy(an_mask),
            "an_seg": torch.from_numpy(an_seg),
            "an_score": torch.from_numpy(an_score),
            "tgt": torch.from_numpy(tgt.astype(np.float32)),
            "s": torch.tensor(s, dtype=torch.float32),
            "last_close": torch.tensor(last_close, dtype=torch.float32),
            "index": torch.tensor(i, dtype=torch.long),
        }


class KVariableBatchSampler(Sampler):
    """Shuffles samples each epoch, draws one K from cfg.k_choices per batch,
    and one crop length per sample. Batch size scales ~ k_ref/K so the
    analog-encoder cost per step stays roughly constant across K regimes."""

    def __init__(self, cfg: Config, n: int, seed: int):
        self.cfg, self.n, self.seed, self.epoch = cfg, n, seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batch_size(self, k: int) -> int:
        b = round(self.cfg.batch_size * self.cfg.k_ref / k)
        return max(self.cfg.min_batch, min(self.cfg.max_batch, b))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(self.n))
        rng.shuffle(order)
        pos = 0
        while pos < self.n:
            k = rng.choice(self.cfg.k_choices)
            bs = self._batch_size(k)
            chunk = order[pos:pos + bs]
            pos += bs
            batch = []
            for p in chunk:
                if rng.random() < self.cfg.crop_prob:
                    crop = rng.choice(self.cfg.query_len_choices)
                else:
                    crop = None  # keep the sample's natural window
                batch.append((p, k, crop))
            yield batch

    def __len__(self):
        # lower bound with the mean batch size; only used for progress display
        mean_bs = int(np.mean([self._batch_size(k) for k in self.cfg.k_choices]))
        return max(1, self.n // mean_bs)


def chrono_split(cfg: Config):
    """Chronological split on q_end_ts: past -> train, most recent -> val.
    Prevents the model from training on futures that overlap validation queries.

    Drops samples with non-finite or non-positive prices inside the valid query
    region: a single such target NaNs the loss and permanently wipes the
    model weights on the next backward pass. The test is `~isfinite`, which
    catches the +inf bars an isnan-only check lets through."""
    d = cfg.data_dir
    meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    qf = np.load(os.path.join(d, "q_future.npy"), mmap_mode="r")
    qp = np.load(os.path.join(d, "q_pattern.npy"), mmap_mode="r")
    n = qp.shape[0]
    in_window = np.arange(qp.shape[1])[None, :] < meta["q_window"][:, None]
    # chunked so a multi-GB q_pattern is never fully resident
    bad = np.zeros(n, dtype=bool)
    for s in range(0, n, 1024):
        e = min(s + 1024, n)
        f = np.asarray(qf[s:e], dtype=np.float64)
        p = np.asarray(qp[s:e], dtype=np.float64)
        bad[s:e] = ((~np.isfinite(f) | (f <= 0)).any(axis=(1, 2))
                    | ((~np.isfinite(p) | (p <= 0)).any(axis=2)
                       & in_window[s:e]).any(axis=1))
    if bad.any():
        print(f"chrono_split: dropping {int(bad.sum())} corrupt samples "
              f"(non-finite/non-positive query data): {np.where(bad)[0].tolist()}")
    order = np.argsort(meta["q_end_ts"])
    order = order[~bad[order]]
    n_val = int(len(order) * cfg.val_frac)
    return order[:-n_val], order[-n_val:]


def compute_stats(cfg: Config, train_idx: np.ndarray, n_sample: int = 2000) -> dict:
    """Per-channel mean/std of vol-scaled target features on a train subsample."""
    if os.path.exists(cfg.stats_path):
        with open(cfg.stats_path) as f:
            return json.load(f)
    rng = np.random.default_rng(cfg.seed)
    pick = rng.choice(len(train_idx), size=min(n_sample, len(train_idx)), replace=False)
    ds = CandleTrajectoryDataset(cfg, train_idx[pick], stats=None)
    feats = [ds[(j, 10, None)]["tgt"].numpy() for j in range(len(ds))]
    x = np.concatenate(feats, axis=0)
    # median + winsorized std: robust to lone monsters without over-compressing
    # the genuinely heavy tails of return distributions (IQR-based scale left
    # ~8% of gap values pinned at the clip)
    med = np.median(x, axis=0)
    lo = np.percentile(x, 0.1, axis=0)
    hi = np.percentile(x, 99.9, axis=0)
    sd = np.std(np.clip(x, lo, hi), axis=0)
    stats = {
        "mean": med.tolist(),
        "std": np.clip(sd, 1e-3, None).tolist(),
    }
    with open(cfg.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
