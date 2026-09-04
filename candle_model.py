"""Standalone model definition + inference wrapper for the candle trajectory
generator. This file plus a trained checkpoint (.pt) is everything needed to
generate trajectories — no other pipeline files required (the checkpoint
embeds the config and normalization stats).

Dependencies: torch, numpy.

Usage:
    import numpy as np
    from candle_model import TrajectoryGenerator

    gen = TrajectoryGenerator("checkpoints/best.pt")        # device auto-picked
    trajs = gen.generate(
        query_ohlc,        # (w, 4) raw OHLC, w in {1..50, 60, 70, 80, 90, 100}
        analog_patterns,   # list of K arrays (wi, 4) raw OHLC, score-sorted desc
        analog_futures,    # list of K arrays (50, 4) raw OHLC
        scores,            # list/array of K similarity scores
        n_traj=64,
        temperature=1.0,   # >1 more diverse component choice, <1 more confident
        sigma_scale=1.0,   # >1 wilder candles, <1 tighter candles
    )
    # trajs: (n_traj, 50, 4) numpy OHLC continuing from query_ohlc's last close
"""

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


# ---------------------------------------------------------------------------
# candle <-> feature transforms
# (gap, body, upper wick, lower wick) in log space; reconstruction is always
# a valid OHLC bar: high >= max(open, close) and low <= min(open, close)
# ---------------------------------------------------------------------------

def ohlc_to_features(ohlc: np.ndarray, prev_close: float | None = None) -> np.ndarray:
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
    rng_proxy = float(np.mean(np.abs(pattern_feat[:, 1]) + pattern_feat[:, 2]
                              + pattern_feat[:, 3]))
    cc = pattern_feat[1:, 0] + pattern_feat[1:, 1]
    mad = 0.0
    if cc.size >= 4:
        mad = 1.4826 * float(np.median(np.abs(cc - np.median(cc))))
    return float(max(mad, rng_proxy, floor))


# ---------------------------------------------------------------------------
# model definition (must match the training-time architecture exactly)
# ---------------------------------------------------------------------------

def _encoder(d_model, n_heads, ffn_mult, dropout, layers):
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, d_model * ffn_mult, dropout,
        batch_first=True, norm_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, layers)


class CandleGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        L_analog = cfg.max_pattern_len + cfg.horizon

        self.in_proj = nn.Linear(cfg.n_features, d)
        self.analog_in_proj = nn.Linear(cfg.n_features * cfg.analog_patch, d)
        self.pos_pattern = nn.Embedding(cfg.max_pattern_len, d)
        self.pos_analog = nn.Embedding(L_analog, d)
        self.pos_future = nn.Embedding(cfg.horizon, d)
        self.seg_emb = nn.Embedding(2, d)
        self.rank_emb = nn.Embedding(cfg.max_analogs, d)
        self.score_proj = nn.Linear(1, d)

        self.query_enc = _encoder(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout, cfg.enc_layers)
        self.analog_enc = _encoder(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout, cfg.analog_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d, cfg.n_heads, d * cfg.ffn_mult, cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Linear(d, cfg.n_mix * (1 + 2 * cfg.n_features))

    def encode(self, batch):
        cfg = self.cfg
        qp, qp_mask = batch["qp_feat"], batch["qp_mask"]
        an, an_mask, an_seg = batch["an_feat"], batch["an_mask"], batch["an_seg"]

        def bucket(n, cap):
            m = cfg.pad_multiple
            return min(-(-n // m) * m, cap)

        Lq = bucket(int(qp_mask.sum(1).max().item()), qp.size(1))
        qp, qp_mask = qp[:, :Lq], qp_mask[:, :Lq]
        La = bucket(int(an_mask.sum(2).max().item()), an.size(2))
        an, an_mask, an_seg = an[:, :, :La], an_mask[:, :, :La], an_seg[:, :, :La]
        B, K, L, _ = an.shape

        pos = torch.arange(qp.size(1), device=qp.device)
        q = self.in_proj(qp) + self.pos_pattern(pos)[None]
        q = self.query_enc(q, src_key_padding_mask=~qp_mask)

        P = cfg.analog_patch
        Lp = L // P
        a = self.analog_in_proj(an.reshape(B * K, Lp, P * cfg.n_features))
        apos = torch.arange(Lp, device=an.device)
        seg = an_seg.reshape(B * K, Lp, P).amax(-1)
        a = a + self.pos_analog(apos)[None] + self.seg_emb(seg)
        a_mask = an_mask.reshape(B * K, Lp, P).any(-1)
        empty = ~a_mask.any(dim=1)
        enc_mask = a_mask.clone()
        enc_mask[empty, 0] = True
        a = self.analog_enc(a, src_key_padding_mask=~enc_mask)
        denom = enc_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (a * enc_mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = pooled.reshape(B, K, -1)

        rank = torch.arange(K, device=an.device)
        pooled = pooled + self.rank_emb(rank)[None] + self.score_proj(batch["an_score"].unsqueeze(-1))

        memory = torch.cat([q, pooled], dim=1)
        analog_pad = empty.reshape(B, K)
        memory_pad = torch.cat([~qp_mask, analog_pad], dim=1)
        return memory, memory_pad

    def _decode(self, prev_candles, memory, memory_pad):
        B = memory.size(0)
        tok = self.start_token.expand(B, 1, -1)
        if prev_candles is not None and prev_candles.size(1) > 0:
            tok = torch.cat([tok, self.in_proj(prev_candles)], dim=1)
        T = tok.size(1)
        pos = torch.arange(T, device=memory.device)
        tok = tok + self.pos_future(pos)[None]
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=memory.device)
        h = self.decoder(tok, memory, tgt_mask=causal, tgt_is_causal=True,
                         memory_key_padding_mask=memory_pad)
        return self._split_head(self.head(h))

    def _split_head(self, out):
        cfg = self.cfg
        B, T, _ = out.shape
        M, Fdim = cfg.n_mix, cfg.n_features
        logits = out[..., :M]
        rest = out[..., M:].reshape(B, T, M, 2 * Fdim)
        mean = rest[..., :Fdim]
        log_sigma = rest[..., Fdim:].clamp(-7.0, 3.0)
        return logits, mean, log_sigma

    @torch.no_grad()
    def sample(self, batch, n_traj=32, temperature=1.0, sigma_scale=1.0):
        cfg = self.cfg
        memory, memory_pad = self.encode(batch)
        B = memory.size(0)
        memory = memory.repeat_interleave(n_traj, dim=0)
        memory_pad = memory_pad.repeat_interleave(n_traj, dim=0)

        prev = None
        for _ in range(cfg.horizon):
            logits, mean, log_sigma = self._decode(prev, memory, memory_pad)
            logits, mean, log_sigma = logits[:, -1], mean[:, -1], log_sigma[:, -1]
            comp = torch.distributions.Categorical(
                logits=logits / max(temperature, 1e-6)).sample()
            idx = comp[:, None, None].expand(-1, 1, cfg.n_features)
            mu = mean.gather(1, idx).squeeze(1)
            sigma = log_sigma.gather(1, idx).squeeze(1).exp() * sigma_scale
            x = mu + sigma * torch.randn_like(mu)
            prev = x.unsqueeze(1) if prev is None else torch.cat([prev, x.unsqueeze(1)], dim=1)
        return prev.reshape(B, n_traj, cfg.horizon, cfg.n_features)


# ---------------------------------------------------------------------------
# inference wrapper
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Loads a checkpoint and generates OHLC trajectories from raw arrays."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        ck = torch.load(ckpt_path, map_location=self.device)
        self.cfg = SimpleNamespace(**ck["cfg"])
        self.mu = np.asarray(ck["stats"]["mean"], dtype=np.float32)
        self.sd = np.asarray(ck["stats"]["std"], dtype=np.float32)
        self.model = CandleGen(self.cfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def _prep(self, query_ohlc, analog_patterns, analog_futures, scores):
        cfg = self.cfg
        query_ohlc = np.asarray(query_ohlc, dtype=np.float64)
        w = query_ohlc.shape[0]
        if w > cfg.max_pattern_len:
            query_ohlc = query_ohlc[-cfg.max_pattern_len:]
            w = cfg.max_pattern_len

        qp_feat_valid = ohlc_to_features(query_ohlc)
        s = vol_scale(qp_feat_valid, cfg.vol_floor)
        last_close = float(query_ohlc[-1, 3])

        qp_feat = np.zeros((cfg.max_pattern_len, 4), dtype=np.float32)
        qp_feat[:w] = qp_feat_valid / s
        qp_mask = np.zeros(cfg.max_pattern_len, dtype=bool)
        qp_mask[:w] = True

        K = len(analog_patterns)
        L = cfg.max_pattern_len + cfg.horizon
        an_feat = np.zeros((K, L, 4), dtype=np.float32)
        an_mask = np.zeros((K, L), dtype=bool)
        an_seg = np.zeros((K, L), dtype=np.int64)
        for j in range(K):
            ap = np.asarray(analog_patterns[j], dtype=np.float64)[-cfg.max_pattern_len:]
            af = np.asarray(analog_futures[j], dtype=np.float64)[:cfg.horizon]
            aw = ap.shape[0]
            ap_feat = ohlc_to_features(ap)
            sa = vol_scale(ap_feat, cfg.vol_floor)
            af_feat = ohlc_to_features(af, prev_close=float(ap[-1, 3]))
            an_feat[j, :aw] = ap_feat / sa
            an_feat[j, aw:aw + af.shape[0]] = af_feat / sa
            an_mask[j, :aw + af.shape[0]] = True
            an_seg[j, aw:aw + af.shape[0]] = 1

        c = cfg.feat_clip
        qp_feat[qp_mask] = np.clip((qp_feat[qp_mask] - self.mu) / self.sd, -c, c)
        an_feat[an_mask] = np.clip((an_feat[an_mask] - self.mu) / self.sd, -c, c)

        batch = {
            "qp_feat": torch.from_numpy(qp_feat)[None],
            "qp_mask": torch.from_numpy(qp_mask)[None],
            "an_feat": torch.from_numpy(an_feat)[None],
            "an_mask": torch.from_numpy(an_mask)[None],
            "an_seg": torch.from_numpy(an_seg)[None],
            "an_score": torch.tensor(np.asarray(scores, dtype=np.float32))[None],
        }
        return {k: v.to(self.device) for k, v in batch.items()}, s, last_close

    def generate(self, query_ohlc, analog_patterns, analog_futures, scores,
                 n_traj: int = 64, temperature: float = 1.0,
                 sigma_scale: float = 1.0) -> np.ndarray:
        """Returns (n_traj, horizon, 4) raw OHLC continuing the query."""
        batch, s, last_close = self._prep(query_ohlc, analog_patterns,
                                          analog_futures, scores)
        feat = self.model.sample(batch, n_traj=n_traj, temperature=temperature,
                                 sigma_scale=sigma_scale)[0].cpu().numpy()
        feat = (feat * self.sd + self.mu) * s
        return np.stack([features_to_ohlc(f, last_close) for f in feat])
