"""Retrieval-conditioned autoregressive trajectory generator.

Architecture
  1. Query encoder      — transformer over the query pattern's candles.
  2. Analog encoder     — shared-weight transformer over each top-K analog's
                          [pattern ; future] sequence (segment embeddings mark
                          which part is which), mean-pooled to one token per
                          analog, enriched with its similarity score and rank.
  3. Decoder            — causal transformer over future candles that
                          cross-attends to [query tokens ; analog tokens].
  4. Mixture head       — per step, a mixture of diagonal Gaussians over the
                          4-dim candle vector (gap, body, upper, lower).
                          Sampling the head N times yields N distinct
                          plausible trajectories.

Training is teacher-forced on the actual historical future (NLL loss).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def _encoder(d_model, n_heads, ffn_mult, dropout, layers):
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, d_model * ffn_mult, dropout,
        batch_first=True, norm_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, layers)


class CandleGen(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        L_analog = cfg.max_pattern_len + cfg.horizon

        self.in_proj = nn.Linear(cfg.n_features, d)
        # analogs are patched: `analog_patch` adjacent candles form one token
        self.analog_in_proj = nn.Linear(cfg.n_features * cfg.analog_patch, d)
        self.pos_pattern = nn.Embedding(cfg.max_pattern_len, d)
        self.pos_analog = nn.Embedding(L_analog, d)
        self.pos_future = nn.Embedding(cfg.horizon, d)
        self.seg_emb = nn.Embedding(2, d)          # analog: 0=pattern, 1=future
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

        nn.init.normal_(self.start_token, std=0.02)
        # zero-init the mixture head: training starts from a uniform mixture of
        # unit Gaussians instead of random tiny sigmas that spike the first
        # few losses into the hundreds
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ---------------- encoding ----------------

    def encode(self, batch):
        """Build the cross-attention memory from query + analogs.
        Returns (memory (B, 100+K, d), memory_pad_mask (B, 100+K) True=pad)."""
        cfg = self.cfg
        qp, qp_mask = batch["qp_feat"], batch["qp_mask"]          # (B,100,4) (B,100)
        an, an_mask, an_seg = batch["an_feat"], batch["an_mask"], batch["an_seg"]

        # trim padding to the batch's longest sequences — valid steps are
        # contiguous from index 0, so this changes nothing but the work done
        # (attention over 150 padded steps was the dominant training cost).
        # Lengths are bucketed to multiples of pad_multiple so the backend
        # compiles kernels for a handful of shapes instead of one per window.
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

        # patch analogs: P adjacent candles -> one token (P^2 less attention)
        P = cfg.analog_patch
        Lp = L // P  # L is a multiple of pad_multiple, which P divides
        a = self.analog_in_proj(an.reshape(B * K, Lp, P * cfg.n_features))
        apos = torch.arange(Lp, device=an.device)
        seg = an_seg.reshape(B * K, Lp, P).amax(-1)   # patch is "future" if any half is
        a = a + self.pos_analog(apos)[None] + self.seg_emb(seg)
        a_mask = an_mask.reshape(B * K, Lp, P).any(-1)
        # fully-empty analog slots would NaN the encoder; give them one fake step
        empty = ~a_mask.any(dim=1)
        enc_mask = a_mask.clone()
        enc_mask[empty, 0] = True
        a = self.analog_enc(a, src_key_padding_mask=~enc_mask)
        denom = enc_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (a * enc_mask.unsqueeze(-1)).sum(dim=1) / denom  # (B*K, d)
        pooled = pooled.reshape(B, K, -1)

        rank = torch.arange(K, device=an.device)
        pooled = pooled + self.rank_emb(rank)[None] + self.score_proj(batch["an_score"].unsqueeze(-1))

        memory = torch.cat([q, pooled], dim=1)
        analog_pad = empty.reshape(B, K)
        memory_pad = torch.cat([~qp_mask, analog_pad], dim=1)
        return memory, memory_pad

    # ---------------- decoding ----------------

    def _decode(self, prev_candles, memory, memory_pad):
        """prev_candles (B, T, 4): candles 0..T-1 already emitted (teacher-forced
        or sampled). Returns mixture params for steps 0..T (start token included)."""
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

    def forward(self, batch):
        """Teacher-forced NLL of the actual future under the mixture head."""
        memory, memory_pad = self.encode(batch)
        tgt = batch["tgt"]                                        # (B, 50, 4)
        logits, mean, log_sigma = self._decode(tgt[:, :-1], memory, memory_pad)
        return mdn_nll(tgt, logits, mean, log_sigma)

    @torch.no_grad()
    def sample(self, batch, n_traj: int = 32, temperature: float = 1.0,
               sigma_scale: float = 1.0):
        """Draw n_traj trajectories per batch element. Returns (B, n_traj, 50, 4)
        in normalized feature space (caller un-standardizes and un-vol-scales)."""
        cfg = self.cfg
        memory, memory_pad = self.encode(batch)
        B, d = memory.size(0), memory.size(-1)
        memory = memory.repeat_interleave(n_traj, dim=0)
        memory_pad = memory_pad.repeat_interleave(n_traj, dim=0)

        prev = None
        for _ in range(cfg.horizon):
            logits, mean, log_sigma = self._decode(prev, memory, memory_pad)
            logits, mean, log_sigma = logits[:, -1], mean[:, -1], log_sigma[:, -1]
            comp = torch.distributions.Categorical(
                logits=logits / max(temperature, 1e-6)).sample()   # (B*n,)
            idx = comp[:, None, None].expand(-1, 1, cfg.n_features)
            mu = mean.gather(1, idx).squeeze(1)
            sigma = log_sigma.gather(1, idx).squeeze(1).exp() * sigma_scale
            x = mu + sigma * torch.randn_like(mu)
            prev = x.unsqueeze(1) if prev is None else torch.cat([prev, x.unsqueeze(1)], dim=1)
        return prev.reshape(B, n_traj, cfg.horizon, cfg.n_features)


def mdn_nll(target, logits, mean, log_sigma):
    """Mixture-of-diagonal-Gaussians negative log likelihood, averaged over
    batch, time, and per-step dims. target (B,T,F); params (B,T,M,·)."""
    t = target.unsqueeze(2)                                       # (B,T,1,F)
    inv_var = torch.exp(-2.0 * log_sigma)
    log_prob = -0.5 * ((t - mean) ** 2 * inv_var + 2.0 * log_sigma
                       + math.log(2.0 * math.pi))                 # (B,T,M,F)
    log_prob = log_prob.sum(dim=-1) + F.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_prob, dim=-1).mean()
