# Retrieval-Conditioned Candle Trajectory Generator

Generative model that takes a **scanned query pattern** plus **K similar
historical patterns (analogs) and what actually followed them**, and samples
plausible **50-candle full-OHLC trajectories** for the query. Training is
teacher-forced on the actual historical future of each query, so the model
learns "patterns that looked like this tended to continue like that."

Everything you need to *use* a trained model is `candle_model.py` + a `.pt`
checkpoint — the checkpoint embeds both the config and the normalization
stats, so no other file in this repo is required at inference time.

---

## Contents

| file | role |
|---|---|
| `config.py` | every hyperparameter (dataclass `Config`) |
| `data.py` | dataset, candle↔feature transforms, K-variable batch sampler, chrono split, feature stats |
| `model.py` | `CandleGen` — the training-time architecture + MDN NLL loss |
| `train.py` | training loop (resumable, bf16 autocast, cosine LR) |
| `generate.py` | sample from a checkpoint and plot fan charts vs. the actual future |
| `candle_model.py` | **standalone** inference wrapper (`TrajectoryGenerator`) — copy this + a `.pt` anywhere |
| `bench.py` | per-step speed benchmark across K / query-length regimes |
| `feature_stats.json` | cached global feature mean/std (auto-generated) |
| `checkpoints/`, `checkpoints_1y/` | trained weights (see below) |
| `candles_train/`, `candles_1y/` | the two prepared datasets |
| `samples_out/` | plots written by `generate.py` |

---

## 1. Trained models

Two checkpoints ship with the repo. **Both use the identical architecture and
identical hyperparameters** — they differ only in which dataset they were
trained on (the analog-retrieval filter) and how long they trained.

| | `checkpoints/` | `checkpoints_1y/` |
|---|---|---|
| dataset | `candles_train/` (filter `all`) | `candles_1y/` (filter `1y`) |
| analog pool | analogs may come from **any** point in history | analogs restricted to the **last 1 year** relative to the query |
| samples | 9 991 | 10 000 |
| mean analogs/query | 196.3 / 200 | 177.8 / 200 |
| NaN (empty) analog slots | 1.8 % | 11.1 % |
| median similarity score | 0.907 | 0.875 |
| epochs completed | `best.pt` @ epoch 8, `last.pt` @ epoch 22 (of 40 — **run was stopped early**) | **40 / 40 — fully trained** |
| optimizer steps | 2 340 (best) / 6 468 (last) | 11 852 |
| **best val NLL** | **4.409** | **2.472** |
| parameters | 8.31 M | 8.31 M |
| file size | ~99.8 MB each | ~99.8 MB each |

**Which one to use:** `checkpoints_1y/best.pt`. It is the only fully-trained
run (40/40 epochs) and its validation NLL is far lower (2.47 vs 4.41). The
`checkpoints/` run was interrupted at epoch 22 and its `best.pt` is only an
epoch-8 snapshot. The `all`-filter data is not worse in principle — that run
simply never finished.

> Note: NLL values are only comparable *within* a dataset (different data →
> different target distribution). The 2.47 vs 4.41 gap is dominated by the
> `checkpoints/` run being unfinished, not by `1y` being intrinsically easier.

### Embedded config in every checkpoint

```python
{"model", "opt", "sched", "epoch", "step", "best_val", "cfg", "stats"}
```

- `cfg` — a dict of the full `Config` used for that run (`vars(cfg)`).
- `stats` — `{"mean": [...4], "std": [...4]}`, the global feature
  standardization used for that run.
- `opt` / `sched` — optimizer + LR-scheduler state, so training is resumable.

Stats baked into the checkpoints:

| checkpoint | mean (gap, body, upper, lower) | std |
|---|---|---|
| `checkpoints/best.pt` | `[0.0, 0.0, 0.17769, 0.18763]` | `[0.18822, 0.79065, 0.31452, 0.33943]` |
| `checkpoints/last.pt` | `[0.0, 0.0, 0.17554, 0.18532]` | `[0.18062, 0.75884, 0.29532, 0.31715]` |
| `checkpoints_1y/*.pt` | `[0.0, 0.0, 0.17554, 0.18532]` | `[0.18062, 0.75884, 0.29532, 0.31715]` |

⚠️ The `feature_stats.json` currently on disk matches the **second** row
(`checkpoints_1y` / `checkpoints/last.pt`), **not** `checkpoints/best.pt`.
`candle_model.py` and `generate.py` both read stats *from the checkpoint*, so
inference is always correct regardless. But if you resume `checkpoints/best.pt`
with `train.py`, delete `feature_stats.json` first or the stats will silently
disagree with what those weights were trained under.

---

## 2. Data

Both datasets cover **49 symbols** across equities, FX, crypto, indices and
metals (AAPL, MSFT, NVDA, TSLA … EUR_USD, GBP_JPY, USD_JPY … BTCUSDT, ETHUSDT,
SOLUSDT … SPX500_USD, NAS100_USD, XAU_USD, XAG_USD, XCU_USD, BCO_USD).
Query end timestamps span **2010-01-06 → 2026-07-02**.

| file | shape | contents |
|---|---|---|
| `q_pattern.npy` | (N, 100, 4) | query pattern, NaN-padded tail, raw OHLC |
| `q_future.npy` | (N, 50, 4) | actual future — the training target |
| `a_pattern.npy` | (N, 1, 200, 100, 4) | analog patterns (score-sorted desc) |
| `a_future.npy` | (N, 1, 200, 50, 4) | what followed each analog |
| `meta.npz` | — | `scores`, `n_analogs`, `q_window`, `q_end_ts`, `q_symbol`, `symbols`, `a_sym_id`, `a_end_ts`, `filters`, `ks`, `horizon`, `maxw` |

`N` = 9 991 (`candles_train`) / 10 000 (`candles_1y`).
Query windows (`q_window`) range 10–100 candles; analog counts 20–200.
Similarity scores are correlation-like, roughly in `[-0.9, 1.0]`; invalid analog
slots are NaN.

Arrays are memory-mapped (`mmap_mode="r"`) — the 3.2 GB `a_pattern.npy` is
never fully loaded.

### Candle parameterization

Each bar becomes 4 log-space numbers that always reconstruct to a *valid*
candle (`data.py` / `candle_model.py`):

- `gap   = log(open / prev_close)`
- `body  = log(close / open)`
- `upper = log(high / max(open, close))` ≥ 0  (upper wick)
- `lower = log(min(open, close) / low)` ≥ 0  (lower wick)

Because `high` and `low` are built *outward* from `max/min(open, close)` with a
non-negative exponent, no sampled vector can ever produce an invalid bar.

### Normalization

1. Divide every sequence by its **own pattern's** volatility scale `s` —
   `max(1.4826·MAD(close-to-close returns), mean candle range, vol_floor)`.
   The range fallback keeps flat-close patterns from exploding and is the only
   estimate available for 1–4 bar queries.
2. Standardize with global per-channel stats (median + winsorized std over a
   2 000-sample train subsample), cached in `feature_stats.json`.
3. Winsorize at ±10 (`feat_clip`).

Each analog is normalized by *its own* vol scale, so analogs from BTC and from
EUR_USD land on the same footing.

### Split

Chronological on `q_end_ts`: oldest 90 % train, most recent 10 % validation —
no future leakage. Samples with NaN or non-positive prices inside the valid
query region are dropped (one such target NaNs the loss and destroys the
weights on the next backward pass).

---

## 3. Architecture (`model.py`, 8.31 M params)

1. **Query encoder** — 3-layer transformer encoder over the query pattern's
   candles (learned positional embeddings, pre-norm, GELU).
2. **Analog encoder** — 2-layer shared-weight transformer over each analog's
   `[pattern ; future]` sequence. Segment embeddings mark which half is which.
   Adjacent candles are **patched 2→1 token** (`analog_patch=2`), cutting the
   dominant attention cost ~4×. Mean-pooled to one token per analog, then
   enriched with a learned **rank embedding** and a projected **similarity
   score**.
3. **Decoder** — 4-layer causal transformer over the future candles,
   cross-attending to `[query tokens ; analog tokens]`.
4. **Mixture head** — per step, a mixture of **5 diagonal Gaussians** over the
   4-dim candle vector. Output width = `n_mix * (1 + 2 * n_features)` = 45.
   Sampling it N times yields N distinct trajectories.

Loss: mixture-of-Gaussians **negative log-likelihood**, teacher-forced on the
actual historical future, averaged over batch × time × dims.

Implementation details worth knowing:
- The mixture head is **zero-initialized**, so training starts from a uniform
  mixture of unit Gaussians instead of random tiny sigmas that spike early loss
  into the hundreds.
- `log_sigma` is clamped to `[-7, 3]`.
- Padding is trimmed to the batch's longest sequence and bucketed to multiples
  of `pad_multiple=16`, so MPS/CUDA compile a handful of kernel shapes instead
  of one per window length.
- Fully-empty analog slots get one fake unmasked step (otherwise the encoder
  NaNs) and are then masked out of cross-attention.

### Variable inference shapes

The scanner can hand the model any of:
- **K analogs** ∈ {10, 20, 50, 80, 100, 150, 200}
- **query length** ∈ {1…50, 60, 70, 80, 90, 100}

Training mirrors this: each batch draws one K (batch size scales ~`k_ref/K` to
keep step cost flat) and each sample gets a random tail-crop of its query with
probability 0.7 (analog patterns are cropped to match). Validation always runs
at K=50 with full windows so the NLL is comparable across epochs.

---

## 4. All hyperparameters (`config.py`)

### Data
| param | value | meaning |
|---|---|---|
| `data_dir` | `"candles_train"` | dataset directory — **set to `"candles_1y"` to use the 1y data** |
| `max_pattern_len` | 100 | `maxw` in `meta.npz` |
| `horizon` | 50 | future candles generated — always 50 |
| `max_analogs` | 200 | analog slots stored per sample |
| `k_choices` | `(10,20,50,80,100,150,200)` | K values the scanner may supply |
| `val_k` | 50 | fixed K for comparable val NLL |
| `query_len_choices` | `1..50, 60,70,80,90,100` | query lengths supported |
| `crop_prob` | 0.7 | probability of tail-cropping a training query |
| `val_frac` | 0.10 | chronological tail used for validation |
| `vol_floor` | 1e-4 | floor on the per-sample volatility scale |
| `feat_clip` | 10.0 | winsorize standardized features at ±this |
| `stats_path` | `"feature_stats.json"` | cached global stats |

### Model
| param | value |
|---|---|
| `d_model` | 256 |
| `n_heads` | 4 |
| `enc_layers` | 3 (query encoder) |
| `analog_layers` | 2 (shared analog encoder) |
| `dec_layers` | 4 (autoregressive decoder) |
| `ffn_mult` | 4 (FFN width = 1024) |
| `dropout` | 0.1 |
| `n_mix` | 5 mixture components |
| `n_features` | 4 (gap, body, upper wick, lower wick) |
| `analog_patch` | 2 candles per analog token |
| `pad_multiple` | 16 (length bucketing) |

### Training
| param | value | meaning |
|---|---|---|
| `batch_size` | 32 | reference batch size at K = `k_ref` |
| `k_ref` | 50 | batch size scales ~`k_ref/K` |
| `min_batch` / `max_batch` | 8 / 64 | clamps on the scaled batch size |
| `epochs` | 40 |
| `lr` | 3e-4 | AdamW, cosine decay to 5 % after warmup |
| `weight_decay` | 0.01 |
| `warmup_steps` | 500 | linear warmup |
| `grad_clip` | 1.0 | global grad-norm clip |
| `num_workers` | 2 | DataLoader workers |
| `seed` | 42 |
| `ckpt_dir` | `"checkpoints"` | **set to `"checkpoints_1y"` when training on 1y data** |
| `log_every` | 10 | steps between train-loss log lines |

### Sampling
| param | default | meaning |
|---|---|---|
| `temperature` | 1.0 | scales the **mixture-component logits** |
| `sigma_scale` | 1.0 | scales the **Gaussian std** at sampling time |

---

## 5. Temperature and `sigma_scale` — how sampling is controlled

The head emits, per step: 5 component logits, 5×4 means, 5×4 log-sigmas.
Sampling one candle is two draws:

```python
comp  = Categorical(logits = logits / temperature).sample()   # which component
mu    = mean[comp]
sigma = exp(log_sigma[comp]) * sigma_scale
x     = mu + sigma * randn()                                  # the candle
```

So the two knobs are **independent**:

| knob | acts on | effect |
|---|---|---|
| `temperature` | *which* mixture component is chosen | `>1` flattens the categorical → more regime diversity across trajectories (some bullish, some bearish, some choppy). `<1` sharpens it → trajectories concentrate on the model's favourite regime. `→0` always picks the argmax component. |
| `sigma_scale` | *how wide* the chosen Gaussian is | `>1` wilder individual candles, wider fan. `<1` tighter, smoother candles. `0` = deterministic mean of the chosen component. |

Practical settings:

| goal | temperature | sigma_scale |
|---|---|---|
| **Calibrated** — the fan should actually cover the truth ~90 % of the time | **1.0** | **1.0** |
| Explore more distinct regimes, same candle realism | 1.3–1.8 | 1.0 |
| Cleaner "central scenario" candles, still multi-regime | 1.0 | 0.6–0.8 |
| A few sharp modal scenarios | 0.5–0.7 | 0.8–1.0 |
| Near-deterministic modal path | 0.1 | 0.0–0.2 |
| Stress / tail scenarios | 1.5 | 1.3–1.8 |

Leave both at **1.0** whenever you care about probabilities being right
(coverage, quantiles, risk numbers). Anything else deliberately mis-calibrates
the fan — turning them up widens it, turning them down makes it
overconfident. Use them for presentation and scenario generation, not for
measuring probability.

Temperature is applied as `logits / max(temperature, 1e-6)`, so a value of 0 is
safe (it degenerates to argmax rather than dividing by zero).

---

## 6. Setup

All commands use the shared venv: `~/Documents/model/.venv/bin/python`
(Python 3.14).

```bash
pip install -r requirements.txt      # torch>=2.12, numpy>=2.4.6, matplotlib>=3.10.9
```

Device is auto-selected: CUDA → MPS → CPU. bf16 autocast is auto-detected on
MPS/CUDA and can be forced with `--amp` / `--no-amp`.

---

## 7. Using the model (inference)

### The 3-line version

```python
from candle_model import TrajectoryGenerator

gen = TrajectoryGenerator("checkpoints_1y/best.pt")   # device auto-picked
trajs = gen.generate(query_ohlc, analog_patterns, analog_futures, scores,
                     n_traj=64, temperature=1.0, sigma_scale=1.0)
# trajs: (64, 50, 4) raw OHLC continuing from query_ohlc's last close
```

### Full signature

```python
TrajectoryGenerator(ckpt_path: str, device: str | None = None)

gen.generate(
    query_ohlc,        # (w, 4) raw OHLC. w in {1..50, 60,70,80,90,100}.
                       #   longer than 100 -> tail-truncated automatically
    analog_patterns,   # list of K arrays, each (wi, 4) raw OHLC,
                       #   sorted by similarity DESCENDING
    analog_futures,    # list of K arrays, each (50, 4) raw OHLC
    scores,            # list/array of K similarity scores (same order)
    n_traj=64,         # how many trajectories to draw
    temperature=1.0,
    sigma_scale=1.0,
) -> np.ndarray        # (n_traj, 50, 4) raw OHLC
```

Requirements on the inputs:
- **All prices strictly positive** (log space). Zeros/negatives get floored to
  1e-12 and will produce garbage.
- **Analogs must be score-sorted descending** — the model has a learned rank
  embedding and takes ordering as signal.
- `len(analog_patterns) == len(analog_futures) == len(scores) == K`, and
  K ≤ 200. K need not be one of `k_choices` — those are just what training
  saw — but stay near them for best results.
- Analog futures should be 50 candles; longer is truncated.
- Column order is **(open, high, low, close)**.

### Worked example

```python
import numpy as np
from candle_model import TrajectoryGenerator

gen = TrajectoryGenerator("checkpoints_1y/best.pt")

query   = np.array([...])                 # (30, 4) the last 30 bars you scanned
patterns = [np.array([...]) for _ in range(50)]   # (30, 4) each
futures  = [np.array([...]) for _ in range(50)]   # (50, 4) each
scores   = [0.98, 0.97, ...]                      # descending, len 50

trajs = gen.generate(query, patterns, futures, scores, n_traj=200)

closes = trajs[:, :, 3]                             # (200, 50)
q05, q50, q95 = np.percentile(closes, [5, 50, 95], axis=0)
last = query[-1, 3]
print("median 50-bar return:", q50[-1] / last - 1)
print("5-95% terminal range:", q05[-1] / last - 1, q95[-1] / last - 1)
print("P(up in 50 bars):", (closes[:, -1] > last).mean())
```

### Loading a specific device / batching

```python
gen = TrajectoryGenerator("checkpoints_1y/best.pt", device="cpu")
```

`TrajectoryGenerator.generate` handles **one query at a time**. For many
queries, loop — the model is re-used, only `_prep` runs per query. If you need
true batching, build the batch dict yourself and call
`gen.model.sample(batch, n_traj=..., ...)` directly; it returns
`(B, n_traj, 50, 4)` in normalized feature space, which you un-standardize with
`feat * gen.sd + gen.mu`, multiply by each sample's vol scale `s`, and convert
via `features_to_ohlc(f, last_close)`.

Cost scales as `n_traj × 50` decoder steps, so `n_traj=200` on a 50-analog
query is a couple of seconds on MPS.

### Plot-and-inspect CLI

```bash
~/Documents/model/.venv/bin/python generate.py --ckpt checkpoints_1y/best.pt
~/Documents/model/.venv/bin/python generate.py --ckpt checkpoints_1y/best.pt \
    --pos 12 --k 100 --query-len 30 --num-traj 128 \
    --temperature 1.0 --sigma-scale 1.0 --out samples_out
```

| flag | default | meaning |
|---|---|---|
| `--ckpt` | `checkpoints/best.pt` | checkpoint to load |
| `--pos` | 4 spread across val | validation-set positions to plot |
| `--k` | 50 | analogs to condition on |
| `--query-len` | full window | tail-crop the query |
| `--num-traj` | 64 | trajectories sampled |
| `--temperature` | 1.0 | |
| `--sigma-scale` | 1.0 | |
| `--out` | `samples_out` | output directory |

It prints the fraction of steps where the actual close falls inside the sampled
5–95 % fan — a **calibration check that should read ≈90 %** on a well-trained
model at temperature 1.0 / sigma_scale 1.0. Materially below 90 % → the model
is overconfident; materially above → too wide. It also renders the fan chart
plus sampled OHLC candle panels next to the actual future.

Note `generate.py` reads the dataset via `Config()` (i.e. `data_dir` from
`config.py`), so to plot the 1y checkpoint against 1y validation samples set
`data_dir = "candles_1y"` in `config.py` first.

---

## 8. Training from scratch

```bash
# train — writes {ckpt_dir}/last.pt every epoch, best.pt on val improvement
~/Documents/model/.venv/bin/python train.py

# override the two most common knobs
~/Documents/model/.venv/bin/python train.py --epochs 60 --lr 2e-4

# force / disable bf16 autocast (default: auto-detect)
~/Documents/model/.venv/bin/python train.py --amp
~/Documents/model/.venv/bin/python train.py --no-amp

# smoke test: 25 optimizer steps, one epoch
~/Documents/model/.venv/bin/python train.py --epochs 1 --max-steps 25
```

CLI flags: `--epochs`, `--lr`, `--max-steps`, `--resume`, `--amp/--no-amp`.
**Everything else is edited in `config.py`** — there is no flag for `d_model`,
`batch_size`, `dropout`, `data_dir`, `ckpt_dir`, etc.

To train on the other dataset, edit `config.py`:

```python
data_dir: str = "candles_1y"
ckpt_dir: str = "checkpoints_1y"
```

and **delete `feature_stats.json`** so stats are recomputed for that data.

What a run prints per step: `K`, batch size, loss, LR, samples/s, GPU vs. data
time share, and an ETA computed from a 30-step rolling window (so warmup steps
don't pollute it).

Check speed before committing to a long run:

```bash
~/Documents/model/.venv/bin/python bench.py
```

It reports ms/step and samples/s for K ∈ {10, 50, 100, 200} with and without
bf16, plus an epoch and full-run estimate.

---

## 9. Resuming and fine-tuning

### Resume an interrupted run

Restores model, optimizer, LR-scheduler, epoch, step and best-val — it
continues exactly where it stopped:

```bash
~/Documents/model/.venv/bin/python train.py --resume checkpoints/last.pt
```

Two things to get right:
- `cfg.epochs` must be **≥** the checkpoint's epoch or the loop body never
  runs. The `checkpoints/` run stopped at epoch 22 of 40, so a plain
  `train.py --resume checkpoints/last.pt` picks up at epoch 23 and finishes
  the remaining 18.
- The LR schedule is a function of `steps_per_epoch × cfg.epochs`. If you
  change `--epochs` on resume, the cosine curve is re-derived over the new
  total — the loaded scheduler state then sits at a different point on a
  different curve. Keep `--epochs` the same as the original run unless you
  intend that.

**To finish the unfinished `all`-filter model:**

```bash
# config.py must have data_dir="candles_train", ckpt_dir="checkpoints"
rm feature_stats.json      # stats on disk are the 1y ones; last.pt's stats match,
                           # but recomputing on candles_train is the safe path
~/Documents/model/.venv/bin/python train.py --resume checkpoints/last.pt
```

### Fine-tune on new data

`train.py --resume` loads optimizer + scheduler state too, which is wrong for
fine-tuning — you want fresh optimizer state and a small constant-ish LR. The
minimal recipe is a short script that loads *only* the weights:

```python
# finetune.py
import numpy as np, torch, os
from torch.utils.data import DataLoader
from config import Config
from data import CandleTrajectoryDataset, KVariableBatchSampler, chrono_split, compute_stats
from model import CandleGen
from train import pick_device, evaluate

cfg = Config()
cfg.data_dir  = "candles_1y"        # <- your new dataset
cfg.ckpt_dir  = "checkpoints_ft"
cfg.epochs    = 5                   # fine-tunes are short
cfg.lr        = 5e-5                # ~1/6 of the pretraining LR
cfg.warmup_steps = 100
cfg.dropout   = 0.1

device = pick_device()
base = torch.load("checkpoints_1y/best.pt", map_location=device)

# CRITICAL: reuse the base checkpoint's normalization stats, not fresh ones.
# Recomputing stats on new data shifts the input distribution under weights
# that were trained on the old one.
stats = base["stats"]

train_idx, val_idx = chrono_split(cfg)
train_ds = CandleTrajectoryDataset(cfg, train_idx, stats)
val_ds   = CandleTrajectoryDataset(cfg, val_idx,   stats)
sampler  = KVariableBatchSampler(cfg, len(train_ds), cfg.seed)
train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.num_workers)
val_dl   = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

model = CandleGen(cfg).to(device)
model.load_state_dict(base["model"])          # weights only — no opt/sched state
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

os.makedirs(cfg.ckpt_dir, exist_ok=True)
best = float("inf")
model.train()
for epoch in range(cfg.epochs):
    sampler.set_epoch(epoch)
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    val = evaluate(model, val_dl, device)
    print(f"epoch {epoch+1}: val NLL {val:.4f}")
    state = {"model": model.state_dict(), "opt": opt.state_dict(),
             "sched": {}, "epoch": epoch, "step": 0,
             "best_val": min(best, val), "cfg": vars(cfg), "stats": stats}
    torch.save(state, f"{cfg.ckpt_dir}/last.pt")
    if val < best:
        best = val
        torch.save(state, f"{cfg.ckpt_dir}/best.pt")
```

The saved checkpoint keeps the same `{"model", "cfg", "stats"}` keys, so
`TrajectoryGenerator("checkpoints_ft/best.pt")` loads it unchanged.

**Fine-tuning rules that matter here:**

1. **Never recompute `stats`.** Carry `base["stats"]` forward. New global stats
   under old weights is the single easiest way to wreck a fine-tune. (This is
   why the script above never calls `compute_stats`.)
2. **Architecture must match exactly.** `d_model`, `n_heads`, `*_layers`,
   `ffn_mult`, `n_mix`, `n_features`, `analog_patch`, `max_pattern_len`,
   `horizon`, `max_analogs` are all baked into the weight shapes.
   `load_state_dict` will refuse a mismatch. You *may* freely change
   `dropout`, `lr`, `epochs`, `batch_size`, `crop_prob`, `k_choices`,
   `query_len_choices`, `val_k`, `warmup_steps` — none affect weight shapes.
3. **Use a lower LR** — 3e-5 to 1e-4 vs. the 3e-4 used for pretraining, with
   short warmup (~100 steps).
4. **Keep it short** — 3–10 epochs. Watch val NLL and stop when it turns up.
5. **Freezing (optional).** To adapt only the decoder/head to a new regime
   while keeping the retrieval encoders fixed:
   ```python
   for m in (model.query_enc, model.analog_enc):
       for p in m.parameters():
           p.requires_grad = False
   opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
   ```
6. **New data must be built to the same spec** — same directory layout, same
   `(N,100,4)` / `(N,50,4)` / `(N,1,200,100,4)` / `(N,1,200,50,4)` shapes, same
   `meta.npz` keys, raw positive OHLC, analogs score-sorted descending, NaN
   padding at the *head* of the query window (`q_window` gives the valid count)
   and at the tail of short analog patterns.
7. **Re-check calibration afterwards** with `generate.py` — the inside-fan
   percentage should still be ≈90 % at temperature 1.0.

### Changing the horizon or max pattern length

`horizon=50` and `max_pattern_len=100` are structural (positional embedding
sizes, dataset array shapes). Changing them requires rebuilding the dataset and
training from scratch — they are not fine-tunable.

---

## 10. Gotchas

- **Delete `feature_stats.json`** whenever you change anything affecting
  features or normalization, or switch datasets — it is cached and silently
  reused otherwise.
- **`data_dir` and `ckpt_dir` must be changed together** in `config.py`, or a
  1y run will overwrite the `checkpoints/` weights.
- Similarity ordering matters — feeding shuffled analogs degrades output, since
  rank is an input feature.
- The NaN-drop in `chrono_split` is not optional: a single NaN target
  permanently wipes the weights on the next backward pass.
- `checkpoints/best.pt` and `checkpoints/last.pt` have **different** embedded
  stats (best is an epoch-8 snapshot from before a stats recompute). Always let
  the checkpoint supply its own stats; never mix them.
- Val NLL is only comparable within one dataset and at the fixed `val_k=50`.
