"""Central configuration for the trajectory-generation pipeline."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- data ----
    data_dir: str = "candles_7y"
    max_pattern_len: int = 300          # maxw in meta.npz — MUST be >= the dataset's
                                        # widest q_window, and sizes pos_pattern /
                                        # pos_analog, so changing it forces a fresh
                                        # dataset build AND training from scratch
    horizon: int = 50                   # future candles to generate (always 50)
    max_analogs: int = 200              # analog slots stored per sample
    # K (number of analogs) the scanner can hand the model at inference:
    k_choices: tuple = (10, 20, 50, 80, 100, 150, 200)
    val_k: int = 50                     # fixed K for comparable val NLL across epochs
    # query lengths the scanner can produce: 1..50, then 60..100 by 10, then the
    # long windows (must mirror LENGTHS in build_analogs.py — a crop longer than a
    # sample's own window is a no-op, so extra values here only cost sampling mass)
    query_len_choices: tuple = (tuple(range(1, 51))
                                + (60, 70, 80, 90, 100, 130, 160, 200, 250, 300))
    crop_prob: float = 0.7              # prob. of tail-cropping the query during training
    val_frac: float = 0.10              # chronological tail used for validation
    vol_floor: float = 1e-4             # floor for the per-sample volatility scale
    feat_clip: float = 10.0             # winsorize standardized features at +/- this
    stats_path: str = "feature_stats.json"

    # ---- model ----
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 3                 # query-pattern encoder depth
    analog_layers: int = 2              # shared analog encoder depth
    dec_layers: int = 4                 # autoregressive decoder depth
    ffn_mult: int = 4
    dropout: float = 0.1
    n_mix: int = 5                      # mixture components in the output head
    n_features: int = 4                 # (gap, body, upper wick, lower wick)
    analog_patch: int = 2               # candles merged per analog-encoder token
                                        # (halves the dominant attention cost)
    pad_multiple: int = 16              # bucket trimmed lengths to multiples of
                                        # this so MPS re-compiles fewer kernel shapes

    # ---- training ----
    batch_size: int = 32                # reference batch size at K = k_ref
    k_ref: int = 50                     # batch size scales ~ k_ref/K to keep step cost flat
    min_batch: int = 8
    max_batch: int = 64
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 42
    ckpt_dir: str = "checkpoints_7y"
    log_every: int = 10                 # steps between train-loss log lines

    # ---- sampling ----
    temperature: float = 1.0            # scales mixture-component logits
    sigma_scale: float = 1.0            # scales Gaussian std at sampling time
