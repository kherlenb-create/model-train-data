"""Sample trajectories from a trained checkpoint and plot them vs the actual path.

For each requested validation sample this draws N trajectories, reconstructs
full OHLC in price space, and renders:
  - top panel: query history as candles, a quantile fan of the sampled close
    paths, and the actual historical future for comparison
  - bottom row: the actual future as candles next to three sampled OHLC
    trajectories, so wick/body behaviour can be eyeballed directly

Usage:
  ~/Documents/model/.venv/bin/python generate.py --ckpt checkpoints/best.pt
  ~/Documents/model/.venv/bin/python generate.py --pos 12 --k 100 --query-len 30 --num-traj 128
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from config import Config
from data import CandleTrajectoryDataset, chrono_split, features_to_ohlc
from model import CandleGen
from train import pick_device

# palette (validated defaults): blue sequential fan, blue/red candle polarity,
# neutral ink for the actual path so identity never rides on the fan's hue
FAN_OUTER = "#cde2fb"   # 5-95% band
FAN_INNER = "#9ec5f4"   # 25-75% band
FAN_MEDIAN = "#256abf"
UP, DOWN = "#2a78d6", "#e34948"
INK = "#1a1a18"
GRID = "#e5e4e0"


def draw_candles(ax, x0, ohlc, width=0.6):
    for t in range(ohlc.shape[0]):
        o, h, l, c = ohlc[t]
        color = UP if c >= o else DOWN
        x = x0 + t
        ax.plot([x, x], [l, h], color=color, lw=0.8, zorder=2)
        body_lo, body_hi = min(o, c), max(o, c)
        ax.add_patch(Rectangle((x - width / 2, body_lo), width,
                               max(body_hi - body_lo, 1e-12),
                               facecolor=color, edgecolor="none", zorder=3))


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors="#6b6a66", labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--pos", type=int, nargs="*", default=None,
                    help="validation-set positions; default 4 spread across val")
    ap.add_argument("--k", type=int, default=50, help="number of analogs to condition on")
    ap.add_argument("--query-len", type=int, default=None,
                    help="tail-crop the query to this length (default: full window)")
    ap.add_argument("--num-traj", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--out", default="samples_out")
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if hasattr(Config, k)})
    stats = ck["stats"]
    model = CandleGen(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    _, val_idx = chrono_split(cfg)
    ds = CandleTrajectoryDataset(cfg, val_idx, stats)
    positions = args.pos or list(np.linspace(0, len(ds) - 1, 4).astype(int))
    os.makedirs(args.out, exist_ok=True)
    mu = np.asarray(stats["mean"], dtype=np.float32)
    sd = np.asarray(stats["std"], dtype=np.float32)

    for pos in positions:
        item = ds[(pos, args.k, args.query_len)]
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
        with torch.no_grad():
            feat = model.sample(batch, n_traj=args.num_traj,
                                temperature=args.temperature,
                                sigma_scale=args.sigma_scale)
        feat = feat[0].cpu().numpy()                       # (N, 50, 4) normalized
        s = float(item["s"])
        last_close = float(item["last_close"])
        feat = feat * sd + mu                              # un-standardize
        feat = feat * s                                    # un-vol-scale

        trajs = np.stack([features_to_ohlc(f, last_close) for f in feat])  # (N,50,4)
        closes = trajs[:, :, 3]
        q05, q25, q50, q75, q95 = np.percentile(closes, [5, 25, 50, 75, 95], axis=0)

        hist, actual = ds.raw(pos)
        if args.query_len:
            hist = hist[-args.query_len:]
        hist = hist[-60:]                                  # keep the plot readable
        H, F = hist.shape[0], cfg.horizon
        fx = np.arange(H, H + F)
        inside = np.mean((actual[:, 3] >= q05) & (actual[:, 3] <= q95))
        print(f"pos {pos} (sample {int(item['index'])}): "
              f"actual close inside 5-95% fan {inside*100:.0f}% of steps")

        fig = plt.figure(figsize=(11, 7.5), facecolor="white")
        gs = fig.add_gridspec(2, 4, height_ratios=[1.7, 1], hspace=0.35, wspace=0.25)

        ax = fig.add_subplot(gs[0, :])
        draw_candles(ax, 0, hist)
        ax.fill_between(fx, q05, q95, color=FAN_OUTER, label="5–95% of samples", zorder=1)
        ax.fill_between(fx, q25, q75, color=FAN_INNER, label="25–75% of samples", zorder=1)
        ax.plot(fx, q50, color=FAN_MEDIAN, lw=2, label="Sampled median", zorder=4)
        ax.plot(fx, actual[:, 3], color=INK, lw=2, label="Actual close", zorder=5)
        ax.axvline(H - 0.5, color=GRID, lw=1)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title(f"Sample {int(item['index'])} — K={args.k} analogs, "
                     f"query={H} bars, {args.num_traj} trajectories",
                     fontsize=11, color=INK, loc="left")
        style(ax)

        panels = [("Actual future", actual)] + [
            (f"Sampled #{j + 1}", trajs[j]) for j in range(3)]
        for col, (title, ohlc) in enumerate(panels):
            axs = fig.add_subplot(gs[1, col])
            draw_candles(axs, 0, ohlc)
            axs.set_title(title, fontsize=9, color="#6b6a66", loc="left")
            style(axs)

        out = os.path.join(args.out, f"traj_{int(item['index'])}_k{args.k}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
