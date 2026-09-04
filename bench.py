"""Benchmark one training step across the K/query-length regimes.

Run it yourself whenever you change config or code:
  ~/Documents/model/.venv/bin/python bench.py

Prints warm ms/step and samples/s per regime, with and without bf16 autocast,
plus a rough full-epoch / full-run estimate.
"""

import time

import numpy as np
import torch

from config import Config
from data import CandleTrajectoryDataset, KVariableBatchSampler, chrono_split, compute_stats
from model import CandleGen
from train import pick_device


def main():
    cfg = Config()
    device = pick_device()
    print(f"device: {device}")
    tr, _ = chrono_split(cfg)
    stats = compute_stats(cfg, tr)
    ds = CandleTrajectoryDataset(cfg, tr, stats)
    torch.manual_seed(0)
    model = CandleGen(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def step(b, amp):
        b = {k: v.to(device) for k, v in b.items()}
        if amp and device.type in ("mps", "cuda"):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model(b)
        else:
            loss = model(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync()

    # representative (K, batch size from the sampler's rule, query crop)
    sampler = KVariableBatchSampler(cfg, len(ds), cfg.seed)
    cases = [(10, sampler._batch_size(10), 30),
             (50, sampler._batch_size(50), 30),
             (100, sampler._batch_size(100), 30),
             (200, sampler._batch_size(200), 30),
             (50, sampler._batch_size(50), 100)]
    batches = {c: torch.utils.data.default_collate([ds[(i, c[0], c[2])] for i in range(c[1])])
               for c in cases}

    print("warming up (kernel compilation)...")
    for b in batches.values():
        step(b, amp=False)
        step(b, amp=True)

    for amp in (False, True):
        print(f"\n--- bf16 autocast: {amp}")
        rates = []
        for (K, B, crop), b in batches.items():
            t0 = time.time()
            n = 5
            for _ in range(n):
                step(b, amp)
            dt = (time.time() - t0) / n
            rates.append(B / dt)
            print(f"  K={K:3d} B={B:2d} qlen={crop:3d}: {dt*1000:7.0f} ms/step  "
                  f"{B/dt:6.1f} samp/s")
        mean_rate = float(np.mean(rates))
        epoch_s = len(ds) / mean_rate
        print(f"  ~epoch: {epoch_s/60:.1f} min   ~{cfg.epochs} epochs: "
              f"{epoch_s*cfg.epochs/3600:.1f} h")


if __name__ == "__main__":
    main()
