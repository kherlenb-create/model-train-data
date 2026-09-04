"""Train the retrieval-conditioned trajectory generator.

Each training batch uses one K drawn from {10,20,50,80,100,150,200} analogs
and per-sample query tail-crops from {1..50, 60,70,80,90,100}, mirroring what
the scanner can hand the model at inference. Validation runs at a fixed
K (cfg.val_k) with full windows so the NLL is comparable across epochs.

Usage (with the shared venv):
  ~/Documents/model/.venv/bin/python train.py
  ~/Documents/model/.venv/bin/python train.py --epochs 60
  ~/Documents/model/.venv/bin/python train.py --max-steps 30 --epochs 1   # smoke test
"""

import argparse
import math
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import (CandleTrajectoryDataset, KVariableBatchSampler, chrono_split,
                  compute_stats)
from model import CandleGen


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    t0 = time.time()
    n_batches = len(loader)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch)
            bs = batch["tgt"].size(0)
            total += loss.item() * bs
            n += bs
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                print(f"    val {bi+1}/{n_batches} batches  "
                      f"nll {total/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    model.train()
    return total / max(n, 1)


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    if sec < 3600:
        return f"{sec//60}m{sec%60:02d}s"
    return f"{sec//3600}h{(sec%3600)//60:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps per epoch (smoke tests)")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="bfloat16 autocast on the forward pass (faster, cooler); "
                         "default: auto-detect support")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers; each one keeps its own mmap pages "
                         "resident, so on a memory-tight box fewer is faster")
    args = ap.parse_args()

    cfg = Config()
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device()
    if args.amp is None and device.type in ("mps", "cuda"):
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                (torch.ones(2, 2, device=device) @ torch.ones(2, 2, device=device)).sum().item()
            args.amp = True
        except Exception:
            args.amp = False
    args.amp = bool(args.amp)
    print(f"device: {device}  amp(bf16): {args.amp}")

    train_idx, val_idx = chrono_split(cfg)
    print("computing feature stats (cached in feature_stats.json after first run)...",
          flush=True)
    stats = compute_stats(cfg, train_idx)
    print(f"train {len(train_idx)}  val {len(val_idx)}  "
          f"stats mean {np.round(stats['mean'], 4).tolist()} std {np.round(stats['std'], 4).tolist()}")

    train_ds = CandleTrajectoryDataset(cfg, train_idx, stats)
    val_ds = CandleTrajectoryDataset(cfg, val_idx, stats)
    sampler = KVariableBatchSampler(cfg, len(train_ds), cfg.seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler,
                          num_workers=cfg.num_workers,
                          persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers,
                        persistent_workers=cfg.num_workers > 0)

    model = CandleGen(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = args.max_steps or len(sampler)
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.warmup_steps, total_steps))

    start_epoch, best_val, step = 0, float("inf"), 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_val, step = ck["epoch"] + 1, ck["best_val"], ck["step"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    model.train()

    run_t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        print(f"\n=== epoch {epoch+1}/{cfg.epochs} — ~{steps_per_epoch} batches ===",
              flush=True)
        t0, running, seen, n_samples = time.time(), 0.0, 0, 0
        data_t, compute_t = 0.0, 0.0
        window = deque(maxlen=30)   # (step_wall_time, batch_samples) of recent steps
        t_fetch = time.time()
        for bi, batch in enumerate(train_dl):
            data_t += time.time() - t_fetch
            if args.max_steps and bi >= args.max_steps:
                break
            t_step = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.amp and device.type in ("mps", "cuda"):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = model(batch)
            else:
                loss = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            # Sync explicitly and attribute the wait to its own bucket. Folding
            # it into compute_t reported "gpu 100%" while the real stall was
            # page-faulting mmap reads, which sent the diagnosis the wrong way.
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            compute_t += time.time() - t_step
            loss_val = loss.item()
            running += loss_val
            seen += 1
            n_samples += batch["tgt"].size(0)
            window.append((time.time() - t_step, batch["tgt"].size(0)))
            if seen <= 3 or seen % cfg.log_every == 0:
                el = time.time() - t0
                # rate/eta from the recent window, not since-epoch averages,
                # so warmup steps stop polluting the numbers
                w_time = sum(t for t, _ in window)
                w_samp = sum(s for _, s in window)
                eta = w_time / len(window) * (steps_per_epoch - seen)
                print(f"  step {bi+1}/{steps_per_epoch}  "
                      f"K={batch['an_feat'].shape[1]} B={batch['tgt'].size(0)}  "
                      f"loss {loss_val:.4f} (avg {running/seen:.4f})  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{w_samp/max(w_time, 1e-9):.1f} samp/s  "
                      f"gpu {compute_t/el*100:.0f}% data {data_t/el*100:.0f}%  "
                      f"eta {fmt_dur(eta)}", flush=True)
            t_fetch = time.time()

        print(f"  validating ({len(val_dl)} batches, K={cfg.val_k})...", flush=True)
        val_nll = evaluate(model, val_dl, device)
        dt = time.time() - t0
        done = epoch - start_epoch + 1
        left = cfg.epochs - epoch - 1
        run_eta = (time.time() - run_t0) / done * left
        print(f"epoch {epoch+1}/{cfg.epochs}: train NLL {running/max(seen,1):.4f}  "
              f"val NLL {val_nll:.4f}  ({fmt_dur(dt)}, run eta {fmt_dur(run_eta)})",
              flush=True)

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "step": step,
                 "best_val": min(best_val, val_nll), "cfg": vars(cfg), "stats": stats}
        torch.save(state, os.path.join(cfg.ckpt_dir, "last.pt"))
        if val_nll < best_val:
            best_val = val_nll
            torch.save(state, os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"  new best val NLL {best_val:.4f} -> checkpoints/best.pt")

    print(f"done. best val NLL {best_val:.4f}")


if __name__ == "__main__":
    main()
