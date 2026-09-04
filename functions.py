from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from candle_model import TrajectoryGenerator
from data_model import CandleGroup, CandleGroupList, GenerateModel, GroupLabel, ReturnCandle

from model import (
    CKPT_PATHS,
    DEFAULT_MODEL_VERSION,
    DEVICE,
    HORIZON,
    MIN_WINDOW,
    NUM_SAMPLES,
    SIGMA_SCALE,
    TEMPERATURE,
)

log = logging.getLogger(__name__)


class LoadedModel:
    """Thin wrapper around TrajectoryGenerator keeping the `.cfg` dict /
    `.device` interface the service (health endpoint, logging) expects."""

    def __init__(self, gen: TrajectoryGenerator, cfg: dict):
        self.gen = gen
        self.cfg = cfg
        self.device = gen.device


def resolve_ckpt_path(model_version: str | None) -> str:
    version = model_version or DEFAULT_MODEL_VERSION
    if version not in CKPT_PATHS:
        raise BatchError(
            f"model_version: {version!r} unknown, expected one of {sorted(CKPT_PATHS)}"
        )
    return CKPT_PATHS[version]


@functools.lru_cache(maxsize=len(CKPT_PATHS))
def load_model(ckpt_path: str = CKPT_PATHS[DEFAULT_MODEL_VERSION], device: str = DEVICE) -> LoadedModel:
    gen = TrajectoryGenerator(ckpt_path, device=device)
    cfg = dict(vars(gen.cfg))
    cfg.setdefault("architecture", "candle_gen_mdn")

    if int(cfg["horizon"]) != HORIZON:
        raise RuntimeError(
            f"checkpoint config mismatch: horizon={cfg['horizon']!r} "
            f"but service expects {HORIZON!r}"
        )

    n_params = sum(p.numel() for p in gen.model.parameters())
    log.info("loaded %s | architecture=%s | params=%d", ckpt_path, cfg.get("architecture"), n_params)
    return LoadedModel(gen, cfg)


class BatchError(ValueError):
    """Input that cannot be turned into a valid batch (caller's fault -> 422)."""


def candles_to_ohlc(candles) -> np.ndarray:
    """list[CandleModel] -> (n,4) float64 O,H,L,C."""
    if not candles:
        return np.zeros((0, 4), np.float64)
    return np.array([[c.o, c.h, c.l, c.c] for c in candles], dtype=np.float64)


def _validate_ohlc(ohlc: np.ndarray, what: str) -> None:
    if not np.isfinite(ohlc).all():
        raise BatchError(f"{what}: contains NaN/inf")
    if (ohlc <= 0).any():
        # the log-space candle features require strictly positive prices
        raise BatchError(f"{what}: prices must be strictly positive")


def build_inputs(req, cfg: dict) -> dict:
    """Turn the request into the raw arrays TrajectoryGenerator.generate takes:
    query OHLC plus score-sorted analog patterns/futures/scores."""
    q_pat = candles_to_ohlc(req.scanned_candles)
    if q_pat.shape[0] < MIN_WINDOW:
        raise BatchError(f"scanned_candles: need at least {MIN_WINDOW} candles")
    max_w = int(cfg["max_pattern_len"])
    if q_pat.shape[0] > max_w:
        q_pat = q_pat[-max_w:]
    _validate_ohlc(q_pat, "scanned_candles")

    # the model's rank embedding expects analogs score-sorted descending and
    # holds at most max_analogs slots — drop the least similar beyond that
    pats = sorted(req.similar_patterns or [],
                  key=lambda p: p.similarity_score, reverse=True)
    max_k = int(cfg["max_analogs"])
    pats = pats[:max_k]
    k_real = len(pats)

    patterns: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    scores: list[float] = []
    for p in pats:
        a_pat = candles_to_ohlc(p.pattern_candles)
        a_fut = candles_to_ohlc(p.reaction_candles)
        if a_pat.shape[0] < 2 or a_fut.shape[0] < 1:
            continue            # unusable analog -> skip
        try:
            _validate_ohlc(a_pat, "pattern_candles")
            _validate_ohlc(a_fut, "reaction_candles")
        except BatchError:
            continue            # a bad analog must not fail the whole request
        # generator truncates patterns to max_pattern_len / futures to horizon
        patterns.append(a_pat)
        futures.append(a_fut)
        scores.append(float(p.similarity_score))

    if not patterns:
        if k_real == 0:
            raise BatchError("similar_patterns: required — none were provided")
        raise BatchError(
            f"similar_patterns: {k_real} pattern(s) provided but all were unusable "
            "(bad OHLC or insufficient candles)"
        )

    return {"q_pat": q_pat, "q_win": q_pat.shape[0], "patterns": patterns,
            "futures": futures, "scores": scores, "k_real": k_real,
            "k_used": len(patterns)}


N_SCEN_MIN = 6
N_SCEN_MAX = 15
N_SCEN_BASE = 2
N_SCEN_SLOPE = 3


@dataclass
class Scenario:
    path: np.ndarray        # (HORIZON,) close prices — the representative sample
    idx: int                # its index in the sampled fan
    cluster: int            # percentile bucket index (bear=0 .. bull=N-1)
    size: int               # members in that bucket
    share: float            # bucket population fraction
    end_ret: float          # terminal return vs anchor
    rmse: float | None = None   # vs realised future, when available
    dtw: float | None = None    # shape distance vs realised future (DTW path only)


DTW_TOP_FRACTION = 0.8   # keep the best-scoring 80% of candidates, by DTW rank
DTW_MAX_SCENARIOS = 20   # hard cap regardless of how many pass the 80% cut
DTW_WINDOW = 5


def side_quotas(n_total: int, n_up_pool: int, n_down_pool: int) -> tuple[int, int]:
    """Split `n_total` picks between up/down in proportion to how many
    candidates actually exist on each side of the sampled fan (n_up_pool /
    n_down_pool) — so a 70/30 fan reports roughly 70/30 scenarios instead of
    a ratio driven purely by which side happens to score better downstream.
    Each side gets at least 1 pick as long as it has any candidates, so a
    lopsided ratio (99/1) doesn't zero out the minority side entirely."""
    n_pool = n_up_pool + n_down_pool
    if n_pool == 0 or n_total <= 0:
        return 0, 0
    n_total = min(n_total, n_pool)

    both_sides_present = n_up_pool > 0 and n_down_pool > 0
    if both_sides_present and n_total == 1:
        # Only one slot to give but both sides have candidates: award it to
        # whichever side has the larger share instead of hard-coding a winner.
        return (1, 0) if n_up_pool >= n_down_pool else (0, 1)

    up_n = int(round(n_total * n_up_pool / n_pool))
    up_n = max(1, up_n) if n_up_pool > 0 else 0
    up_n = min(up_n, n_total - (1 if n_down_pool > 0 else 0), n_up_pool)
    down_n = min(n_total - up_n, n_down_pool)
    return up_n, down_n


def dtw_top_picks(dists: np.ndarray, down_idx: np.ndarray, up_idx: np.ndarray,
                  top_fraction: float = DTW_TOP_FRACTION,
                  max_total: int = DTW_MAX_SCENARIOS) -> tuple[np.ndarray, np.ndarray]:
    """Keep the best-scoring `top_fraction` of the pool by GLOBAL DTW rank
    (capped at `max_total`), then top up either side that fell below its
    `side_quotas` floor — so a strong minority-side match is never displaced
    by a weaker majority-side one just because the majority pool is bigger,
    while both sides still get at least one pick when they have candidates."""
    n = len(dists)
    n_keep = min(max_total, math.ceil(n * top_fraction)) if n else 0
    if n_keep == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    floor_up, floor_down = side_quotas(n_keep, len(up_idx), len(down_idx))

    global_order = np.argsort(dists)[:n_keep]
    picked = set(global_order.tolist())

    down_set = set(down_idx.tolist())
    up_set = set(up_idx.tolist())
    down_pick = {i for i in picked if i in down_set}
    up_pick = {i for i in picked if i in up_set}

    # top up whichever side fell short of its guaranteed floor, pulling in
    # that side's next-best-DTW candidates not already picked; drop the
    # globally-picked set's worst entries on the OTHER side to hold the
    # total at n_keep instead of letting a top-up inflate it.
    def _top_up(side_pick: set, side_idx: np.ndarray, floor: int):
        if len(side_pick) >= floor:
            return side_pick, 0
        remaining = [i for i in side_idx[np.argsort(dists[side_idx])] if i not in side_pick]
        need = min(floor - len(side_pick), len(remaining))
        return side_pick | set(remaining[:need]), need

    down_pick, down_added = _top_up(down_pick, down_idx, floor_down)
    up_pick, up_added = _top_up(up_pick, up_idx, floor_up)

    def _trim(side_pick: set, other_added: int):
        if other_added <= 0:
            return side_pick
        worst_first = sorted(side_pick, key=lambda i: -dists[i])
        drop = set(worst_first[:other_added])
        return side_pick - drop

    if up_added:
        down_pick = _trim(down_pick, up_added)
    if down_added:
        up_pick = _trim(up_pick, down_added)

    down_pick = np.array(sorted(down_pick, key=lambda i: dists[i]), dtype=int)
    up_pick = np.array(sorted(up_pick, key=lambda i: dists[i]), dtype=int)

    return down_pick, up_pick


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int = DTW_WINDOW) -> float:
    """DTW with a Sakoe-Chiba band (max warp = `window` bars), path-length
    normalized so values are comparable across series lengths. Identical to
    the reference implementation in tester.py."""
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def znorm(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def choose_scenarios_dtw(paths: np.ndarray, last_close: float,
                        actual_closes: np.ndarray) -> tuple[list["Scenario"], dict]:
    """Pick the closest-in-shape samples from the sampled fan, ranked by DTW
    distance of z-normalized close prices against the realised future (same
    metric as tester.py). No fixed per-side count: candidates are pooled and
    ranked by DTW score, the best-scoring DTW_TOP_FRACTION survive (capped at
    DTW_MAX_SCENARIOS total), and each side (up/down) is guaranteed at least
    one pick — so the result size scales with how many candidates are actually
    a good match instead of always returning the same count.

    actual_closes may be shorter than HORIZON (the reaction is still unfolding):
    each candidate path is truncated to the same length before comparison, so
    the match is judged only on the bars that have actually happened rather
    than letting DTW warp a short realised segment against the full fan."""
    paths = np.asarray(paths, np.float64)
    n = paths.shape[0]
    end_ret_all = paths[:, -1] / last_close - 1.0
    # bucket by the larger-magnitude excursion (max vs min close reached along
    # the path), not the terminal close: a path that spikes up and drifts back
    # down to close negative is a bull move that round-tripped, not a bear move.
    max_ret_all = paths.max(axis=1) / last_close - 1.0
    min_ret_all = paths.min(axis=1) / last_close - 1.0

    actual_closes = np.asarray(actual_closes, np.float64)
    n_actual = min(len(actual_closes), HORIZON)
    za = znorm(actual_closes[:n_actual])
    dists = np.array([dtw_distance(znorm(paths[i, :n_actual]), za) for i in range(n)])

    down_idx = np.where(max_ret_all < -min_ret_all)[0]
    up_idx = np.where(max_ret_all >= -min_ret_all)[0]

    down_pick, up_pick = dtw_top_picks(dists, down_idx, up_idx)

    picked = list(down_pick) + list(up_pick)

    scenarios = [
        Scenario(path=paths[idx], idx=int(idx), cluster=j, size=1, share=1.0 / n,
                 end_ret=float(paths[idx, -1] / last_close - 1.0),
                 dtw=float(dists[idx]))
        for j, idx in enumerate(picked)
    ]
    # best-fitting shape first: this path exists precisely because a realised
    # future is available to rank against, so fit is the meaningful order.
    # `cluster` is reassigned post-sort to stay a stable 0..N-1 position index.
    scenarios.sort(key=lambda s: s.dtw)
    for j, s in enumerate(scenarios):
        s.cluster = j

    meta = {
        "n_scenarios": len(scenarios),
        "selection": "dtw_nearest_shape",
        "sorted_by": "dtw",
        "top_fraction": DTW_TOP_FRACTION,
        "max_scenarios": DTW_MAX_SCENARIOS,
        "n_up_picked": len(up_pick),
        "n_down_picked": len(down_pick),
        "n_down_available": int(len(down_idx)),
        "n_up_available": int(len(up_idx)),
        "terminal_std": float(end_ret_all.std()),
    }
    return scenarios, meta


def _percentile_reps(end_ret_side: np.ndarray, idx_side: np.ndarray, n_pick: int,
                     paths: np.ndarray, last_close: float, n_total: int) -> list["Scenario"]:
    """Pick `n_pick` representative paths from one side (up or down) by terminal-
    return percentile, same mechanics as the original single-pool version but
    scoped to that side's own subset of paths."""
    if n_pick <= 0 or len(idx_side) == 0:
        return []
    n_pick = min(n_pick, len(idx_side))
    pcts = np.linspace(10, 90, n_pick) if n_pick > 1 else np.array([50.0])

    rep_targets = np.percentile(end_ret_side, pcts)
    rep_local = [int(np.argmin(np.abs(end_ret_side - t))) for t in rep_targets]

    edges_pct = np.concatenate([[0.0], (pcts[:-1] + pcts[1:]) / 2, [100.0]])
    edges = np.percentile(end_ret_side, edges_pct)
    sizes = [int(np.sum((end_ret_side >= edges[j]) & (end_ret_side <= edges[j + 1])))
             for j in range(n_pick)]

    return [
        Scenario(path=paths[idx_side[local]], idx=int(idx_side[local]), cluster=0,
                 size=sizes[j], share=sizes[j] / n_total,
                 end_ret=float(paths[idx_side[local], -1] / last_close - 1.0))
        for j, local in enumerate(rep_local)
    ]


def choose_scenarios(paths: np.ndarray, last_close: float,
                     actual_closes: np.ndarray | None = None,
                     seed: int = 0, q_pat: np.ndarray | None = None) -> tuple[list["Scenario"], dict]:
    paths = np.asarray(paths, np.float64)
    n = paths.shape[0]
    end_ret_all = paths[:, -1] / last_close - 1.0
    # split by the larger-magnitude excursion (max vs min close reached along
    # the path), consistent with the DTW path and the final display label —
    # a path that spikes up and drifts back down to close negative is a bull
    # move that round-tripped, not a bear move.
    max_ret_all = paths.max(axis=1) / last_close - 1.0
    min_ret_all = paths.min(axis=1) / last_close - 1.0
    down_idx = np.where(max_ret_all < -min_ret_all)[0]
    up_idx = np.where(max_ret_all >= -min_ret_all)[0]

    if q_pat is not None and len(q_pat) > 1:
        qvol = float(np.diff(np.log(np.asarray(q_pat, np.float64)[:, 3])).std())
    else:
        qvol = 0.0
    rw_band = qvol * math.sqrt(HORIZON)
    ratio = float(end_ret_all.std()) / (rw_band + 1e-9)

    n_scen = int(np.clip(round(N_SCEN_BASE + N_SCEN_SLOPE * ratio), N_SCEN_MIN, N_SCEN_MAX))
    n_scen = min(n_scen, n)     # can't have more scenarios than sampled paths

    # split n_scen by the fan's actual up:down ratio (e.g. 70/30), not by
    # whichever side the raw percentile grid happens to favor.
    up_n, down_n = side_quotas(n_scen, len(up_idx), len(down_idx))

    up_scenarios = _percentile_reps(end_ret_all[up_idx], up_idx, up_n, paths, last_close, n)
    down_scenarios = _percentile_reps(end_ret_all[down_idx], down_idx, down_n, paths, last_close, n)

    scenarios = down_scenarios + up_scenarios
    scenarios.sort(key=lambda s: s.end_ret)
    for j, s in enumerate(scenarios):
        s.cluster = j

    meta = {
        "n_scenarios": len(scenarios),
        "selection": "terminal_quantile",
        "sorted_by": "end_ret",
        "n_up_picked": len(up_scenarios),
        "n_down_picked": len(down_scenarios),
        "n_up_available": int(len(up_idx)),
        "n_down_available": int(len(down_idx)),
        "terminal_std": float(end_ret_all.std()),
        "query_vol": qvol,
        "rw_band": rw_band,
        "width_ratio": ratio,
    }
    return scenarios, meta


def score_against_actual(scenarios: list[Scenario], actual_closes: np.ndarray,
                         last_close: float) -> int | None:
    n = min(len(actual_closes), HORIZON)
    if n == 0:
        return None
    for s in scenarios:
        d = s.path[:n] - actual_closes[:n]
        s.rmse = float(np.sqrt((d ** 2).mean()) / last_close)
    return min(range(len(scenarios)), key=lambda i: scenarios[i].rmse)


TEMPERATURE_MIN = 0.5
TEMPERATURE_MAX = 1.5


def generate_detailed(req: GenerateModel, loaded: LoadedModel | None = None,
                      temperature: float = TEMPERATURE,
                      seed: int | None = 0, model_version: str | None = None) -> dict:
    loaded = loaded or load_model(resolve_ckpt_path(model_version))

    # num_samples is fixed at NUM_SAMPLES — not caller-configurable, so a bad
    # or hostile value (0, negative, huge) can never reach the sampler.
    num_samples = NUM_SAMPLES
    # clamp rather than reject: an out-of-range temperature is a caller mistake,
    # not a malformed request, and the fan is still meaningful at the boundary.
    # np.clip passes NaN through, so fall back to the default for non-finite input.
    temperature = float(temperature)
    if not math.isfinite(temperature):
        temperature = TEMPERATURE
    temperature = float(np.clip(temperature, TEMPERATURE_MIN, TEMPERATURE_MAX))

    t0 = time.perf_counter()
    prep = build_inputs(req, loaded.cfg)
    q_pat = prep["q_pat"]
    t1 = time.perf_counter()

    # q_anchor is the caller's anchor price; fall back to the query's last close
    last_close = float(req.q_anchor) if req.q_anchor else float(q_pat[-1, 3])
    if not np.isfinite(last_close) or last_close <= 0:
        last_close = float(q_pat[-1, 3])

    # The generator continues from the query's own last close, and its candle
    # features are pure price ratios — rescaling the whole query window is
    # exact and simply moves the anchor to the caller's price.
    q_anchored = q_pat * (last_close / float(q_pat[-1, 3]))

    if seed is not None:
        torch.manual_seed(seed)

    trajs = loaded.gen.generate(
        q_anchored, prep["patterns"], prep["futures"], prep["scores"],
        n_traj=num_samples, temperature=temperature, sigma_scale=SIGMA_SCALE,
    )                                                    # (N, HORIZON, 4) OHLC
    paths = np.ascontiguousarray(trajs[:, :, 3])         # (N, HORIZON) closes
    t2 = time.perf_counter()

    actual_closes = None
    if req.forward_candle_data:
        actual = candles_to_ohlc(req.forward_candle_data)
        if actual.shape[0]:
            # Unvalidated forward prices poison the whole ranking: one NaN close
            # makes every DTW distance NaN, argsort degenerates to index order,
            # and the "closest in shape" scenarios come back arbitrary with a 200.
            _validate_ohlc(actual, "forward_candle_data")
            actual_closes = actual[:, 3]

    if actual_closes is not None:
        scenarios, meta = choose_scenarios_dtw(paths, last_close, actual_closes=actual_closes)
    else:
        scenarios, meta = choose_scenarios(paths, last_close,
                                           actual_closes=actual_closes,
                                           seed=0 if seed is None else seed,
                                           q_pat=q_pat)
    t3 = time.perf_counter()

    closest = None
    if actual_closes is not None:
        closest = score_against_actual(scenarios, actual_closes, last_close)

    groups: list[CandleGroup] = []
    for s in scenarios:
        # the generator emits full OHLC bars — return the picked trajectory's
        # candles directly instead of re-synthesizing wicks around the closes
        ohlc = trajs[s.idx]
        candles = [ReturnCandle(o=float(o), h=float(h), l=float(l), c=float(c))
                   for o, h, l, c in ohlc]
        # label by the larger-magnitude excursion (max vs min close reached along
        # the path), not the terminal close: a path that spikes +10% and drifts
        # back to close -1% is a bull move that round-tripped, not a bear move.
        max_ret = float(s.path.max()) / last_close - 1.0
        min_ret = float(s.path.min()) / last_close - 1.0
        label = GroupLabel.UPWARD if max_ret >= -min_ret else GroupLabel.DOWNWARD
        groups.append(CandleGroup(label=label, candles=candles))
    t4 = time.perf_counter()

    log.info(
        "generate_detailed timing: build_inputs=%.3fs generate=%.3fs "
        "choose_scenarios=%.3fs (dtw=%s) groups=%.3fs total=%.3fs "
        "k_used=%d num_samples=%d",
        t1 - t0, t2 - t1, t3 - t2, actual_closes is not None, t4 - t3,
        t4 - t0, prep["k_used"], num_samples,
    )

    meta["scenarios"] = [
        {"cluster": s.cluster, "size": s.size, "share": s.share,
         "idx": s.idx, "end_ret": s.end_ret,
         "label": "UPWARD" if s.end_ret > 0 else "DOWNWARD",
         "rmse": s.rmse, "dtw": s.dtw}
        for s in scenarios
    ]
    meta["closest_scenario"] = closest
    meta["anchor_close"] = last_close
    meta["n_analogs_used"] = prep["k_used"]
    meta["n_analogs_provided"] = prep["k_real"]
    meta["model_version"] = model_version or DEFAULT_MODEL_VERSION
    meta["num_samples"] = num_samples
    meta["temperature"] = temperature
    meta["sigma_scale"] = SIGMA_SCALE

    return {"groups": CandleGroupList(groups=groups), "meta": meta}
