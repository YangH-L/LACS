"""E1: interaction stress test of returned groups under opinion dynamics.

A *variant* fixes the dynamics model, the interaction graph and the weights:
  model   hk | sjt | dw | follow
          (eq. (1), eq. (2), Deffuant-Weisbuch gossip, and "follow-one": each
          member moves a fraction gamma towards ONE accepted neighbour drawn at
          random each step -- convex, but as concentrated as weights can be)
  graph   G | K             (social ties inside the group, or everybody hears everybody)
  weights equal | dirichlet (raw affinities ~ Gamma(alpha), normalised over the
                             accepted set; drawn once per run)
Robustness knobs (assumptions of Theorems 1-2 violated on purpose):
  open         open group: members also average with the outside neighbours they
               accept (bounded confidence on the whole network, outsiders held still)
  lambda_ext   pull towards the fixed mean of all outside neighbours, whatever
               their distance (stubborn outsiders; harsher than open)
  theta_noise  true tolerance = estimated * exp(nu * Z): the group was selected
               with the estimated tolerances, the dynamics use the true ones
Stopping: a run ends after T_max synchronous steps (rounds for DW) or as soon
as no member moves more than move_tol_rel * initial diameter in one step.
Events after the stop are not observed, so break rates are lower bounds.
Tolerances: HK and DW use theta_acc = theta. SJT uses theta_rej = theta and
theta_acc = acc_ratio * theta, so an LCC computed with theta is an LCC^rej and
Theorem 2 applies.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from . import ffi
from .graphs import complete_csr, group_stats, induced_csr, means_from_lists, outside_neighbors, reverse_index

DEFAULTS = {
    "T_max": 300,
    "rounds": 100,
    "move_tol_rel": 1e-6,
    "rel_eps": 1e-9,
    "diam_every": 10,
    "cons_rel": 1e-3,
    "cluster_rel": 1e-3,
}


@dataclass
class Variant:
    name: str
    model: str = "hk"
    graph: str = "G"
    weights: str = "equal"
    runs: int = 1
    beta: float = 0.05
    acc_ratio: float = 1.0
    lambda_ext: float = 0.0
    open: bool = False
    theta_noise: float = 0.0
    mu: float = 0.3
    gamma: float = 0.5
    dirichlet_alpha: float = 1.0
    max_n: int = 5000
    T_max: Optional[int] = None
    rounds: Optional[int] = None
    params: Dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict, defaults: dict) -> "Variant":
        d = dict(d)
        known = {f for f in Variant.__dataclass_fields__ if f != "params"}
        kw = {k: d.pop(k) for k in list(d) if k in known}
        v = Variant(**kw)
        v.params = {**DEFAULTS, **(defaults or {}), **d}
        if v.T_max is not None:
            v.params["T_max"] = v.T_max
        if v.rounds is not None:
            v.params["rounds"] = v.rounds
        if v.model not in ("hk", "sjt", "dw", "follow"):
            raise ValueError(f"unknown model {v.model}")
        if v.graph not in ("G", "K"):
            raise ValueError(f"unknown graph {v.graph}")
        if v.weights not in ("equal", "dirichlet"):
            raise ValueError(f"unknown weights {v.weights}")
        if v.model == "dw" and (v.lambda_ext > 0 or v.weights != "equal"):
            raise ValueError("dw supports neither external pull nor weights")
        if v.model in ("hk", "follow") and v.acc_ratio != 1.0:
            raise ValueError(f"{v.model} uses theta_acc = theta (acc_ratio must be 1)")
        if v.model == "follow" and v.weights != "equal":
            raise ValueError("follow draws its own weights; use weights: equal")
        if v.open and v.model not in ("hk", "follow"):
            raise ValueError("open groups are implemented for hk and follow")
        if v.open and v.lambda_ext > 0:
            raise ValueError("use either open or lambda_ext")
        return v


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


class GroupContext:
    """Everything the kernels need for one group, computed once."""

    def __init__(self, gptr, gidx, X, theta, members, pos_buf=None, need_ext=False, ext_max_neighbors=200, seed=0,
                 need_open=False):
        self.members = np.asarray(members, dtype=np.int64)
        self.n = len(self.members)
        self.Xh = np.asarray(X[self.members], dtype=np.float64)
        self.th = np.asarray(theta[self.members], dtype=np.float64)
        self.stats = group_stats(self.Xh, self.th)
        self.ptr, self.idx = induced_csr(gptr, gidx, self.members, pos_buf)
        self.stats["n_edges"] = int(len(self.idx) // 2)
        self.stats["min_degree"] = int(np.diff(self.ptr).min()) if self.n else 0
        self._rev = None
        self._K = None
        self._Krev = None
        self.ext = None
        self.open = None
        if need_ext or need_open:
            rng = np.random.default_rng(seed)
            optr, oX, cnt = outside_neighbors(gptr, gidx, self.members, X, pos_buf, ext_max_neighbors, rng)
            self.stats["share_with_outside_nbrs"] = float(np.mean(cnt > 0)) if self.n else 0.0
            self.stats["mean_outside_nbrs"] = float(np.mean(cnt)) if self.n else 0.0
            if need_ext:
                self.ext = means_from_lists(optr, oX)
            if need_open:
                self.open = (optr, oX)

    def graph(self, which: str):
        if which == "G":
            return self.ptr, self.idx
        if self._K is None:
            self._K = complete_csr(self.n)
        return self._K

    def rev(self, which: str):
        if which == "G":
            if self._rev is None:
                self._rev = reverse_index(self.ptr, self.idx)
            return self._rev
        if self._Krev is None:
            p, i = self.graph("K")
            self._Krev = reverse_index(p, i)
        return self._Krev


def _diam_every(base: int, n: int, T: int) -> int:
    """Diameter checks cost O(n^2 m) on G[H]; space them out on large groups."""
    scale = max(1, int(np.ceil((n / 1000.0) ** 2)))
    return int(max(1, min(base * scale, max(T, 1))))


def run_variant(ctx: GroupContext, var: Variant, seed: int) -> List[dict]:
    P = var.params
    rng = np.random.default_rng(seed)
    ptr, idx = ctx.graph(var.graph)
    out = []
    for r in range(max(1, int(var.runs))):
        th = ctx.th
        if var.theta_noise > 0:
            th = th * np.exp(var.theta_noise * rng.standard_normal(ctx.n))
        if var.model == "sjt":
            th_acc, th_rej = var.acc_ratio * th, th
        else:
            th_acc, th_rej = th, th
        t0 = time.perf_counter()
        if var.model == "dw":
            de = _diam_every(max(1, int(P["diam_every"]) // 2), ctx.n, int(P["rounds"]))
            res = ffi.sim_dw(
                ctx.Xh, th_acc, th_rej, ptr, idx, ctx.rev(var.graph), mu=var.mu, rounds=int(P["rounds"]),
                seed=int(rng.integers(0, 2**62)), move_tol_rel=P["move_tol_rel"], rel_eps=P["rel_eps"],
                diam_every=de, cons_rel=P["cons_rel"], cluster_rel=P["cluster_rel"],
            )
        else:
            W_self = W_edge = None
            if var.weights == "dirichlet":
                W_self = rng.gamma(var.dirichlet_alpha, size=ctx.n)
                W_edge = rng.gamma(var.dirichlet_alpha, size=len(idx))
            ext_mean = ext_mask = None
            if var.lambda_ext > 0:
                if ctx.ext is None:
                    raise RuntimeError("external pull requested but GroupContext built without need_ext")
                ext_mean, ext_mask = ctx.ext
            open_ptr = open_X = open_W = None
            if var.open:
                if ctx.open is None:
                    raise RuntimeError("open group requested but GroupContext built without need_open")
                open_ptr, open_X = ctx.open
                if var.weights == "dirichlet":
                    open_W = rng.gamma(var.dirichlet_alpha, size=len(open_X))
            de = _diam_every(int(P["diam_every"]), ctx.n, int(P["T_max"]))
            res = ffi.sim_sync(
                ctx.Xh, th_acc, th_rej, ptr, idx, W_self=W_self, W_edge=W_edge, ext_mean=ext_mean,
                ext_mask=ext_mask, lambda_ext=var.lambda_ext, open_ptr=open_ptr, open_X=open_X, open_W=open_W,
                model=var.model, beta=var.beta, T_max=int(P["T_max"]), move_tol_rel=P["move_tol_rel"],
                rel_eps=P["rel_eps"], diam_every=de, cons_rel=P["cons_rel"], cluster_rel=P["cluster_rel"],
                gamma=var.gamma, seed=int(rng.integers(0, 2**62)),
            )
        res["sim_seconds"] = time.perf_counter() - t0
        res["n"] = ctx.n
        out.append(res)
    return out


def aggregate(runs: List[dict]) -> dict:
    a = {k: np.array([r[k] for r in runs], dtype=float) for k in runs[0]}
    npairs = np.maximum(a["npairs"], 1.0)
    tc = a["t_consensus"][a["t_consensus"] >= 0]
    fb = a["first_break"][a["first_break"] >= 0]
    return {
        "runs": len(runs),
        "p_broken": float(np.mean(a["broken"] > 0)),
        "p_any_refusal": float(np.mean((a["broken"] + a["t0_refused"]) > 0)),
        "p_t0_refusal": float(np.mean(a["t0_refused"] > 0)),
        "p_new_rejection": float(np.mean(a["new_rejected"] > 0)),
        "p_any_rejection": float(np.mean(a["ever_rejected"] > 0)),
        "mean_broken_frac": float(np.mean(a["broken"] / npairs)),
        "mean_t0_refused_frac": float(np.mean(a["t0_refused"] / npairs)),
        "median_first_break": float(np.median(fb)) if len(fb) else -1.0,
        "mean_clusters": float(np.mean(a["n_clusters"])),
        "mean_largest_frac": float(np.mean(a["largest_frac"])),
        "p_consensus": float(np.mean(a["n_clusters"] == 1)),
        "median_t_consensus": float(np.median(tc)) if len(tc) else -1.0,
        "p_converged": float(np.mean(a["converged"] > 0)),
        "mean_diam_growth": float(np.mean(a["diam_max"] / np.maximum(a["diam0"], 1e-300))),
        "mean_steps": float(np.mean(a["steps"])),
        "mean_open_acc_share": float(np.mean(a["open_acc0"] / np.maximum(a["n"], 1.0))),
        "sim_seconds": float(np.sum(a["sim_seconds"])),
    }


def variant_table(variants: List[Variant]) -> List[dict]:
    rows = []
    for v in variants:
        d = asdict(v)
        d.update(d.pop("params"))
        rows.append(d)
    return rows
