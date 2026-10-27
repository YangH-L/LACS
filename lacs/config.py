"""Configuration: YAML file deep-merged over DEFAULTS, plus --set overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "run",
    "seed": 0,
    "out_dir": "runs/run",
    "dataset": {
        "type": "debagreement",          # debagreement | cmv | synthetic | generic
        "raw": "data/raw/debagreement",
        "extra_raw": None,               # unlabelled interactions (same schema, no label)
        "columns": {},
        "label_map": None,
        "exclude_authors": ["[deleted]", "[removed]", "AutoModerator", "DeltaBot", "nan"],
        "exclude_author_regex": "(?i)(^|_)bot$",
        "synthetic": {"n_users": 500, "n_threads": 260, "seed": 0},
    },
    "split": {"ratios": [0.7, 0.1, 0.2]},
    "encoder": {
        "backend": "sbert",              # sbert | tfidf
        "model": "sentence-transformers/all-mpnet-base-v2",
        "batch_size": 256,
        "device": "cuda",
        "max_seq_length": 256,
        "tfidf_dim": 256,
    },
    "pca_dim": 128,
    "anchors": {
        "receiver_repr": "anchor",       # anchor | op_post (CMV only)
        "source_repr": "anchor",         # anchor | message
        "recency_lambda": 3.0,
        "max_hist": 50,
        "shrink_kappa": 0.5,
    },
    "features": {"psych_csv": None, "ego": True, "vader": "auto"},
    "model": {
        "proj_dim": [8, 16],
        "l2_W": [0.3, 3.0],
        "l2_w": [1.0],
        "l2_b": [1.0, 10.0],
        "max_iter": 400,
        "variants": ["global", "feat", "resid", "full"],
        "band": False,
        "n_jobs": 2,
        "cold_start_max_train": 2,
    },
    "baselines": ["prior", "dist_logit", "mf", "balance", "reply_text"],
    "gates": {"g1_bins": 10, "alpha": 0.05, "g3_min_median": 5, "g3_max_r_over_kcore": 0.8, "g3_k": [2, 3, 4, 5],
              "g3_queries": 200},
    "graph": {"metapaths": [{"name": "reply"}, {"name": "cothread", "max_thread_users": 50}]},
    "search": {
        "k": 3,
        "n_queries": 200,
        "time_limit": 120.0,
        "max_exact_size": 3000,
        "n_proc": 4,                     # parallel engine processes (queries are split across them)
        "query_pool": "test_active",     # kcore | test_active
        "latitude_model": "full",        # which SJT-Lat variant supplies positions and latitudes
        "methods": ["lcc_c_exact", "lcc_c_greedy", "lcc_c_ballcore_f", "lcc_c_ballcore_r", "lcc_r_exact",
                    "pairwise_c", "kcore", "kr_global", "edgeacc", "cluster_q", "random_bfs"],
        "cluster_k": 20,
    },
    "simulation": {"enabled": True, "max_steps": 500, "max_steps_external": 200, "max_steps_sjt": 300,
                   "tol": 1e-7, "mu": 0.3, "beta": 0.3, "external": True, "max_size": 3000,
                   "max_external_size": 2000, "n_jobs": 4},
    "efficiency": {},
}


def deep_merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path: str | Path | None, sets: list[str] | None = None) -> dict:
    user = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = apply_overrides(cfg, sets)
    cfg["_config_path"] = str(path) if path else None
    return cfg


def run_paths(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["out_dir"])
    p = {
        "root": root,
        "data": root / "data",
        "emb": root / "emb",
        "feat": root / "features",
        "models": root / "models",
        "graph": root / "graph",
        "results": root / "results",
        "figs": root / "figures",
        "work": root / "work",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p
