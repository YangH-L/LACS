"""Command line entry point.

    python -m lacs <command> -c configs/<file>.yaml [--set key=value ...]

Commands (in pipeline order):
    synth         generate the synthetic test dataset (dataset.type=synthetic does this on demand)
    prepare       raw data -> canonical tables + time split            (data/)
    encode        message embeddings (+PCA)                             (emb/)
    export-texts  per-user training-period texts for an external psych pipeline
    features      anchors, user features, positions                    (features/)
    g1            gate G1: ordinal structure of stance distance        (results/g1.json)
    train         E1 acceptance prediction, G2, H3                     (results/e1_*, g2.json, h3_*)
    graph         meta-path user graph up to the end of validation    (graph/)
    g3            gate G3: community sizes over k                      (results/g3.json)
    e2            E2 effectiveness + dynamics simulation (+E4 ablations, E5 case)
    e3            E3 efficiency (uses the `efficiency` config section)
    report        summary of all results                               (results/REPORT.md)
    diagnose      read-only diagnosis of a finished run, rules R0-R4   (diagnosis/)
    geo-prepare / geo-gates / geo-e2 / geo-all (= the three in order) / geo-e2-sweep (k sweep)
                  second application: meetups in location-based social networks
                  (Gowalla / Brightkite; see lacs/apps/geo.py)
    all           prepare .. report (without e3)
    build-cpp     compile the C++ engine
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import load_config, run_paths
from .utils import dump_json, setup_logging

STEPS = ["prepare", "encode", "features", "g1", "train", "graph", "g3", "e2", "report"]


def dispatch(cmd: str, cfg: dict, paths: dict):
    if cmd == "prepare":
        from .data import prepare
        return prepare.run(cfg)
    if cmd == "synth":
        from .data import synthetic
        return synthetic.generate(cfg["dataset"]["raw"], **cfg["dataset"].get("synthetic", {}))
    if cmd == "encode":
        from .features import encode
        return encode.run(cfg, paths)
    if cmd == "export-texts":
        import json

        import pandas as pd

        from .features import psych
        comments = pd.read_pickle(paths["data"] / "comments.pkl")
        meta = json.loads((paths["data"] / "meta.json").read_text())
        return psych.export_user_texts(comments, meta["t_train_end"], paths["feat"] / "user_texts.jsonl")
    if cmd == "features":
        from .features import build
        return build.run(cfg, paths)
    if cmd == "g1":
        from .experiments import g1
        return g1.run(cfg, paths)
    if cmd == "train":
        from .model import train
        return train.run(cfg, paths)
    if cmd == "graph":
        from .graph import build
        return build.run(cfg, paths)
    if cmd == "g3":
        from .experiments import e2
        return e2.run_g3(cfg, paths)
    if cmd == "e2":
        from .experiments import e2
        return e2.run(cfg, paths)
    if cmd == "e3":
        from .experiments import e3
        return e3.run(cfg, paths)
    if cmd == "report":
        from .experiments import report
        return report.run(cfg, paths)
    if cmd == "diagnose":
        from .experiments import diagnose
        return diagnose.run(cfg, paths)
    if cmd in ("geo-prepare", "geo-gates", "geo-e2", "geo-e2-sweep"):
        from .apps import geo
        return {"geo-prepare": geo.prepare, "geo-gates": geo.gates, "geo-e2": geo.e2,
                "geo-e2-sweep": geo.e2_sweep}[cmd](cfg, paths)
    if cmd == "build-cpp":
        from .search.cpp import ensure_built
        return str(ensure_built(force=True))
    raise SystemExit(f"unknown command {cmd}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lacs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=STEPS + ["all", "synth", "export-texts", "e3", "diagnose", "build-cpp",
                                                "geo-prepare", "geo-gates", "geo-e2", "geo-all",
                                                "geo-e2-sweep"])
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. --set search.k=4")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if args.command == "build-cpp":  # needs no configuration and writes nothing under runs/
        from .search.cpp import ensure_built
        print(f"[lacs] C++ tools in {ensure_built(force=True)}", flush=True)
        return 0
    cfg = load_config(args.config, args.set)
    paths = run_paths(cfg)
    setup_logging(args.log_level, str(paths["root"] / "lacs.log"))
    sub = (cfg.get("geo") or {}).get("results_subdir")
    if args.command == "diagnose":  # read-only on the run: keep its config_resolved.json untouched
        (paths["root"] / "diagnosis").mkdir(parents=True, exist_ok=True)
        dump_json(cfg, paths["root"] / "diagnosis" / "config_resolved.json")
    elif sub and args.command.startswith("geo-"):  # re-evaluation of an existing run: keep its record too
        (paths["results"] / sub).mkdir(parents=True, exist_ok=True)
        dump_json(cfg, paths["results"] / sub / "config_resolved.json")
    else:
        dump_json(cfg, paths["root"] / "config_resolved.json")
    if args.command == "all":
        cmds = STEPS
    elif args.command == "geo-all":
        cmds = ["geo-prepare", "geo-gates", "geo-e2"]
    else:
        cmds = [args.command]
    for c in cmds:
        t0 = time.time()
        dispatch(c, cfg, paths)
        print(f"[lacs] {c} finished in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
