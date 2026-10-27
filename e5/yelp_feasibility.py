#!/usr/bin/env python3
"""E5 (W3) feasibility check on the Yelp Open Dataset -- LACS, 2026-10-25.

Run before the E5 test plan is written. It compares no methods and computes no group outcome;
it only answers whether the planned application can work, and which pre-declared variant to use:

  1. Download: sha256 of the zip, zip / inner-tar listing, the five JSON files, line counts,
     review date range and reviews per year.
  2. Friendship graph: friend lists that are non-empty, friend ids that resolve to users of the
     dataset, reciprocity; per time split, the graph among "taste-ready" users restricted to
     friends with the same home metro, its k-cores and the eligible queries.
  3. Sparsity of the outcome: share of friend pairs (and friend-of-friend pairs, random
     same-metro pairs) that review at least one common business in the test period, under the
     two pre-declared time splits; the same for the validation period and within 30 days.
  4. Does condition (iii) bind?  For each pre-declared position / tolerance definition
     (S1, S2, S3 and the geographic variant G): the distribution of d(u,v) / min(theta_u, theta_v)
     over friend pairs and random same-metro pairs and, with the v0.5 engine on a sample of
     queries (exact search with a short time limit), the share of queries with an LCC, its size
     relative to the connected k-core, and the share proven optimal; greedy only as a side note.
The decision rule of the plan (docs/PLANS.md (lacs_e5_prereg_draft_20261025); fixed before any number was seen) is applied at the
end; its outcome is written to feasibility.json and printed.

Only aggregates are written to --out: no user, business or review ids and no per-user rows.
Parsed arrays (integer indices only) are cached in --work and must not leave the server.

  python e5/yelp_feasibility.py --zip data/yelp/Yelp-JSON.zip \
      --data-dir data/yelp/json --work runs/e5_feas/work --out runs/e5_feas/out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import tarfile
import time
import traceback
import zipfile
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I (no script directory on sys.path)
    sys.path.insert(0, str(ROOT))

try:  # faster parsing when available; lines orjson rejects (NaN literals, lone surrogates) go to json
    import orjson as _orjson

    def _loads(b):
        try:
            return _orjson.loads(b)
        except _orjson.JSONDecodeError:
            return json.loads(b)
except Exception:  # pragma: no cover - depends on the environment
    _orjson = None

    def _loads(b):
        return json.loads(b)


MAX_BAD_LINES = 1000  # malformed JSON lines skipped (and counted) per file before giving up
FLUSH_EVERY = 500_000  # reviews per conversion chunk (tests lower it)


class _Lines:
    """Iterate over the parsed objects of a JSON-lines file, skipping (and counting) malformed
    lines up to MAX_BAD_LINES."""

    def __init__(self, path: Path, stats: Counter):
        self.path, self.stats = path, stats

    def __iter__(self):
        with open(self.path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    o = _loads(line)
                except ValueError:
                    self.stats["malformed_lines"] += 1
                    if self.stats["malformed_lines"] > MAX_BAD_LINES:
                        raise RuntimeError(f"{self.path.name}: more than {MAX_BAD_LINES} malformed lines")
                    continue
                if not isinstance(o, dict):
                    self.stats["malformed_lines"] += 1
                    if self.stats["malformed_lines"] > MAX_BAD_LINES:
                        raise RuntimeError(f"{self.path.name}: more than {MAX_BAD_LINES} malformed lines")
                    continue
                yield o

VERSION = "e5-feas-0.1.1"
EXPECTED_ZIP_SHA256 = "47dd6e4d279ac9d8734ddc30bfb3d78e571b9df4bb95923d7acf9a6ef3d8a4ab"
KINDS = ("business", "checkin", "review", "tip", "user")
EARTH_R = 6371.0088
BAD_DAY = np.iinfo(np.int64).min
LOG = logging.getLogger("e5feas")
assert array("i").itemsize == 4


# ------------------------------------------------------------------ utils --

def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def qdict(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    out = {"n": int(len(x)), "mean": float(x.mean())}
    for q, v in zip(qs, np.quantile(x, qs)):
        out[f"q{int(round(q * 100))}"] = float(v)
    return out


def rate(hits: np.ndarray) -> dict:
    hits = np.asarray(hits, dtype=bool)
    n = int(len(hits))
    if n == 0:
        return {"n": 0, "rate": None}
    p = float(hits.mean())
    return {"n": n, "hits": int(hits.sum()), "rate": p, "se": float(np.sqrt(p * (1 - p) / n))}


def group_quantile(g: np.ndarray, x: np.ndarray, q: float, n_groups: int) -> np.ndarray:
    """Per-group quantile with linear interpolation (same as numpy's default method);
    nan for empty groups."""
    g = np.asarray(g, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    o = np.lexsort((x, g))
    gs, xs = g[o], x[o]
    cnt = np.bincount(gs, minlength=n_groups)
    start = np.cumsum(cnt) - cnt
    res = np.full(n_groups, np.nan)
    ok = cnt > 0
    pos = start[ok] + q * (cnt[ok] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    res[ok] = xs[lo] + (pos - lo) * (xs[hi] - xs[lo])
    return res


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_xyz(lat, lon) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    return np.column_stack([EARTH_R * np.cos(phi) * np.cos(lam), EARTH_R * np.cos(phi) * np.sin(lam),
                            EARTH_R * np.sin(phi)])


def core_numbers(A: sp.csr_matrix) -> np.ndarray:
    """k-core number of every vertex (Batagelj-Zaversnik bin sort, O(m))."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)
    indptr = A.indptr.tolist()
    indices = A.indices.tolist()
    deg = [indptr[i + 1] - indptr[i] for i in range(n)]
    md = max(deg)
    bins = [0] * (md + 1)
    for d in deg:
        bins[d] += 1
    start = 0
    for d in range(md + 1):
        bins[d], start = start, start + bins[d]
    pos = [0] * n
    vert = [0] * n
    for v in range(n):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        bins[deg[v]] += 1
    for d in range(md, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        dv = deg[v]
        for p in range(indptr[v], indptr[v + 1]):
            u = indices[p]
            du = deg[u]
            if du > dv:
                pu = pos[u]
                pw = bins[du]
                w = vert[pw]
                if u != w:
                    pos[u], pos[w] = pw, pu
                    vert[pu], vert[pw] = w, u
                bins[du] += 1
                deg[u] = du - 1
    return np.asarray(deg, dtype=np.int64)


def edges_to_csr(n: int, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    A = sp.csr_matrix((np.ones(2 * len(u), dtype=np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1
    A.sort_indices()
    return A


def upper_pairs(A: sp.csr_matrix):
    up = sp.triu(A, k=1).tocoo()
    return up.row.astype(np.int64), up.col.astype(np.int64)


# ------------------------------------------------------------- extraction --

def kind_of(name: str):
    base = os.path.basename(name).lower()
    if not base.endswith(".json"):
        return None
    for k in KINDS:
        if base == f"{k}.json" or base.endswith(f"_{k}.json"):
            return k
    return None


def _junk(name: str) -> bool:
    return "__MACOSX" in name or os.path.basename(name).startswith("._")


def _copy_hash(src, dst: Path, expect_bytes: int | None = None) -> dict:
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    nbytes = lines = 0
    with open(part, "wb") as f:
        while True:
            b = src.read(1 << 24)
            if not b:
                break
            h.update(b)
            lines += b.count(b"\n")
            nbytes += len(b)
            f.write(b)
    if expect_bytes is not None and nbytes != expect_bytes:
        raise RuntimeError(f"{dst.name}: wrote {nbytes} bytes, archive says {expect_bytes}")
    os.replace(part, dst)
    return {"bytes": nbytes, "sha256": h.hexdigest(), "newlines": lines}


def extract(zip_path: Path, data_dir: Path, expected: str | None) -> dict:
    """Verify the zip, then stream the (compressed) tar inside it and write the five JSON files
    and the PDFs under data_dir (basenames only, '.part' until complete). Skipped when a manifest
    from the same zip exists and the files have the recorded sizes."""
    inv = {"zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size}
    t0 = time.time()
    inv["zip_sha256"] = sha256_file(zip_path)
    inv["zip_sha256_expected"] = expected
    inv["zip_sha256_ok"] = expected is None or inv["zip_sha256"] == expected
    LOG.info("zip sha256 %s (%s, %.0f s)", inv["zip_sha256"], "OK" if inv["zip_sha256_ok"] else "MISMATCH",
             time.time() - t0)
    if not inv["zip_sha256_ok"]:
        raise RuntimeError(f"zip sha256 {inv['zip_sha256']} != expected {expected}: stop, re-copy the file")
    data_dir.mkdir(parents=True, exist_ok=True)
    man_path = data_dir / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("zip_sha256") == inv["zip_sha256"] and man.get("files") and all(
                (data_dir / f["file"]).exists() and (data_dir / f["file"]).stat().st_size == f["bytes"]
                for f in man["files"]):
            LOG.info("extraction already done for this zip (manifest matches): reusing %s", data_dir)
            inv.update({k: v for k, v in man.items() if k != "zip_sha256"})
            inv["reused_extraction"] = True
            return inv
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        inv["zip_entries"] = [{"name": z.filename, "bytes": z.file_size, "compressed_bytes": z.compress_size}
                              for z in infos]
        tars = [z for z in infos if not _junk(z.filename) and z.filename.lower().endswith((".tar", ".tar.gz", ".tgz"))]
        if len(tars) != 1:
            raise RuntimeError(f"expected exactly one tar in the zip, found {[z.filename for z in tars]}")
        docs_dir = data_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        docs = []
        for z in infos:
            if not _junk(z.filename) and z.filename.lower().endswith(".pdf"):
                with zf.open(z) as src:
                    r = _copy_hash(src, docs_dir / os.path.basename(z.filename), z.file_size)
                docs.append({"file": "docs/" + os.path.basename(z.filename), **r})
        files, members = [], []
        t0 = time.time()
        with zf.open(tars[0]) as fz:
            with tarfile.open(fileobj=fz, mode="r|*") as tf:
                for m in tf:
                    members.append({"name": m.name, "bytes": int(m.size),
                                    "type": "file" if m.isfile() else ("dir" if m.isdir() else "other")})
                    if not m.isfile():
                        continue
                    base = os.path.basename(m.name)
                    if _junk(base):
                        continue
                    k = kind_of(base)
                    if k is None and not base.lower().endswith(".pdf"):
                        continue
                    dst = data_dir / base if k else docs_dir / base
                    r = _copy_hash(tf.extractfile(m), dst, int(m.size))
                    if k:
                        if any(f["kind"] == k for f in files):
                            raise RuntimeError(f"two '{k}' JSON files in the tar")
                        files.append({"kind": k, "file": base, **r})
                    else:
                        docs.append({"file": "docs/" + base, **r})
                    LOG.info("extracted %s (%.2f GB, %d lines)", base, r["bytes"] / 1e9, r["newlines"])
            while fz.read(1 << 24):  # drain to the end of the zip member so its CRC is checked
                pass
        LOG.info("tar streamed in %.0f s", time.time() - t0)
    inv.update({"tar_member": tars[0].filename, "tar_members": members, "files": files, "docs": docs,
                "missing_kinds": [k for k in KINDS if not any(f["kind"] == k for f in files)]})
    man_path.write_text(json.dumps({k: inv[k] for k in ("zip_sha256", "zip_entries", "tar_member", "tar_members",
                                                        "files", "docs", "missing_kinds")}, indent=1))
    return inv


def inventory_dir(json_dir: Path) -> dict:
    files = []
    for p in sorted(json_dir.iterdir()):
        k = kind_of(p.name)
        if k:
            with open(p, "rb") as f:
                r = _copy_hash_count(f)
            files.append({"kind": k, "file": p.name, **r})
    return {"json_dir": str(json_dir), "files": files,
            "missing_kinds": [k for k in KINDS if not any(f["kind"] == k for f in files)]}


def _copy_hash_count(f) -> dict:
    h = hashlib.sha256()
    n = lines = 0
    while True:
        b = f.read(1 << 24)
        if not b:
            break
        h.update(b)
        n += len(b)
        lines += b.count(b"\n")
    return {"bytes": n, "sha256": h.hexdigest(), "newlines": lines}


# ---------------------------------------------------------------- parsing --

def _cats(c) -> list:
    if isinstance(c, str):
        return [s.strip() for s in c.split(",") if s.strip()]
    if isinstance(c, list):
        return [str(s).strip() for s in c if str(s).strip()]
    return []


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_business(path: Path):
    bid, lat, lon, state, city, cats = {}, array("d"), array("d"), [], [], []
    stats = Counter()
    if True:
        for o in _Lines(path, stats):
            b = o.get("business_id")
            if b is None:
                stats["no_id"] += 1
                continue
            if b in bid:
                stats["duplicate_id"] += 1
                continue
            bid[b] = len(lat)
            lat.append(_f(o.get("latitude")))
            lon.append(_f(o.get("longitude")))
            state.append(str(o.get("state") or "").strip())
            city.append(str(o.get("city") or "").strip())
            c = _cats(o.get("categories"))
            cats.append(c)
            stats["no_categories"] += int(not c)
    lat = np.frombuffer(lat, dtype=np.float64).copy()
    lon = np.frombuffer(lon, dtype=np.float64).copy()
    stats["bad_coordinates"] = int((~np.isfinite(lat) | ~np.isfinite(lon)).sum())
    return bid, lat, lon, state, city, cats, dict(stats)


def _friend_list(fr) -> list:
    if isinstance(fr, str):
        s = fr.strip()
        if not s or s == "None":
            return []
        lst = s.split(",")
    elif isinstance(fr, list):
        lst = fr
    else:
        return []
    out = []
    for x in lst:
        x = str(x).strip()
        if x and x != "None":
            out.append(x)
    return out


def load_users(path: Path):
    """Two passes: ids first, then friend lists mapped to indices (unknown ids counted, dropped)."""
    uid = {}
    stats = Counter()
    t0 = time.time()
    if True:
        for o in _Lines(path, stats):
            u = o.get("user_id")
            if u is None:
                stats["no_id"] += 1
            elif u in uid:
                stats["duplicate_id"] += 1
            else:
                uid[u] = len(uid)
    LOG.info("user.json pass 1: %d users (%.0f s)", len(uid), time.time() - t0)
    n = len(uid)
    done = bytearray(n)
    src, dst = array("i"), array("i")
    deg_listed = np.zeros(n, np.int32)
    get = uid.get
    t0 = time.time()
    pass2 = Counter()
    if True:
        for o in _Lines(path, pass2):
            u = uid.get(o.get("user_id"))
            if u is None or done[u]:
                continue
            done[u] = 1
            lst = _friend_list(o.get("friends"))
            if lst:
                stats["users_with_nonempty_friends"] += 1
            stats["friend_refs"] += len(lst)
            deg_listed[u] = len(lst)
            got = 0
            for x in lst:
                v = get(x)
                if v is None:
                    stats["friend_refs_unknown"] += 1
                elif v == u:
                    stats["friend_refs_self"] += 1
                else:
                    src.append(u)
                    dst.append(v)
                    got += 1
            if got:
                stats["users_with_resolved_friends"] += 1
    stats["friend_refs_resolved"] = len(src)
    LOG.info("user.json pass 2: %d friend references, %d resolved (%.0f s)", stats["friend_refs"], len(src),
             time.time() - t0)
    s = np.frombuffer(src, dtype=np.int32).astype(np.int64)
    d = np.frombuffer(dst, dtype=np.int32).astype(np.int64)
    return uid, s, d, deg_listed, dict(stats)


def _to_days(strs: list) -> np.ndarray:
    try:
        return np.array(strs, dtype="datetime64[D]").astype(np.int64)
    except ValueError:
        out = np.empty(len(strs), np.int64)
        for i, s in enumerate(strs):
            try:
                out[i] = np.datetime64(s, "D").astype(np.int64)
            except ValueError:
                out[i] = BAD_DAY
        return out


def load_reviews(path: Path, uid: dict, bid: dict):
    parts_u, parts_b, parts_d = [], [], []
    cu, cb, cd = array("i"), array("i"), []
    stats = Counter()

    def flush():
        if not cd:
            return
        d = _to_days(cd)
        ok = d != BAD_DAY
        stats["bad_date"] += int((~ok).sum())
        parts_u.append(np.frombuffer(cu, dtype=np.int32)[ok].copy())
        parts_b.append(np.frombuffer(cb, dtype=np.int32)[ok].copy())
        parts_d.append(d[ok].astype(np.int32))
        del cu[:]
        del cb[:]
        cd.clear()

    t0 = time.time()
    if True:
        for i, o in enumerate(_Lines(path, stats)):
            u = uid.get(o.get("user_id"))
            b = bid.get(o.get("business_id"))
            if u is None:
                stats["unknown_user"] += 1
                continue
            if b is None:
                stats["unknown_business"] += 1
                continue
            ds = str(o.get("date") or "")[:10]
            cu.append(u)
            cb.append(b)
            cd.append(ds)
            if len(cd) >= FLUSH_EVERY:
                flush()
            if i and i % 2_000_000 == 0:
                LOG.info("review.json: %d lines (%.0f s)", i, time.time() - t0)
    flush()
    U = np.concatenate(parts_u) if parts_u else np.zeros(0, np.int32)
    B = np.concatenate(parts_b) if parts_b else np.zeros(0, np.int32)
    D = np.concatenate(parts_d) if parts_d else np.zeros(0, np.int32)
    LOG.info("review.json: %d reviews kept (%.0f s)", len(U), time.time() - t0)
    return U.astype(np.int64), B.astype(np.int64), D.astype(np.int64), dict(stats)


_REUSED = {}


def parsed_reused(work: Path) -> bool:
    return _REUSED.get(str(work), False)


def _parse_code_hash() -> str:
    """sha256 of the source of every function that shapes the parsed arrays: a cache written by
    other parsing code is never reused."""
    import inspect

    h = hashlib.sha256()
    for f in (_Lines, _cats, _f, load_business, _friend_list, load_users, _to_days, load_reviews):
        h.update(inspect.getsource(f).encode("utf-8"))
    return h.hexdigest()


def load_or_parse(fpath: dict, inv: dict, work: Path):
    """Parse the three JSON files that matter, or load the integer-indexed arrays cached in `work`
    by an earlier run on exactly the same files with the same parsing code (both recorded inside
    the cache file, written atomically). Cache problems never fail the run: an unreadable or stale
    cache is ignored, a failed write is logged. The cache holds user-level data and never leaves
    the server."""
    key = {"files": {k: next(f["sha256"] for f in inv["files"] if f["kind"] == k) for k in ("business", "user", "review")},
           "code": _parse_code_hash()}
    cache = work / "parsed.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                meta = json.loads(str(z["meta_json"]))
                if meta.get("key") == key:
                    arr = {k: z[k] for k in ("blat", "blon", "deg_listed")}
                    arr.update({k: z[k].astype(np.int64) for k in ("fs", "fd", "U", "B", "D")})  # int32 on disk
                    LOG.info("reusing parsed arrays from %s", cache)
                    _REUSED[str(work)] = True
                    return (arr["blat"], arr["blon"], meta["bstate"], meta["bcity"], meta["bcats"], meta["bstats"],
                            arr["fs"], arr["fd"], arr["deg_listed"], meta["ustats"], arr["U"], arr["B"], arr["D"],
                            meta["rstats"], meta["NU"], meta["NB"])
                LOG.info("parsed-array cache in %s is for other files or other parsing code: parsing again", work)
        except Exception as e:
            LOG.warning("ignoring unreadable parsed-array cache %s (%s)", cache, e)
    t0 = time.time()
    bid, blat, blon, bstate, bcity, bcats, bstats = load_business(fpath["business"])
    LOG.info("business.json: %d businesses (%.0f s)", len(bid), time.time() - t0)
    uid, fs, fd, deg_listed, ustats = load_users(fpath["user"])
    U, B, D, rstats = load_reviews(fpath["review"], uid, bid)
    NU, NB = len(uid), len(bid)
    del uid, bid
    _REUSED[str(work)] = False
    try:
        work.mkdir(parents=True, exist_ok=True)
        meta = {"key": key, "bstate": bstate, "bcity": bcity, "bcats": bcats, "bstats": bstats, "ustats": ustats,
                "rstats": rstats, "NU": NU, "NB": NB}
        tmp = work / "parsed.tmp.npz"
        np.savez(tmp, blat=blat, blon=blon, fs=fs.astype(np.int32), fd=fd.astype(np.int32), deg_listed=deg_listed,
                 U=U.astype(np.int32), B=B.astype(np.int32), D=D.astype(np.int32), meta_json=np.array(json.dumps(meta)))
        os.replace(tmp, cache)
    except Exception as e:
        LOG.warning("could not write the parsed-array cache (%s); continuing without it", e)
    return (blat, blon, bstate, bcity, bcats, bstats, fs, fd, deg_listed, ustats, U, B, D, rstats, NU, NB)


def day_str(d) -> str:
    return str(np.datetime64(int(d), "D"))


def day_of(s: str) -> int:
    return int(np.datetime64(s, "D").astype(np.int64))


# ----------------------------------------------------------------- metros --

def find_metros(lat, lon, state, city, cell: float, min_business: int):
    """Connected components of occupied cell x cell degree grid cells (8-neighbourhood);
    components with >= min_business businesses are metros (0 = largest), the rest -1."""
    nb = len(lat)
    metro = np.full(nb, -1, np.int64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return metro, []
    ci = np.floor(lat[ok] / cell).astype(np.int64)
    cj = np.floor(lon[ok] / cell).astype(np.int64)
    off = 10_000
    key = (ci + off) * 100_000 + (cj + off)
    cells, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    idx = {int(c): i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, c in enumerate(cells.tolist()):
        a, b = divmod(c, 100_000)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                j = idx.get((a + da) * 100_000 + (b + db))
                if j is not None:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    comp_of_cell = np.array([find(i) for i in range(len(cells))], dtype=np.int64)
    comp = comp_of_cell[inv]
    labels, counts = np.unique(comp, return_counts=True)
    big = labels[counts >= min_business][np.argsort(-counts[counts >= min_business], kind="stable")]
    mapping = {int(c): m for m, c in enumerate(big.tolist())}
    ids = np.flatnonzero(ok)
    metro[ids] = np.array([mapping.get(int(c), -1) for c in comp], dtype=np.int64)
    info = []
    for m in range(len(big)):
        sel = np.flatnonzero(metro == m)
        st = Counter(state[i] for i in sel)
        ct = Counter(city[i] for i in sel)
        info.append({"metro": m, "businesses": int(len(sel)),
                     "state_top": st.most_common(3), "city_top": ct.most_common(3),
                     "lat_mean": round(float(lat[sel].mean()), 2), "lon_mean": round(float(lon[sel].mean()), 2)})
    info.append({"metro": -1, "businesses": int((metro == -1).sum()), "note": "outside every metro or no coordinates"})
    return metro, info


def plurality(u: np.ndarray, m: np.ndarray, n_users: int, n_metros: int):
    """Most frequent metro of each user's reviews (-1 = outside every metro counts as a value of
    its own; ties -> lower metro id, and any metro before -1). Returns (home, share of the user's
    reviews there); users without reviews get home -2."""
    home = np.full(n_users, -2, np.int64)
    share = np.zeros(n_users)
    if len(u) == 0:
        return home, share
    code = u * (n_metros + 1) + (m + 1)
    cu, cnt = np.unique(code, return_counts=True)
    uu, mm = cu // (n_metros + 1), cu % (n_metros + 1) - 1
    tie = np.where(mm >= 0, mm, n_metros)  # on ties a metro beats 'outside every metro'
    o = np.lexsort((tie, -cnt, uu))
    uu, mm, cnt = uu[o], mm[o], cnt[o]
    first = np.r_[True, uu[1:] != uu[:-1]]
    tot = np.bincount(u, minlength=n_users)
    home[uu[first]] = mm[first]
    share[uu[first]] = cnt[first] / tot[uu[first]]
    return home, share


# ------------------------------------------------------------ base rates --

class ItemSets:
    """Per-user set of reviewed businesses in one period (and the days of those reviews)."""

    def __init__(self, users: np.ndarray, items: np.ndarray, days: np.ndarray):
        o = np.lexsort((days, items, users))
        users, items, days = users[o], items[o], days[o]
        self.sets: dict[int, frozenset] = {}
        self.days: dict[int, dict] = {}
        if len(users) == 0:
            return
        b = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
        e = np.r_[b[1:], len(users)]
        for s_, e_ in zip(b.tolist(), e.tolist()):
            u = int(users[s_])
            it = items[s_:e_].tolist()
            dy = days[s_:e_].tolist()
            dd = {}
            for i_, d_ in zip(it, dy):
                dd.setdefault(i_, []).append(d_)
            self.sets[u] = frozenset(dd)
            self.days[u] = dd

    def co(self, pu, pv) -> np.ndarray:
        S = self.sets
        empty = frozenset()
        return np.fromiter((not S.get(a, empty).isdisjoint(S.get(b, empty)) for a, b in zip(pu.tolist(), pv.tolist())),
                           dtype=bool, count=len(pu))

    def common(self, a: int, b: int) -> frozenset:
        return self.sets.get(a, frozenset()) & self.sets.get(b, frozenset())

    def within_days(self, a: int, b: int, w: int) -> bool:
        da, db = self.days.get(a, {}), self.days.get(b, {})
        for it in self.common(a, b):
            if min(abs(x - y) for x in da[it] for y in db[it]) <= w:
                return True
        return False


def sample_random_pairs(groups: list, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n random pairs of distinct members of the same group, groups weighted by their number of
    pairs (i.e. uniform over all same-group pairs)."""
    sizes = np.array([len(g) for g in groups], dtype=np.float64)
    w = sizes * (sizes - 1)
    if n <= 0 or w.sum() <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    gi = rng.choice(len(groups), size=n, p=w / w.sum())
    pu = np.empty(n, np.int64)
    pv = np.empty(n, np.int64)
    for g in np.unique(gi):
        sel = np.flatnonzero(gi == g)
        arr = groups[g]
        a = rng.integers(0, len(arr), len(sel))
        b = (a + rng.integers(1, len(arr), len(sel))) % len(arr)
        pu[sel], pv[sel] = arr[a], arr[b]
    return pu, pv


def sample_fof_pairs(A: sp.csr_matrix, ok: np.ndarray, n: int, rng, max_tries: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Pairs at distance exactly 2 in A, both endpoints with ok = True: u uniform among ok vertices
    with a neighbour, w uniform in N(u), v uniform in N(w); rejected if v = u, v in N(u) or not ok."""
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    starts = np.flatnonzero(ok & (deg > 0))
    out_u, out_v = [], []
    if len(starts) == 0 or n <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    got = 0
    for _ in range(max_tries):
        m = 2 * (n - got)
        u = starts[rng.integers(0, len(starts), m)]
        w = indices[indptr[u] + (rng.random(m) * deg[u]).astype(np.int64)]
        dw = deg[w]
        v = indices[indptr[w] + (rng.random(m) * dw).astype(np.int64)]
        keep = (v != u) & ok[v]
        u, v = u[keep], v[keep]
        # drop direct friends (sorted CSR rows: binary search)
        isnb = np.zeros(len(u), bool)
        for i in range(len(u)):
            row = indices[indptr[u[i]]:indptr[u[i] + 1]]
            j = np.searchsorted(row, v[i])
            isnb[i] = j < len(row) and row[j] == v[i]
        u, v = u[~isnb], v[~isnb]
        out_u.append(u)
        out_v.append(v)
        got += len(u)
        if got >= n:
            break
    u = np.concatenate(out_u)[:n]
    v = np.concatenate(out_v)[:n]
    return u, v


# ------------------------------------------------------------ taste specs --

def business_vectors(cats: list, used: np.ndarray, min_business: int):
    """Vocabulary = categories carried by >= min_business of the businesses in `used`; each
    business -> L2-normalised multi-hot row over the vocabulary (zero row if none)."""
    cnt = Counter()
    for b in used.tolist():
        cnt.update(set(cats[b]))
    vocab = sorted([c for c, k in cnt.items() if k >= min_business], key=lambda c: (-cnt[c], c))
    col = {c: j for j, c in enumerate(vocab)}
    rows, cols = [], []
    for b, cs in enumerate(cats):
        js = sorted({col[c] for c in cs if c in col})
        rows += [b] * len(js)
        cols += js
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    nnz_row = np.bincount(rows, minlength=len(cats)).astype(np.float64)
    vals = 1.0 / np.sqrt(nnz_row[rows]) if len(rows) else np.zeros(0)
    Bv = sp.csr_matrix((vals, (rows, cols)), shape=(len(cats), len(vocab)))
    Bv.sort_indices()
    return Bv, vocab, cnt


def row_dots(Bv: sp.csr_matrix, rb: np.ndarray, C: np.ndarray, ru: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """<Bv[rb[i]], C[ru[i]]> for every i (Bv sparse, C dense)."""
    out = np.zeros(len(rb))
    ip, ix, dv = Bv.indptr, Bv.indices, Bv.data
    for s in range(0, len(rb), chunk):
        b = rb[s:s + chunk]
        u = ru[s:s + chunk]
        cnt = ip[b + 1] - ip[b]
        tot = int(cnt.sum())
        if tot == 0:
            continue
        rows = np.repeat(np.arange(len(b)), cnt)
        base = np.repeat(ip[b], cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        out[s:s + chunk] = np.bincount(rows, weights=dv[base] * C[u[rows], ix[base]], minlength=len(b))
    return out


def pair_dist(X: np.ndarray, pu: np.ndarray, pv: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(pu))
    for s in range(0, len(pu), chunk):
        a = X[pu[s:s + chunk]].astype(np.float64)
        b = X[pv[s:s + chunk]].astype(np.float64)
        out[s:s + chunk] = np.sqrt(((a - b) ** 2).sum(1))
    return out


def ratio_stats(d: np.ndarray, th: np.ndarray, pu: np.ndarray, pv: np.ndarray) -> dict:
    tmin = np.minimum(th[pu], th[pv])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(tmin > 0, d / tmin, np.where(d > 0, np.inf, 0.0))
    out = {"n": int(len(r))}
    if len(r) == 0:
        return out
    qs = np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9], method="inverted_cdf")  # data points; inf stays inf
    out.update({f"q{p}": (float(v) if np.isfinite(v) else None) for p, v in zip((10, 25, 50, 75, 90), qs)})
    out["share_inf"] = float(np.isinf(r).mean())
    out["share_mutually_within"] = float((r <= 1.0).mean())  # d <= min(theta_u, theta_v)
    out["share_le_half"] = float((r <= 0.5).mean())
    out["share_ge_2"] = float((r >= 2.0).mean())
    return out


# ----------------------------------------------------------------- engine --

# Every engine call: no member lists, and no post-hoc exact diameter of the result (O(s^2 d), not
# included in time_ms and never used here; at several hundred dimensions it can cost more than the
# search itself). These and the time-limit settings are recorded in feasibility.json.
ENGINE_COMMON = ["--no-members", "--diam-cap", "0"]
UB_TIME = lambda limit: max(5.0, 0.1 * limit)  # noqa: E731  engine default for the bound pass


def run_engine_budget(cs, queries, algos, k: int, time_limit: float, n_proc: int, tag: str, budget_s: float,
                      extra=()) -> tuple[list, dict]:
    """Like CppSearch.run (same command line) but with a wall-clock budget for the whole batch: on
    the budget the engine processes still running are killed and the queries they finished are
    kept (the engine writes one line per query per algorithm and flushes after each query). A
    process that exits non-zero on its own is an error, whatever happens to the others."""
    import subprocess

    queries = [int(q) for q in queries]
    n_proc = max(1, min(int(n_proc), len(queries)))
    procs = []
    for ci in range(n_proc):
        chunk = queries[ci::n_proc]
        qpath = cs.workdir / f"queries_{tag}_{ci}.txt"
        qpath.write_text("\n".join(map(str, chunk)) + "\n")
        out_path = cs.workdir / f"out_{tag}_{ci}.jsonl"
        err_path = cs.workdir / f"err_{tag}_{ci}.txt"
        for p_ in (out_path, err_path):
            if p_.exists():
                p_.unlink()
        cmd = [str(cs.exe), "--graph", str(cs.graph_path), "--pos", str(cs.pos_path), "--theta",
               str(cs.theta_paths["acc"]), "--queries", str(qpath), "--algo", ",".join(algos), "--k", str(k),
               "--time-limit", str(time_limit), "--out", str(out_path), *ENGINE_COMMON, *map(str, extra)]
        ef = open(err_path, "w")
        procs.append({"p": subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ef), "out": out_path,
                      "err": err_path, "ef": ef, "killed": False})
    t0 = time.time()
    while any(d["p"].poll() is None for d in procs):
        if time.time() - t0 > budget_s:
            for d in procs:
                if d["p"].poll() is None:
                    d["p"].kill()
                    d["killed"] = True
            break
        time.sleep(0.2)
    rows, errors, rcs = [], [], []
    for d in procs:
        d["p"].wait()
        d["ef"].close()
        rcs.append({"rc": d["p"].returncode, "killed": d["killed"]})
        if d["p"].returncode != 0 and not d["killed"]:
            errors.append(f"process rc={d['p'].returncode}: " + d["err"].read_text()[-500:])
        if d["out"].exists():
            for line in d["out"].read_text().splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:  # line cut by the kill
                        pass
    meta = {"killed_at_budget": any(d["killed"] for d in procs), "seconds": round(time.time() - t0, 1),
            "budget_s": round(float(budget_s), 1), "processes": rcs}
    if errors:
        raise RuntimeError("lacs_search failed: " + " | ".join(errors))
    return rows, meta


def summarize_exact(qq, rows: list) -> dict:
    """Exact LatSweep + connected k-core rows -> per-specification statistics.

    A query is 'proven' when the engine says optimal, or when its upper bound equals the size it
    returned (certified by the bound pass). 'found' = a non-empty LCC was returned (any status);
    'proven empty' = proven with size 0; everything else is 'unknown' (not finished, or no
    incumbent and no proof). Shares: lower = found, upper = not proven empty. Relative size for
    criterion (b) over the 'not proven empty' queries: proven size, otherwise the upper bound (1 if
    there is none), divided by the size of q's connected k-core, capped at 1."""
    kc = {r["q"]: r for r in rows if r["algo"] == "kcore"}
    xr = {r["q"]: r for r in rows if r["algo"] == "exact"}
    st = [xr[int(q)]["status"] if int(q) in xr else "budget" for q in qq]
    xs = np.array([xr[int(q)]["size"] if int(q) in xr else 0 for q in qq], float)
    ub = np.array([xr[int(q)].get("ub", -1) if int(q) in xr else -1 for q in qq], float)
    ks = np.array([kc[int(q)]["size"] if int(q) in kc else np.nan for q in qq], float)
    proven = np.array([s == "optimal" for s in st]) | ((ub >= 0) & (ub == xs))
    found = xs > 0
    proven_empty = proven & ~found
    unknown = ~found & ~proven_empty
    open_ = ~proven_empty
    num = np.where(proven, xs, np.where(ub >= 0, ub, np.nan))
    rel = num / np.where(ks > 0, ks, np.nan)
    rel = np.where(np.isfinite(rel), np.minimum(rel, 1.0), 1.0)
    n = max(len(qq), 1)
    return {"status": dict(Counter(st)), "queries": int(len(qq)),
            "share_proven": float(proven.sum() / n),
            "share_certified_by_bound": float((proven & np.isin(np.array(st), ["timeout", "too_large"])).sum() / n),
            "share_found": float(found.sum() / n), "share_not_proven_empty": float(open_.sum() / n),
            "share_unknown": float(unknown.sum() / n),
            "size_found": qdict(xs[found]), "size_proven_nonempty": qdict(xs[proven & found]),
            "rel_size_open": qdict(rel[open_]),
            "median_rel_size_open": float(np.median(rel[open_])) if open_.any() else None,
            "kcore_size": qdict(ks),
            "time_ms": qdict([xr[int(q)]["time_ms"] for q in qq if int(q) in xr])}, (proven & found, xs)


def engine_check(A, specs: dict, eligible: np.ndarray, args, workdir: Path, rng) -> dict:
    """For every specification, on the same sample of eligible queries (structure only):
      * exact LatSweep with a short time limit, and the connected k-core of q: is there an LCC,
        how large is it relative to the k-core, is it proven (summarize_exact);
      * greedy (auxiliary, on a sub-sample): how often it finds an LCC where exact found one.
    One specification failing is recorded and does not discard the others."""
    from lacs.search.cpp import CppSearch  # v0.5 engine of this package

    k = 3
    qq = np.sort(rng.choice(eligible, size=min(int(args.engine_queries), len(eligible)), replace=False)) \
        if len(eligible) else np.zeros(0, np.int64)
    gq = np.sort(rng.choice(qq, size=min(int(args.greedy_queries), len(qq)), replace=False)) if len(qq) else qq
    n_proc = max(1, int(args.n_proc))
    per_proc = int(np.ceil(len(qq) / n_proc)) if len(qq) else 0
    need = per_proc * (float(args.exact_time) + UB_TIME(float(args.exact_time)) + 10.0)
    budget_x = max(float(args.budget_exact), 1.25 * need)
    out = {"k": k, "sampled_queries": int(len(qq)), "greedy_queries": int(len(gq)), "n_proc": n_proc,
           "engine_settings": {"common": ENGINE_COMMON, "exact_time_limit_s": float(args.exact_time),
                               "greedy_extra": ["--max-greedy-size", str(int(args.max_greedy_size))],
                               "budget_exact_s": round(budget_x, 1), "budget_greedy_s": float(args.budget_greedy),
                               "engine_defaults_not_passed": {"ub_time_s": UB_TIME(float(args.exact_time)),
                                                              "max_exact_size": 3000, "max_bound_size": 20000,
                                                              "max_greedy_size_in_exact_seeding": 20000}}}
    if len(qq) == 0:
        return out
    graph_bin, pos_bins = None, {}
    ours = core_numbers(A)
    for name, (pos, th) in specs.items():
        t0 = time.time()
        try:
            wd = workdir / f"engine_{name}"
            pos = np.ascontiguousarray(pos, dtype=np.float32)
            cs = CppSearch(A, pos, {"acc": th}, wd, graph_bin=graph_bin, pos_bin=pos_bins.get(id(pos)))
            graph_bin = cs.graph_path
            pos_bins[id(pos)] = cs.pos_path
            if "core_check" not in out:
                ce = cs.core_numbers().astype(np.int64)
                out["core_check"] = bool(np.array_equal(ce, ours))
                if not out["core_check"]:
                    raise RuntimeError("engine core numbers differ from the Python ones: graph file mismatch?")
            res = {"dims": int(pos.shape[1])}
            rows, meta = run_engine_budget(cs, qq, ["kcore", "exact"], k, float(args.exact_time), n_proc, "xk",
                                           budget_x)
            res["exact"], (pf, xs) = summarize_exact(qq, rows)
            res["exact"]["batch"] = meta
            rows_g, meta_g = run_engine_budget(cs, gq, ["greedy"], k, 60.0, n_proc, "gr", float(args.budget_greedy),
                                               ["--max-greedy-size", str(int(args.max_greedy_size))])
            gr = {r["q"]: r for r in rows_g}
            pos_in_qq = {int(q): i for i, q in enumerate(qq)}
            fq = [int(q) for q in gq if pf[pos_in_qq[int(q)]]]
            res["greedy"] = {"status": dict(Counter(gr[int(q)]["status"] if int(q) in gr else "budget" for q in gq)),
                             "batch": meta_g, "queries": int(len(gq)),
                             "recall_on_proven_nonempty": (float(np.mean([int(q) in gr and gr[q]["size"] > 0
                                                                          for q in fq])) if fq else None),
                             "size_ratio_to_exact": qdict([gr[q]["size"] / xs[pos_in_qq[q]] for q in fq
                                                           if q in gr and gr[q]["size"] > 0])}
            res["seconds"] = round(time.time() - t0, 1)
            out[name] = res
            e = res["exact"]
            LOG.info("engine %s: proven %.2f, LCC found %.2f (upper %.2f), median size/k-core %s; greedy recall %s "
                     "(%.0f s)", name, e["share_proven"], e["share_found"], e["share_not_proven_empty"],
                     e["median_rel_size_open"], res["greedy"]["recall_on_proven_nonempty"], res["seconds"])
        except Exception as ex:  # recorded; the decision rule then reports 'incomplete' where it matters
            out[name] = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-3000:],
                         "seconds": round(time.time() - t0, 1)}
            LOG.error("engine %s failed: %s", name, ex)
            if "core_check" in out and not out["core_check"]:
                break
    return out


# --------------------------------------------------------------- analysis --

def split_periods(D: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    if kind == "r4":  # round 4: global 70/10/20 split of all reviews by time
        ts = np.sort(D)
        t1 = int(ts[int(np.floor(0.7 * (len(ts) - 1)))])
        t2 = int(ts[int(np.floor(0.8 * (len(ts) - 1)))])
        per = np.where(D <= t1, 0, np.where(D <= t2, 1, 2))
        return per, {"train_end": day_str(t1), "val_end": day_str(t2), "test_end": day_str(ts[-1]),
                     "rule": "70/10/20 quantiles of all review dates (round 4)"}
    if kind == "precovid":
        t1, t2, t3 = day_of("2017-12-31"), day_of("2018-12-31"), day_of("2019-12-31")
        per = np.where(D <= t1, 0, np.where(D <= t2, 1, np.where(D <= t3, 2, 3)))
        return per, {"train_end": "2017-12-31", "val_end": "2018-12-31", "test_end": "2019-12-31",
                     "rule": "calendar: train <= 2017, validation 2018, test 2019, later reviews unused"}
    raise ValueError(kind)


def analyse_split(name: str, ctx: dict, args, rng) -> tuple[dict, dict]:
    """Everything for one time split. Returns (json-able results, objects kept for the engine)."""
    U, B, D = ctx["U"], ctx["B"], ctx["D"]
    NU, NB, M = ctx["NU"], ctx["NB"], ctx["n_metros"]
    per, bounds = split_periods(D, name)
    out = {"boundaries": bounds,
           "reviews_by_period": {p: int((per == i).sum()) for i, p in enumerate(("train", "val", "test", "unused"))}}
    tr = per == 0
    n_train = np.bincount(U[tr], minlength=NU)
    n_val = np.bincount(U[per == 1], minlength=NU)
    n_test = np.bincount(U[per == 2], minlength=NU)
    home, hshare = plurality(U[tr], ctx["bmetro"][B[tr]], NU, M)
    active = n_train >= args.min_train
    act_home = active & (home >= 0)
    out["users"] = {"train_ge_5": int((n_train >= 5).sum()), f"active_train_ge_{args.min_train}": int(active.sum()),
                    "train_ge_20": int((n_train >= 20).sum()), "active_with_home_metro": int(act_home.sum()),
                    "active_home_outside_metros": int((active & (home == -1)).sum()),
                    "home_share_of_train_reviews": qdict(hshare[act_home]),
                    "train_reviews_per_active": qdict(n_train[act_home])}

    # taste vocabulary from the businesses that active users reviewed in training
    used = np.unique(B[tr & act_home[U]])
    Bv, vocab, catcnt = business_vectors(ctx["cats"], used, args.vocab_min_business)
    has_vec = np.diff(Bv.indptr) > 0
    mt = tr & act_home[U] & has_vec[B]
    n_taste = np.bincount(U[mt], minlength=NU)
    ready = act_home & (n_taste >= args.min_taste)
    V = np.flatnonzero(ready)
    nV = len(V)
    vi = np.full(NU, -1, np.int64)
    vi[V] = np.arange(nV)
    out["taste_vocabulary"] = {"size": len(vocab), "min_businesses": int(args.vocab_min_business),
                               "top20": [[c, int(catcnt[c])] for c in vocab[:20]],
                               "share_train_reviews_with_vector": float(mt.sum() / max((tr & act_home[U]).sum(), 1)),
                               "categories_per_business": qdict(np.diff(Bv.indptr)[used])}
    out["users"]["taste_ready"] = int(nV)
    out["users"]["active_without_taste"] = int((act_home & ~ready).sum())

    # graph among taste-ready users: friends with the same home metro
    eu, ev = ctx["eu"], ctx["ev"]
    keep = (vi[eu] >= 0) & (vi[ev] >= 0) & (home[eu] == home[ev])
    A = edges_to_csr(nV, vi[eu[keep]], vi[ev[keep]])
    cross = (vi[eu] >= 0) & (vi[ev] >= 0) & (home[eu] != home[ev])
    core = core_numbers(A)
    deg = np.diff(A.indptr)
    g = {"vertices": int(nV), "edges": int(A.nnz // 2), "nonisolated": int((deg > 0).sum()),
         "dropped_cross_metro_edges": int(cross.sum()), "degree_nonisolated": qdict(deg[deg > 0]),
         "core_counts": {str(k): int((core >= k).sum()) for k in range(1, 11)}}
    c3 = np.flatnonzero(core >= 3)
    if len(c3):
        nc, lab = connected_components(A[c3][:, c3], directed=False)
        sizes = np.bincount(lab)
        g["core3_components"] = int(nc)
        g["core3_largest_component"] = int(sizes.max())
        g["core3_component_sizes_top5"] = sorted(sizes.tolist(), reverse=True)[:5]
    out["graph"] = g
    vhome = home[V]
    test_act = n_test[V] >= 1
    val_act = n_val[V] >= 1
    out["activity"] = {"test_active": int(test_act.sum()), "val_active": int(val_act.sum()),
                       "test_reviews_per_test_active": qdict(n_test[V][test_act]),
                       "val_reviews_per_val_active": qdict(n_val[V][val_act])}
    out["eligible_queries"] = {f"k{k}": int(((core >= k) & test_act).sum()) for k in (2, 3, 4, 5)}
    bym = []
    for m in range(M):
        sel = vhome == m
        if not sel.any():
            continue
        Am = A[sel][:, sel]
        bym.append({"metro": m, "taste_ready": int(sel.sum()), "edges": int(Am.nnz // 2),
                    "core3": int((core[sel] >= 3).sum()), "eligible_k3": int(((core >= 3) & test_act & sel).sum())})
    out["by_metro"] = bym

    # base rates of the outcome (no grouping by any position or tolerance)
    pu, pv = upper_pairs(A)
    br = {}
    every = np.ones(nV, bool)
    for pname, pidx, act in (("test", 2, test_act), ("val", 1, val_act)):
        m_ = (per == pidx) & (vi[U] >= 0)
        S = ItemSets(vi[U[m_]], B[m_], D[m_])
        co_all = S.co(pu, pv)  # every edge of the graph: the denominator used by the decision rule
        both = act[pu] & act[pv]
        one = act[pu] | act[pv]
        res = {"denominators": "friend_pairs = every edge of the graph (no activity condition); "
                               "*_both_active = both endpoints review in the period; *_one_active = at least one",
               "friend_pairs": rate(co_all), "friend_pairs_one_active": rate(co_all[one]),
               "friend_pairs_both_active": rate(co_all[both])}
        for label, ok in (("", every), ("_both_active", act)):
            groups = [np.flatnonzero((vhome == m) & ok) for m in range(M)]
            ru_, rv_ = sample_random_pairs(groups, int(args.pairs_random), rng)
            res["random_same_metro_pairs" + label] = rate(S.co(ru_, rv_))
            fu2, fv2 = sample_fof_pairs(A, ok, int(args.pairs_fof), rng)
            res["friend_of_friend_pairs" + label] = rate(S.co(fu2, fv2))
        if pname == "test":
            fu, fv, co_f = pu, pv, co_all
            idx = np.flatnonzero(co_f)
            w30 = np.array([S.within_days(int(fu[i]), int(fv[i]), 30) for i in idx], bool)
            res["friend_pairs_within_30_days"] = rate(np.r_[w30, np.zeros(len(fu) - len(idx), bool)])
            bc = np.bincount(B[per == 2], minlength=NB)
            pos_counts = bc[bc > 0]
            thr = float(np.quantile(pos_counts, 0.99)) if len(pos_counts) else np.inf
            top = set(np.flatnonzero(bc >= thr).tolist())
            only_top = np.array([S.common(int(fu[i]), int(fv[i])) <= top for i in idx], bool)
            res["co_friend_pairs_only_top1pct_businesses"] = {"threshold_reviews": thr,
                                                              "share": float(only_top.mean()) if len(idx) else None}
            res["friend_pairs_by_metro"] = []
            for m in range(M):
                s_ = vhome[fu] == m
                if s_.any():
                    res["friend_pairs_by_metro"].append({"metro": m, **rate(co_f[s_])})
        br[pname] = res
    out["base_rates"] = br

    # position / tolerance specifications (structure only)
    keep_obj = {"A": A, "core": core, "eligible_k3": np.flatnonzero((core >= 3) & test_act)}
    if nV >= 2:
        mtr = mt & ready[U]  # reviews with a vector, of taste-ready users only
        ru = vi[U[mtr]]
        rb = B[mtr]
        R = sp.csr_matrix((np.ones(len(ru)), (ru, rb)), shape=(nV, NB))
        C = np.zeros((nV, len(vocab)), np.float32)  # S1 / S3 positions
        inv_n = 1.0 / n_taste[V].astype(np.float64)
        cn2 = np.zeros(nV)
        S_ = np.zeros((len(vocab), len(vocab)))
        mu = np.zeros(len(vocab))
        for s in range(0, nV, 20000):
            blk = (R[s:s + 20000] @ Bv).toarray() * inv_n[s:s + 20000, None]
            C[s:s + 20000] = blk.astype(np.float32)
            blk = C[s:s + 20000].astype(np.float64)  # exactly the stored positions
            cn2[s:s + 20000] = (blk * blk).sum(1)
            S_ += blk.T @ blk
            mu += blk.sum(0)
        del R
        bn2 = np.asarray(Bv.multiply(Bv).sum(1)).ravel()
        dot = row_dots(Bv, rb, C, ru)
        d1 = np.sqrt(np.maximum(bn2[rb] + cn2[ru] - 2 * dot, 0.0))
        del dot
        th1 = group_quantile(ru, d1, 0.5, nV)
        th3 = group_quantile(ru, d1, 0.25, nV)
        # S2: top principal axes of the users' taste centres (between-user variation)
        mu /= nV
        cov = S_ / nV - np.outer(mu, mu)
        w, W = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        top = np.argsort(-w)[:int(args.pca_dim)]
        Wd = W[:, top]
        P = np.zeros((nV, len(top)))
        for s in range(0, nV, 20000):
            P[s:s + 20000] = C[s:s + 20000].astype(np.float64) @ Wd
        Q = np.asarray(Bv @ Wd)
        d2 = np.empty(len(rb))
        for s in range(0, len(rb), 2_000_000):
            d2[s:s + 2_000_000] = np.sqrt(((Q[rb[s:s + 2_000_000]] - P[ru[s:s + 2_000_000]]) ** 2).sum(1))
        th2 = group_quantile(ru, d2, 0.5, nV)
        # G: geographic home and travel radius, as in the meetup application (geo.py): home = median
        # position in the most frequent ~25 km cell of the training reviews, theta = median distance
        mg = tr & ready[U]
        gu = vi[U[mg]]
        glat, glon = ctx["blat"][B[mg]], ctx["blon"][B[mg]]
        okc = np.isfinite(glat) & np.isfinite(glon)
        gu, glat, glon = gu[okc], glat[okc], glon[okc]
        cell = np.floor(glat / 0.25).astype(np.int64) * 100000 + np.floor(glon / 0.25).astype(np.int64)
        o = np.lexsort((cell, gu))
        gu_s, cell_s = gu[o], cell[o]
        grp_start = np.flatnonzero(np.r_[True, (gu_s[1:] != gu_s[:-1]) | (cell_s[1:] != cell_s[:-1])])
        grp_cnt = np.diff(np.r_[grp_start, len(gu_s)])
        grp_u, grp_c = gu_s[grp_start], cell_s[grp_start]
        oo = np.lexsort((grp_c, -grp_cnt, grp_u))
        firsts = oo[np.r_[True, grp_u[oo][1:] != grp_u[oo][:-1]]]
        best_cell = np.full(nV, np.iinfo(np.int64).min)
        best_cell[grp_u[firsts]] = grp_c[firsts]
        inm = cell == best_cell[gu]
        hlat = group_quantile(gu[inm], glat[inm], 0.5, nV)
        hlon = group_quantile(gu[inm], glon[inm], 0.5, nV)
        miss = ~(np.isfinite(hlat) & np.isfinite(hlon))  # no reviewed business with coordinates
        hlat[miss], hlon[miss] = 0.0, 0.0
        dist = haversine_km(hlat[gu], hlon[gu], glat, glon)
        thg = group_quantile(gu, dist, 0.5, nV)
        thg = np.where(np.isfinite(thg), np.maximum(thg, 1.0), 0.0)
        posg = to_xyz(hlat, hlon)

        ncand = {"S1": (C, th1), "S2": (P, th2), "S3": (C, th3), "G": (posg, thg)}
        groups_all = [np.flatnonzero(vhome == m) for m in range(M)]
        rpu, rpv = sample_random_pairs(groups_all, int(args.pairs_random), rng)
        taste = {"S2_between_user_variance_share": float(w[top].sum() / max(w.sum(), 1e-300)),
                 "S2_within_user_variance_share": float((d2 ** 2).mean() / max((d1 ** 2).mean(), 1e-300)),
                 "G_home_missing": int(miss.sum()),
                 "definitions": {
                     "S1": "position = mean L2-normalised category vector of training reviews; theta = median distance "
                           "of those reviews' vectors to the position",
                     "S2": f"S1 projected on the top {int(args.pca_dim)} principal axes of the users' positions; theta = "
                           "median projected distance",
                     "S3": "S1 position; theta = 25th percentile instead of the median",
                     "G": "home = median location in the most frequent 0.25-degree cell of training reviews; theta = "
                          "median great-circle distance of reviewed businesses to home, floor 1 km"}}
        for sname, (X, th) in ncand.items():
            df = pair_dist(X, pu, pv)
            dr = pair_dist(X, rpu, rpv)
            taste[sname] = {"theta": qdict(th), "share_theta_zero": float((th <= 1e-9).mean()),
                            "friend_pairs": ratio_stats(df, th, pu, pv),
                            "random_same_metro_pairs": ratio_stats(dr, th, rpu, rpv),
                            "friend_pair_distance": qdict(df)}
        out["taste"] = taste
        keep_obj["specs"] = {s: (np.asarray(X, np.float32), np.asarray(th, np.float32)) for s, (X, th) in ncand.items()}
    return out, keep_obj


# ------------------------------------------------------------------ rules --

def choose_split(splits: dict, args) -> dict:
    res = {"rule": "precovid if it has >= 2000 eligible queries at k = 3 and a test-period co-review rate of "
                   ">= 0.5% over all edges of its graph (no activity condition); else r4 under the same test; "
                   "else the larger product of the two (flagged underpowered)"}
    ok = {}
    for s in ("precovid", "r4"):
        e = splits[s]["eligible_queries"]["k3"]
        p = splits[s]["base_rates"]["test"]["friend_pairs"]["rate"] or 0.0
        ok[s] = {"eligible_k3": e, "friend_rate": p, "passes": bool(e >= 2000 and p >= 0.005)}
    res["checks"] = ok
    if ok["precovid"]["passes"]:
        res["split"] = "precovid"
    elif ok["r4"]["passes"]:
        res["split"] = "r4"
    else:
        res["split"] = max(ok, key=lambda s: ok[s]["eligible_k3"] * ok[s]["friend_rate"])
        res["underpowered"] = True
    return res


def choose_spec(engine: dict, eligible_k3: int) -> dict:
    res = {"rule": "first of S1, S2, S3 with, on the sampled queries (k = 3, exact with the short time limit): "
                   "(a) share of queries with a found LCC >= 0.10 and share not proven empty <= 0.90; "
                   "(b) median LCC size / connected k-core size <= 0.5 over the queries not proven empty "
                   "(upper bound when not proven, 1 without one); (c) optimality proven (or certified by the "
                   "bound) on >= 80%; (d) eligible k=3 queries x share found >= 500. A specification that "
                   "failed to run, before the chosen one in this order, makes the outcome 'incomplete'. "
                   "Otherwise G under the same test (E5 = Yelp geographic replication); otherwise infeasible.",
           "checks": {}}

    def check(s_):
        e = engine.get(s_)
        if not e or "exact" not in e:
            return None
        x = e["exact"]
        lo, hi, mr, pr = x["share_found"], x["share_not_proven_empty"], x["median_rel_size_open"], x["share_proven"]
        c = {"a_share_found": lo, "a_share_not_proven_empty": hi, "a": bool(lo >= 0.10 and hi <= 0.90),
             "b_median_rel_size": mr, "b": bool(mr is not None and mr <= 0.5),
             "c_share_proven": pr, "c": bool(pr >= 0.80),
             "d_expected_found": float(eligible_k3 * lo), "d": bool(eligible_k3 * lo >= 500)}
        c["passes"] = bool(c["a"] and c["b"] and c["c"] and c["d"])
        return c

    checks = {s_: check(s_) for s_ in ("S1", "S2", "S3", "G")}
    res["checks"] = {s_: (c if c is not None else {"available": False, "error": (engine.get(s_) or {}).get("error")})
                     for s_, c in checks.items()}
    for s_ in ("S1", "S2", "S3", "G"):
        c = checks[s_]
        if c is None:
            res["spec"], res["outcome"] = None, f"incomplete: {s_} did not run"
            return res
        if c["passes"]:
            res["spec"] = s_
            res["outcome"] = ("taste application" if s_ != "G" else
                              "geographic replication (W9); no 'real preference' claim in the ICDE version")
            return res
    res["spec"], res["outcome"] = None, "infeasible as designed"
    return res


# ----------------------------------------------------------------- report --

def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(R: dict, path: Path) -> None:
    L = ["# E5 Yelp feasibility (" + R["version"] + ")", ""]
    inv = R.get("inventory", {})
    L += [f"- zip sha256 ok: {inv.get('zip_sha256_ok')}  missing JSON kinds: {inv.get('missing_kinds')}",
          f"- reviews {R['dataset']['n_reviews']:,}, businesses {R['dataset']['n_business']:,}, users "
          f"{R['dataset']['n_users']:,}; review dates {R['dataset']['review_date_min']} .. {R['dataset']['review_date_max']}"]
    fr = R["friends"]
    L += [f"- friend lists non-empty: {fr['share_users_nonempty']:.3f}; references resolving to dataset users: "
          f"{fr['share_refs_resolved']:.3f}; reciprocated directed pairs: {fr['share_directed_reciprocated']:.3f}; "
          f"undirected edges {fr['undirected_edges']:,}", ""]
    L += ["| metro | businesses | top state / city |", "| --- | --- | --- |"]
    for m in R["metros"]:
        if m["metro"] >= 0:
            L.append(f"| {m['metro']} | {m['businesses']:,} | {m['state_top'][0][0]} / {m['city_top'][0][0]} |")
    L.append("")
    for s, S in R["splits"].items():
        b = S["boundaries"]
        L += [f"## split {s}: train <= {b['train_end']}, val <= {b['val_end']}, test <= {b['test_end']}", ""]
        u, g, br = S["users"], S["graph"], S["base_rates"]
        L += [f"- active users {u.get('active_with_home_metro'):,} (taste-ready {u['taste_ready']:,}); graph "
              f"{g['vertices']:,} vertices, {g['edges']:,} edges; 3-core {g['core_counts']['3']:,}; eligible k=3 "
              f"{S['eligible_queries']['k3']:,}",
              f"- test co-review rate over all edges: friends {_fmt(br['test']['friend_pairs']['rate'], 4)} "
              f"(n={br['test']['friend_pairs']['n']:,}; both active {_fmt(br['test']['friend_pairs_both_active']['rate'], 4)}), "
              f"friend-of-friend {_fmt(br['test']['friend_of_friend_pairs']['rate'], 4)}, random same-metro "
              f"{_fmt(br['test']['random_same_metro_pairs']['rate'], 4)}; within 30 days (friends) "
              f"{_fmt(br['test']['friend_pairs_within_30_days']['rate'], 4)}", ""]
        if "taste" in S:
            L += ["| spec | theta median | friend pairs d/min theta median | share within | random pairs share within |",
                  "| --- | --- | --- | --- | --- |"]
            for sp_ in ("S1", "S2", "S3", "G"):
                t = S["taste"][sp_]
                L.append(f"| {sp_} | {_fmt(t['theta'].get('q50'))} | {_fmt(t['friend_pairs'].get('q50'))} | "
                         f"{_fmt(t['friend_pairs'].get('share_mutually_within'))} | "
                         f"{_fmt(t['random_same_metro_pairs'].get('share_mutually_within'))} |")
            L.append("")
    if "engine" in R:
        E = R["engine"]
        es = E.get("engine_settings", {})
        L += [f"## engine check (split {R['decision']['split']['split']}, k = 3, {E.get('sampled_queries')} queries, "
              f"exact limit {es.get('exact_time_limit_s')} s)", "",
              "| spec | dims | LCC found | not proven empty | median size / k-core | proven | greedy recall |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        if "error" in E:
            L.append(f"engine check failed: {E['error']}")
        for sp_ in ("S1", "S2", "S3", "G"):
            e = E.get(sp_)
            if e and "error" in e:
                L.append(f"| {sp_} | failed: {e['error'][:80]} | | | | | |")
            elif e:
                x = e["exact"]
                L.append(f"| {sp_} | {e['dims']} | {_fmt(x['share_found'])} | {_fmt(x['share_not_proven_empty'])} | "
                         f"{_fmt(x['median_rel_size_open'])} | {_fmt(x['share_proven'])} | "
                         f"{_fmt(e['greedy']['recall_on_proven_nonempty'])} |")
        L.append("")
    d = R.get("decision", {})
    L += ["## decision (pre-declared rule)", "", "```", json.dumps(d, indent=1, ensure_ascii=False), "```"]
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------- main --

def env_info() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
           "scipy": __import__("scipy").__version__, "orjson": bool(_orjson), "cpus": os.cpu_count()}
    try:
        import pandas
        out["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            out["mem_total_kb"] = int(f.readline().split()[1])
    except Exception:
        pass
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path, help="Yelp-JSON.zip as downloaded")
    src.add_argument("--json-dir", type=Path, help="directory that already holds the five JSON files")
    ap.add_argument("--data-dir", type=Path, help="where to extract (default: <zip dir>/json)")
    ap.add_argument("--work", type=Path, default=Path("runs/e5_feas/work"))
    ap.add_argument("--out", type=Path, default=Path("runs/e5_feas/out"))
    ap.add_argument("--expected-sha256", default=EXPECTED_ZIP_SHA256, help="'none' to skip the comparison")
    ap.add_argument("--min-train", type=int, default=10)
    ap.add_argument("--min-taste", type=int, default=5)
    ap.add_argument("--vocab-min-business", type=int, default=20)
    ap.add_argument("--metro-cell", type=float, default=0.5)
    ap.add_argument("--metro-min-business", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=8)
    ap.add_argument("--pairs-random", type=int, default=300_000)
    ap.add_argument("--pairs-fof", type=int, default=200_000)
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--engine-queries", type=int, default=300)
    ap.add_argument("--greedy-queries", type=int, default=100, help="sub-sample for the auxiliary greedy run")
    ap.add_argument("--exact-time", type=float, default=10.0, help="exact time limit per query (s)")
    ap.add_argument("--max-greedy-size", type=int, default=3000, help="greedy skips larger candidate sets")
    ap.add_argument("--budget-exact", type=float, default=1200.0, help="seconds per specification, exact + k-core")
    ap.add_argument("--budget-greedy", type=float, default=600.0, help="seconds per specification, greedy")
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261025)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out / "feasibility.log", mode="a")])
    t_start = time.time()
    R = {"version": VERSION, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env_info(),
         "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if k not in ("zip", "json_dir", "data_dir", "work", "out")}}
    rng = np.random.default_rng(args.seed)
    try:
        expected = None if str(args.expected_sha256).lower() == "none" else args.expected_sha256
        if args.zip:
            data_dir = args.data_dir or (args.zip.parent / "json")
            inv = extract(args.zip, data_dir, expected)
        else:
            data_dir = args.json_dir
            inv = inventory_dir(data_dir)
        R["inventory"] = {k: v for k, v in inv.items() if k not in ("zip_path", "json_dir")}
        (args.out / "inventory.json").write_text(json.dumps(R["inventory"], indent=1, ensure_ascii=False))
        if inv["missing_kinds"]:
            raise RuntimeError(f"missing JSON files: {inv['missing_kinds']}")
        fpath = {f["kind"]: data_dir / f["file"] for f in inv["files"]}

        parsed = load_or_parse(fpath, inv, args.work)
        (blat, blon, bstate, bcity, bcats, bstats, fs, fd, deg_listed, ustats, U, B, D, rstats, NU, NB) = parsed
        R["inventory"]["reused_parse"] = bool(parsed_reused(args.work))

        yrs = (D.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970) if len(D) else np.zeros(0, int)
        yc = Counter(yrs.tolist())
        R["dataset"] = {"n_business": NB, "n_users": NU, "n_reviews": int(len(U)), "business_stats": bstats,
                        "user_stats": {k: v for k, v in ustats.items() if not k.startswith("friend")},
                        "review_stats": rstats, "review_date_min": day_str(D.min()) if len(D) else None,
                        "review_date_max": day_str(D.max()) if len(D) else None,
                        "reviews_per_year": {str(y): int(yc[y]) for y in sorted(yc)},
                        "tip_lines": next((f["newlines"] for f in inv["files"] if f["kind"] == "tip"), None),
                        "checkin_lines": next((f["newlines"] for f in inv["files"] if f["kind"] == "checkin"), None),
                        "reviews_per_user_with_reviews": qdict(np.bincount(U, minlength=NU)[np.bincount(U, minlength=NU) > 0])}

        # friendship: reciprocity and the undirected graph (union of both directions)
        code = fs * NU + fd
        rev = fd * NU + fs
        code_sorted = np.unique(code)
        recip = np.isin(rev, code_sorted)
        lo, hi = np.minimum(fs, fd), np.maximum(fs, fd)
        und = np.unique(lo * NU + hi)
        eu, ev = und // NU, und % NU
        R["friends"] = {**{k: v for k, v in ustats.items() if k.startswith("friend") or k.startswith("users_with")},
                        "share_users_nonempty": ustats.get("users_with_nonempty_friends", 0) / max(NU, 1),
                        "share_refs_resolved": ustats.get("friend_refs_resolved", 0) / max(ustats.get("friend_refs", 0), 1),
                        "share_directed_reciprocated": float(recip.mean()) if len(recip) else 0.0,
                        "duplicate_directed_refs": int(len(code) - len(code_sorted)),
                        "undirected_edges": int(len(und)),
                        "listed_friends_per_user_nonempty": qdict(deg_listed[deg_listed > 0])}
        del code, rev, code_sorted, recip, lo, hi, und, fs, fd

        bmetro, minfo = find_metros(blat, blon, bstate, bcity, args.metro_cell, args.metro_min_business)
        R["metros"] = minfo
        M = sum(1 for m in minfo if m["metro"] >= 0)
        rmetro = bmetro[B]
        R["dataset"]["reviews_outside_metros"] = int((rmetro < 0).sum())
        ctx = {"U": U, "B": B, "D": D, "NU": NU, "NB": NB, "n_metros": M, "bmetro": bmetro, "cats": bcats,
               "blat": blat, "blon": blon, "eu": eu, "ev": ev}
        (args.out / "feasibility.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))

        R["splits"] = {}
        keep = {}
        for s in ("precovid", "r4"):
            t0 = time.time()
            R["splits"][s], keep[s] = analyse_split(s, ctx, args, rng)
            LOG.info("split %s done (%.0f s): eligible k3 %d, friend test co-review %.4f", s, time.time() - t0,
                     R["splits"][s]["eligible_queries"]["k3"],
                     R["splits"][s]["base_rates"]["test"]["friend_pairs"]["rate"] or 0.0)
            (args.out / "feasibility.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))

        dec = {"split": choose_split(R["splits"], args)}
        sname = dec["split"]["split"]
        if not args.no_engine and "specs" in keep[sname]:
            try:
                R["engine"] = engine_check(keep[sname]["A"], keep[sname]["specs"], keep[sname]["eligible_k3"], args,
                                           args.work / sname, rng)
            except Exception as e:  # report and go on: the rest of the check is still useful
                R["engine"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                LOG.error("engine check failed: %s", e)
        if args.no_engine:
            dec["spec"] = {"spec": None, "outcome": "engine check skipped (--no-engine)"}
        elif "specs" not in keep[sname] or R["splits"][sname]["eligible_queries"]["k3"] == 0:
            dec["spec"] = {"spec": None, "outcome": "infeasible as designed: no taste-ready users or no eligible "
                                                     "queries at k = 3 in the chosen split"}
        elif "engine" in R and "error" not in R["engine"]:
            dec["spec"] = choose_spec(R["engine"], R["splits"][sname]["eligible_queries"]["k3"])
        else:
            dec["spec"] = {"spec": None, "outcome": "incomplete: engine check not available"}
        R["decision"] = dec
        R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        R["seconds"] = round(time.time() - t_start, 1)
        (args.out / "feasibility.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        write_report(R, args.out / "FEASIBILITY.md")
        LOG.info("decision: split %s, spec %s (%s)", sname, dec["spec"].get("spec"), dec["spec"].get("outcome"))
        if str(dec["spec"].get("outcome", "")).startswith("incomplete"):
            print("FEASIBILITY INCOMPLETE")  # results written, but the decision could not be taken
            return 2
        print("FEASIBILITY DONE")
        return 0
    except Exception as e:
        R["error"] = f"{type(e).__name__}: {e}"
        R["traceback"] = traceback.format_exc()
        (args.out / "feasibility.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
        LOG.error("FEASIBILITY FAILED: %s\n%s", e, R["traceback"])
        print("FEASIBILITY FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
