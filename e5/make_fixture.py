"""Synthetic Yelp-format data packaged like Yelp-JSON.zip (tests of yelp_feasibility.py only).

zip:  'Yelp JSON/' + documentation PDF + 'Yelp JSON/yelp_dataset.tar' (+ __MACOSX entries);
tar:  gzip-compressed despite its name, holding the five yelp_academic_dataset_*.json files and a
      PDF. Planted structure: three metros plus scattered businesses, eight taste types, friends
      mostly within a metro and of the same type, friend lists as strings ("a, b"), "None" or JSON
      lists, unknown and self references, a few one-directional listings, reviews 2008-2022 with
      a dip in 2020-21, co-reviews of friends, and a few bad records (unknown user / business,
      invalid date). truth.json records what the checker must find.
"""
from __future__ import annotations

import io
import json
import string
import tarfile
import zipfile
from pathlib import Path

import numpy as np

METROS = [(39.95, -75.16, "PA", "Philadelphia"), (27.95, -82.46, "FL", "Tampa"), (39.53, -119.81, "NV", "Reno")]
TYPES = 8


def _id(rng, used: set) -> str:
    alpha = string.ascii_letters + string.digits + "-_"
    while True:
        s = "".join(rng.choice(list(alpha), 22))
        if s not in used:
            used.add(s)
            return s


def make(out_dir: Path, seed: int = 0, users_per_metro: int = 260, biz_per_metro: int = 220) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set = set()
    truth = {"metros": {}}

    # ---- businesses
    biz = []  # (id, metro, type, lat, lon, state, city, cats)
    for m, (la, lo, st, ct) in enumerate(METROS):
        for _ in range(biz_per_metro):
            t = int(rng.integers(TYPES))
            cats = [f"Type{t}-{s}" for s in rng.choice(list("ABCD"), size=int(rng.integers(1, 4)), replace=False)]
            if rng.random() < 0.6:
                cats.append("Restaurants")
            if rng.random() < 0.3:
                cats.append("Food")
            if rng.random() < 0.02:
                cats.append(f"Rare{int(rng.integers(1000))}")
            biz.append([_id(rng, used), m, t, la + rng.normal(0, 0.08), lo + rng.normal(0, 0.08), st, ct, cats])
        truth["metros"][st] = biz_per_metro
    n_scatter = 25
    for i in range(n_scatter):  # far from every metro and from each other: no metro of their own
        biz.append([_id(rng, used), -1, int(rng.integers(TYPES)), 44.0 + 2.0 * (i % 5), -100.0 + 2.0 * (i // 5),
                    "SD", "Nowhere", [f"Type{int(rng.integers(TYPES))}-A"]])
    for b in biz:  # 4% without categories
        if rng.random() < 0.04:
            b[7] = None
    for b in biz[:6]:  # and the first six businesses of metro 0 (reviewed by the 'not taste-ready' user)
        b[7] = None
    nb = len(biz)
    by_metro_type = {}
    for i, b in enumerate(biz):
        if b[1] >= 0:
            by_metro_type.setdefault((b[1], b[2]), []).append(i)
    by_metro_type = {k: np.array(v) for k, v in by_metro_type.items()}
    by_metro = {m: np.array([i for i, b in enumerate(biz) if b[1] == m]) for m in range(len(METROS))}

    # ---- users
    users = []  # (id, metro, type, breadth, n_reviews)
    for m in range(len(METROS)):
        for _ in range(users_per_metro):
            t = int(rng.integers(TYPES))
            breadth = float(rng.choice([0.55, 0.8, 0.95]))
            n = int(np.clip(rng.lognormal(np.log(22), 0.7), 1, 200))
            users.append([_id(rng, used), m, t, breadth, n])
    n_noreview = 40
    for _ in range(n_noreview):
        users.append([_id(rng, used), int(rng.integers(len(METROS))), int(rng.integers(TYPES)), 0.8, 0])
    # an active user who is not taste-ready: 12 training reviews, only 2 of them at businesses with
    # categories (reviews added below; n_reviews 0 keeps the generic generator away)
    s4 = len(users)
    users.append([_id(rng, used), 0, 0, 0.8, 0])
    nu = len(users)

    # ---- friendships (undirected truth), then listings
    und = set()
    for m in range(len(METROS)):
        idx = [i for i, u in enumerate(users) if u[1] == m]
        if len(idx) <= 1000:  # small fixture: every pair, homophily by type
            for a_i, a in enumerate(idx):
                for b in idx[a_i + 1:]:
                    p = 0.10 if users[a][2] == users[b][2] else 0.012
                    if rng.random() < p:
                        und.add((a, b))
        else:  # large fixture (timing only): a few same-type and random friends per user, O(n)
            same = {}
            for i in idx:
                same.setdefault(users[i][2], []).append(i)
            same = {t: np.array(v) for t, v in same.items()}
            idx_a = np.array(idx)
            for a in idx:
                st = same[users[a][2]]
                for b in st[rng.integers(0, len(st), int(rng.poisson(3)))].tolist() + \
                        idx_a[rng.integers(0, len(idx_a), int(rng.poisson(1.5)))].tolist():
                    if a != b:
                        und.add((min(a, b), max(a, b)))
    for b in [i for i, u in enumerate(users) if u[1] == 0][:3]:  # friends of the special user
        und.add((min(s4, b), max(s4, b)))
    for a in range(nu):  # a few cross-metro friendships
        if rng.random() < 0.05:
            b = int(rng.integers(nu))
            if users[b][1] != users[a][1]:
                und.add((min(a, b), max(a, b)))
    listed = {i: [] for i in range(nu)}
    one_dir = 0
    for a, b in sorted(und):
        if rng.random() < 0.03:  # listed by one side only
            listed[a].append(users[b][0])
            one_dir += 1
        else:
            listed[a].append(users[b][0])
            listed[b].append(users[a][0])
    unknown_refs = 0
    self_refs = 0
    for i in range(nu):
        if rng.random() < 0.10:
            k = int(rng.integers(1, 4))
            for _ in range(k):
                listed[i].append(_id(rng, used))
            unknown_refs += k
        if i % 97 == 0:
            listed[i].append(users[i][0])
            self_refs += 1
    truth["friend_refs"] = int(sum(len(v) for v in listed.values()))
    truth["friend_refs_unknown"] = unknown_refs
    truth["friend_refs_self"] = self_refs
    truth["one_directional_pairs"] = one_dir
    truth["undirected_edges_known"] = len(und)

    # ---- reviews
    years = np.arange(2008, 2022)
    yw = np.array([1, 1.3, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5, 2.4, 2.8], float)
    yw /= yw.sum()

    def rand_day():
        y = int(rng.choice(years, p=yw))
        return np.datetime64(f"{y}-01-01") + np.timedelta64(int(rng.integers(0, 365)), "D")

    reviews = []  # (user_idx or None-string, biz_idx or None-string, date string)
    for ui, u in enumerate(users):
        for _ in range(u[4]):
            m = u[1] if rng.random() < 0.9 else int(rng.integers(len(METROS)))
            pool = by_metro_type.get((m, u[2])) if rng.random() < u[3] else None
            if pool is None or len(pool) == 0:
                pool = by_metro[m]
            reviews.append((ui, int(pool[int(rng.integers(len(pool)))]), rand_day()))
    for a, b in sorted(und):  # planted co-reviews
        if users[a][1] != users[b][1] or users[a][4] == 0 or users[b][4] == 0:
            continue
        for _ in range(int(rng.poisson(1.2))):
            m = users[a][1]
            pool = by_metro_type.get((m, users[a][2]))
            if pool is None or len(pool) == 0:
                pool = by_metro[m]
            bi = int(pool[int(rng.integers(len(pool)))])
            d = rand_day()
            reviews.append((a, bi, d))
            reviews.append((b, bi, d + np.timedelta64(int(rng.integers(0, 20)), "D")))
    cat0 = [i for i in by_metro[0].tolist() if biz[i][7]]
    for j in range(10):
        reviews.append((s4, j % 6, np.datetime64("2015-03-01") + np.timedelta64(j, "D")))
    for j in range(2):
        reviews.append((s4, cat0[j], np.datetime64("2016-03-01") + np.timedelta64(j, "D")))
    reviews.append((s4, cat0[3], np.datetime64("2019-05-05")))
    truth["not_taste_ready_active_user"] = users[s4][0]
    order = rng.permutation(len(reviews))
    reviews = [reviews[i] for i in order]
    truth["n_reviews_valid"] = len(reviews)
    bad = [("UNKNOWN_USER", 0, np.datetime64("2015-05-05")), ("UNKNOWN_USER", 1, np.datetime64("2016-06-06")),
           (0, "UNKNOWN_BIZ", np.datetime64("2017-07-07")), (1, 2, "2019-13-45")]
    truth["unknown_user"], truth["unknown_business"], truth["bad_date"] = 2, 1, 1
    truth["n_business"], truth["n_users"] = nb, nu

    # ---- JSON files
    def biz_line(b):
        return {"business_id": b[0], "name": "Shop " + b[0][:5], "address": "1 Main St", "city": b[6], "state": b[5],
                "postal_code": "00000", "latitude": round(float(b[3]), 7), "longitude": round(float(b[4]), 7),
                "stars": 4.0, "review_count": 10, "is_open": 1,
                "attributes": {"RestaurantsPriceRange2": "2"} if b[7] and "Restaurants" in b[7] else None,
                "categories": ", ".join(b[7]) if b[7] else None, "hours": None}

    def user_line(i, u):
        fl = listed[i]
        if not fl:
            friends = "None"
        elif i % 20 == 0:
            friends = list(fl)  # JSON list format in a few records
        else:
            friends = ", ".join(fl)
        return {"user_id": u[0], "name": "U", "review_count": u[4], "yelping_since": "2010-01-01 00:00:00",
                "useful": 0, "funny": 0, "cool": 0, "elite": "", "friends": friends, "fans": 0, "average_stars": 4.0,
                "compliment_hot": 0, "compliment_more": 0}

    def review_line(r, j):
        u, b, d = r
        uid = users[u][0] if isinstance(u, int) else u
        bid = biz[b][0] if isinstance(b, int) else b
        ds = d if isinstance(d, str) else f"{d} {int(rng.integers(24)):02d}:{int(rng.integers(60)):02d}:00"
        return {"review_id": f"r{j:09d}xxxxxxxxxxxx", "user_id": uid, "business_id": bid, "stars": 4.0, "useful": 0,
                "funny": 0, "cool": 0, "text": "Nice place, would \"go\" again.\nYes.", "date": ds}

    files = {
        "yelp_academic_dataset_business.json": "\n".join(json.dumps(biz_line(b)) for b in biz) + "\n",
        "yelp_academic_dataset_user.json": "\n".join(json.dumps(user_line(i, u)) for i, u in enumerate(users)) + "\n",
        "yelp_academic_dataset_review.json": "\n".join(json.dumps(review_line(r, j))
                                                       for j, r in enumerate(reviews + bad)) + "\n",
        "yelp_academic_dataset_tip.json": "\n".join(json.dumps({"user_id": users[0][0], "business_id": biz[0][0],
                                                                "text": "tip", "date": "2018-01-01 00:00:00",
                                                                "compliment_count": 0}) for _ in range(30)) + "\n",
        "yelp_academic_dataset_checkin.json": "\n".join(json.dumps({"business_id": b[0], "date": "2018-01-01 00:00:00"})
                                                        for b in biz[:20]) + "\n",
    }
    truth["tip_lines"], truth["checkin_lines"] = 30, 20
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
        for name, text in list(files.items()) + [("Dataset_User_Agreement.pdf", "%PDF-1.4 dummy\n")]:
            data = text.encode("utf-8")
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    zpath = out_dir / "Yelp-JSON.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Yelp JSON/", b"")
        zf.writestr("Yelp JSON/Yelp Dataset Documentation & ToS copy.pdf", b"%PDF-1.4 terms\n")
        zf.writestr("__MACOSX/Yelp JSON/._Yelp Dataset Documentation & ToS copy.pdf", b"\x00\x05\x16\x07junk")
        zf.writestr("Yelp JSON/yelp_dataset.tar", tar_buf.getvalue())
        zf.writestr("__MACOSX/Yelp JSON/._yelp_dataset.tar", b"\x00\x05\x16\x07junk")
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    for name, text in files.items():  # plain copies for the reference implementation of the tests
        (raw / name).write_text(text)
    return truth


if __name__ == "__main__":
    import sys

    print(json.dumps(make(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixture")), indent=1))
