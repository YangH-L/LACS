"""Synthetic E6 inputs in the real formats (tests only).

make(root) writes
  root/offline/HS117_members.csv               Voteview format: President row, a member with a House and
                                               a Senate row (the one with more votes must win), rows
                                               without dim1, independents (328), members outside the
                                               network
  root/offline/billstatus/<type>/              BILLSTATUS XML in both formats (old <billType>/<billNumber>
                                               with empty withdrawn tags, new <type>/<number> without them),
                                               decoy amendment (co)sponsors deeper in the tree, withdrawn
                                               cosponsorships, sponsors without a NOMINATE score;
                                               s: a .zip plus one listed file outside it, listing.xml;
                                               sjres: a .zip only, listing names only the package;
                                               hr: loose files, listing.xml; hjres: loose files, listing.json
  root/network.json, root/crosswalk.csv        a Twitter network in the real format and its crosswalk
  root/truth.json                              what build must find, computed here independently
Twitter edges carry a planted acceptance effect: P(v -> u) rises when |dim1_u - dim1_v| <= theta_u
(theta_u = the tolerance estimated from the fixture's own cosponsorships), on top of distance.
"""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np

TRAIN, VAL, TEST = ("2021-01-03", "2022-02-08"), ("2022-02-09", "2022-06-09"), ("2022-06-10", "2023-01-03")
DAY0 = np.datetime64("2021-01-04")
VV_COLS = ["congress", "chamber", "icpsr", "state_abbrev", "district_code", "state_icpsr", "party_code", "occupancy",
           "last_means", "bioname", "bioguide_id", "born", "died", "nominate_dim1", "nominate_dim2",
           "nominate_log_likelihood", "nominate_geo_mean_probability", "nominate_number_of_votes",
           "nominate_number_of_errors", "conditional", "nokken_poole_dim1", "nokken_poole_dim2"]
STATES = ["AL", "AK", "AZ", "CA", "CO", "FL", "GA", "IL", "NY", "TX", "OH", "PA", "WA", "MI"]


def _period(d: str) -> int:
    for i, (lo, hi) in enumerate((TRAIN, VAL, TEST)):
        if lo <= d <= hi:
            return i
    return -1


def _xml_bill(fmt: str, btype: str, num: int, intro: str, sponsor, cos: list, decoy: list) -> str:
    tag_t, tag_n = ("billType", "billNumber") if fmt == "old" else ("type", "number")
    out = ['<?xml version="1.0" encoding="utf-8" standalone="no"?>', "<billStatus>"]
    if fmt == "new":
        out.append("  <version>3.0.0</version>")
    out += ["  <bill>", f"    <{tag_n}>{num}</{tag_n}>", f"    <{tag_t}>{btype.upper()}</{tag_t}>",
            f"    <introducedDate>{intro}</introducedDate>", "    <congress>117</congress>",
            "    <committees><item><name>Some Committee</name><type>Standing</type></item></committees>"]
    if sponsor is None:
        out.append("    <sponsors/>")
    else:
        out.append(f"    <sponsors><item><bioguideId>{sponsor}</bioguideId><fullName>X</fullName></item></sponsors>")
    out.append("    <cosponsors>")
    for b, date, orig, wd in cos:
        out.append(f"      <item><bioguideId>{b}</bioguideId><fullName>Y</fullName>"
                   f"<sponsorshipDate>{date}</sponsorshipDate><isOriginalCosponsor>{orig}</isOriginalCosponsor>")
        if wd:
            out.append(f"        <sponsorshipWithdrawnDate>{wd}</sponsorshipWithdrawnDate>")
        elif fmt == "old":
            out.append("        <sponsorshipWithdrawnDate/>" if num % 4 else "        <sponsorshipWithdrawnDate />")
        out.append("      </item>")
    out.append("    </cosponsors>")
    if decoy:  # amendment (co)sponsors must not be read as the bill's
        out.append("    <amendments><amendment><number>1</number>")
        out.append(f"      <sponsors><item><bioguideId>{decoy[0]}</bioguideId></item></sponsors><cosponsors>")
        for b in decoy[1:]:
            out.append(f"        <item><bioguideId>{b}</bioguideId><sponsorshipDate>{intro}</sponsorshipDate>"
                       "<isOriginalCosponsor>True</isOriginalCosponsor></item>")
        out.append("      </cosponsors></amendment></amendments>")
    out += ["  </bill>", "  <dublinCore/>", "</billStatus>", ""]
    return "\n".join(out)


def _listing_xml(names: list[str], t: str) -> str:
    rows = ["<?xml version='1.0' encoding='UTF-8'?>", "<files>"]
    for n in names:
        ext = n.rsplit(".", 1)[-1]
        rows.append(f"<file><displayLabel>{n}</displayLabel><folder>false</folder><justFileName>{n}</justFileName>"
                    f"<link>https://www.govinfo.gov/bulkdata/BILLSTATUS/117/{t}/{n}</link>"
                    f"<fileExtension>{ext}</fileExtension><size>100</size></file>")
    rows.append("<file><displayLabel>resources</displayLabel><folder>true</folder>"
                "<justFileName>resources</justFileName><link>x</link></file>")
    rows.append("</files>")
    return "\n".join(rows)


def make(root, seed: int = 7, n_accounts: int = 140, n_extra: int = 40, n_senate: int = 30,
         house_bills: int = 560, senate_bills: int = 300, effect: float = 2.0) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    off = root / "offline"
    off.mkdir(parents=True, exist_ok=True)

    # ---- members: network accounts first, then members outside the network
    n_all = n_accounts + n_extra
    bio = [f"X{i:06d}" for i in range(n_all)]
    senate = np.zeros(n_all, bool)
    senate[:n_senate] = True
    senate[n_accounts:n_accounts + 8] = True
    party = np.where(rng.random(n_all) < 0.5, 100, 200)
    party[[3, 17]] = 328  # two independents in the Senate
    dim1 = np.where(party == 200, rng.normal(0.5, 0.15, n_all), rng.normal(-0.38, 0.12, n_all)).clip(-0.98, 0.98)
    dim2 = rng.normal(0.0, 0.3, n_all).clip(-0.98, 0.98)
    dim1, dim2 = np.round(dim1, 3), np.round(dim2, 3)
    theta_true = np.exp(rng.normal(np.log(0.3), 0.35, n_all))
    act = rng.uniform(0.22, 0.45, n_all)
    low = np.arange(40, 48)  # low activity: fewer than 10 training cosponsorships
    act[low] = 0.012
    no_vv = [60, 61, 62, 63]  # not in Voteview at all
    no_dim1 = [64, 65]  # Voteview rows without dim1
    dual = 5  # a senator with an extra (smaller) House row
    state = [STATES[i % len(STATES)] for i in range(n_all)]
    in_vv = np.ones(n_all, bool)
    in_vv[no_vv] = False
    has_pos = in_vv.copy()
    has_pos[no_dim1] = False

    with open(off / "HS117_members.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(VV_COLS)

        def row(i, chamber, d1, d2, votes):
            r = {c: "" for c in VV_COLS}
            r.update({"congress": 117, "chamber": chamber, "icpsr": 10000 + i, "state_abbrev": state[i],
                      "district_code": 0 if chamber == "Senate" else 1, "party_code": int(party[i]),
                      "bioname": f"MEMBER, Number {i}", "bioguide_id": bio[i], "nominate_dim1": d1,
                      "nominate_dim2": d2, "nominate_number_of_votes": votes, "nominate_number_of_errors": 3})
            w.writerow([r[c] for c in VV_COLS])

        w.writerow([117, "President", 99869, 99, 0, 99, 100, "", "", "BIDEN, Joseph Robinette, Jr.", "B000444",
                    1942, "", -0.4, 0.2, "", "", 10, "", "", "", ""])
        for i in range(n_all):
            if not in_vv[i]:
                continue
            ch = "Senate" if senate[i] else "House"
            if i in no_dim1:
                row(i, ch, "", "", 0)
                continue
            row(i, ch, dim1[i], dim2[i], 600)
            if i == dual:
                row(i, "House", -0.9, 0.9, 50)  # fewer votes: must be ignored

    # ---- bills and cosponsorships
    nv_sponsors = ["Z000001", "Z000002"]  # sponsors without a NOMINATE score
    rows = []  # (type, num, intro, sponsor, [(bio, date, orig, withdrawn)])
    pos_ok = has_pos
    for chamber, nb, types in (("House", house_bills, ("hr", "hjres")), ("Senate", senate_bills, ("s", "sjres"))):
        cand = np.flatnonzero((senate if chamber == "Senate" else ~senate) & pos_ok)
        cand_all = np.flatnonzero(senate if chamber == "Senate" else ~senate)
        counters = {t: 0 for t in types}
        for k in range(nb):
            t = types[0] if k % 12 else types[1]
            counters[t] += 1
            num = counters[t]
            intro = str(DAY0 + np.timedelta64(int(rng.integers(0, 715)), "D"))
            r = rng.random()
            sponsor = (nv_sponsors[k % 2] if r < 0.05 else None if r < 0.06 else bio[int(rng.choice(cand))])
            sp_i = bio.index(sponsor) if (sponsor and sponsor.startswith("X")) else None
            cos = []
            for u in cand_all:
                if u == sp_i:
                    continue
                if sp_i is not None and pos_ok[u]:
                    inside = abs(dim1[u] - dim1[sp_i]) <= theta_true[u]
                else:
                    inside = rng.random() < 0.3
                p = act[u] * (0.5 if inside else 0.02)
                if rng.random() < p:
                    if rng.random() < 0.6:
                        date, orig = intro, "True"
                    else:
                        date = str(np.datetime64(intro) + np.timedelta64(int(rng.exponential(30)) + 1, "D"))
                        orig = "False"
                    if date > "2023-01-03":
                        continue
                    wd = str(np.datetime64(date) + np.timedelta64(10, "D")) if rng.random() < 0.03 else None
                    cos.append((bio[u], date, orig, wd))
            rows.append((t, num, intro, sponsor, cos))
    # a member cosponsors the same bill twice (withdrawn, then again): only the second counts
    t, num, intro, sponsor, cos = rows[3]
    if cos:
        b0 = cos[0][0]
        cos[0] = (b0, cos[0][1], cos[0][2], str(np.datetime64(cos[0][1]) + np.timedelta64(3, "D")))
        cos.append((b0, str(np.datetime64(cos[0][1]) + np.timedelta64(5, "D")), "False", None))

    # ---- XML files and listings
    by_type: dict[str, dict[str, str]] = {}
    for k, (t, num, intro, sponsor, cos) in enumerate(rows):
        fmt = "old" if num % 2 else "new"
        decoy = [bio[(k * 7) % n_accounts], bio[(k * 11 + 3) % n_accounts]] if k % 5 == 0 else []
        by_type.setdefault(t, {})[f"BILLSTATUS-117{t}{num}.xml"] = _xml_bill(fmt, t, num, intro, sponsor, cos, decoy)
    for t, files in by_type.items():
        d = off / "billstatus" / t
        d.mkdir(parents=True, exist_ok=True)
        names = sorted(files)
        if t in ("s", "sjres"):
            zname = f"BILLSTATUS-117-{t}.zip"
            outside = names[-1] if t == "s" else None  # listed but not in the package
            with zipfile.ZipFile(d / zname, "w", zipfile.ZIP_DEFLATED) as z:
                for n in names:
                    if n != outside:
                        z.writestr(f"BILLSTATUS-117-{t}/{n}", files[n])
            if outside:
                (d / outside).write_text(files[outside])
            listed = [zname] + (names if t == "s" else [])
            (d / "listing.xml").write_text(_listing_xml(listed, t))
        else:
            for n in names:
                (d / n).write_text(files[n])
            if t == "hjres":
                doc = {"files": [{"justFileName": n, "fileExtension": "xml", "folder": False,
                                  "link": f"https://www.govinfo.gov/bulkdata/BILLSTATUS/117/{t}/{n}"} for n in names]}
                (d / "listing.json").write_text(json.dumps(doc))
            else:
                (d / "listing.xml").write_text(_listing_xml(names, t))

    # ---- truth: tolerances from the fixture's own records (independent of e6.build)
    idx = {b: i for i, b in enumerate(bio)}
    dists1, dists2 = {}, {}
    joins = []  # (bill key, member, period) of counted joins
    for t, num, intro, sponsor, cos in rows:
        key = {"hr": 1, "s": 2, "hjres": 3, "sjres": 4}[t] * 100000 + num
        first = {}
        if sponsor:
            first[sponsor] = intro
        for b, date, orig, wd in cos:
            if wd:
                continue
            first[b] = min(first.get(b, "9999"), date)
            if _period(date) == 0 and sponsor and sponsor.startswith("X") and has_pos[idx[sponsor]] and has_pos[idx[b]]:
                s_i, u_i = idx[sponsor], idx[b]
                dists1.setdefault(b, []).append(abs(dim1[u_i] - dim1[s_i]))
                dists2.setdefault(b, []).append(float(np.hypot(dim1[u_i] - dim1[s_i], dim2[u_i] - dim2[s_i])))
        for b, date in first.items():
            joins.append((key, b, _period(date)))

    def q(v, p):
        return float(max(np.quantile(np.array(v, float), p), 0.02))

    theta_hat = {b: {"q50": q(v, 0.5), "q70": q(v, 0.7), "q90": q(v, 0.9), "n": len(v)} for b, v in dists1.items()}
    theta2 = {b: {"q50": q(v, 0.5), "q70": q(v, 0.7), "q90": q(v, 0.9), "n": len(v)} for b, v in dists2.items()}
    eligible = sorted(b for b in bio[:n_accounts] if has_pos[idx[b]] and theta_hat.get(b, {}).get("n", 0) >= 10)

    # ---- network among the accounts, planted acceptance effect
    handles = [f"Rep{b}" if not senate[i] else f"Sen{b}" for i, b in enumerate(bio[:n_accounts])]
    a_u = rng.normal(0, 0.5, n_accounts)
    b_v = rng.normal(0, 0.5, n_accounts)
    out_l = [[] for _ in range(n_accounts)]
    out_w = [[] for _ in range(n_accounts)]
    for u in range(n_accounts):
        th = theta_hat.get(bio[u], {}).get("q50", theta_true[u])
        for v in range(n_accounts):
            if u == v:
                continue
            d = abs(dim1[u] - dim1[v]) if (has_pos[u] and has_pos[v]) else 0.8
            same = (party[u] == 200) == (party[v] == 200)
            eta = -3.0 + effect * (d <= th) - 2.0 * d + 0.5 * same + a_u[u] + b_v[v]
            if rng.random() < 1.0 / (1.0 + np.exp(-eta)):  # edge v -> u: u engaged with v
                out_l[v].append(u)
                out_w[v].append(float(rng.integers(1, 20)) / float(rng.integers(50, 2000)))
    in_l = [[] for _ in range(n_accounts)]
    in_w = [[] for _ in range(n_accounts)]
    for v in range(n_accounts):
        for u, x in zip(out_l[v], out_w[v]):
            in_l[u].append(v)
            in_w[u].append(x)
    with open(root / "network.json", "w") as f:
        json.dump([{"inList": in_l, "inWeight": in_w, "outList": out_l, "outWeight": out_w,
                    "usernameList": handles}], f)
    with open(root / "crosswalk.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["handle", "node", "bioguide", "name", "chamber", "state", "method", "note"])
        for i in range(n_accounts):
            w.writerow([handles[i], i, bio[i], f"Member {i}", "Senate" if senate[i] else "House", state[i],
                        "manual" if i % 30 == 0 else "auto", ""])
    truth = {"bioguides": bio[:n_accounts], "eligible": eligible, "theta_q50": {b: theta_hat[b]["q50"] for b in eligible},
             "theta_q70": {b: theta_hat[b]["q70"] for b in eligible},
             "theta_q90": {b: theta_hat[b]["q90"] for b in eligible},
             "theta2d_q50": {b: theta2[b]["q50"] for b in eligible}, "n_train": {b: theta_hat[b]["n"] for b in eligible},
             "dual_member": bio[dual], "dual_dim1": float(dim1[dual]), "no_vv": [bio[i] for i in no_vv],
             "no_dim1": [bio[i] for i in no_dim1], "low": [bio[i] for i in low],
             "joins": [list(x) for x in joins], "n_bills": len(rows),
             "edges": int(sum(len(x) for x in out_l))}
    with open(root / "truth.json", "w") as f:
        json.dump(truth, f)
    return truth
