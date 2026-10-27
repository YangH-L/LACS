#!/usr/bin/env python3
"""Twitter handle -> bioguide crosswalk for the Congressional Twitter network (run once, locally).

Writes into e6/data/ (shipped with the package; plan sections 2 and 10):
  congress_network_data.json            copy of the network (gsprint23/CongressionalTwitterNetwork
                                        commit 205ca40d, MIT; LICENSE copied next to it)
  crosswalk.csv                         handle, node, bioguide, name, chamber, state, method, note
  MANIFEST.json                         sha256, sizes and provenance of the shipped files

Matching (plan section 2): a handle is matched automatically when it equals, ignoring case,
`social.twitter` of a legislator in legislators-social-media.yaml of unitedstates/congress-legislators
at commit b6c8c0bd (2022-05-26); the remaining accounts come from the hand-made table of the plan's
appendix. Every manual row is checked against legislators-current.yaml / legislators-historical.yaml
(bioguide exists, last name, state, chamber, a term overlapping the tweet period 2022-02-09..06-09),
and the finished table must have one row per network node and no bioguide twice; the script
exits non-zero otherwise.

Correction of the appendix (found by the uniqueness check and reported, not applied silently):
the appendix maps CongresswomanSC to Nancy Mace (M000194), but M000194 is already matched
automatically through her account RepNancyMace, which is also in the network. CongresswomanSC
is the account of Sheila Cherfilus-McCormick (C001127, House, FL-20, sworn in 2022-01-18; "SC" are
her initials). She has no Twitter account in the 2022-05 social table, and the account's network
neighbours are Democrats, most of them from Florida (the script prints and records this count). Set
APPLY_CORRECTIONS = False to reproduce the appendix literally (the uniqueness check then fails).

  python e6/make_crosswalk.py --current data/e6/current_2022-06.yaml \
      --social data/e6/social_2022-06.yaml --historical data/e6/historical_2022-06.yaml \
      --network data/e6/CongressionalTwitterNetwork/congress_network_data.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import DATA_DIR, now_iso, sha256_file, write_json  # noqa: E402

E6DATA = Path("data/e6")
NETWORK_REPO = "https://github.com/gsprint23/CongressionalTwitterNetwork"
NETWORK_COMMIT = "205ca40d973e13944051e1e44240d005a9b1a1bc"
NETWORK_SHA256 = "ee6d2fc1a2ac07e6481d41c0df463e2e7c5c0db4ae964980b43233e5c7253712"
LEGIS_REPO = "https://github.com/unitedstates/congress-legislators"
LEGIS_COMMIT = "b6c8c0bd07e48f96ff077ceec08aaff139e34496"
TWEET_PERIOD = ("2022-02-09", "2022-06-09")

# plan appendix: handle, member, bioguide, chamber, state
MANUAL_PLAN = [
    ("CoryBooker", "Cory A. Booker", "B001288", "Senate", "NJ"),
    ("SenGillibrand", "Kirsten E. Gillibrand", "G000555", "Senate", "NY"),
    ("LindseyGrahamSC", "Lindsey Graham", "G000359", "Senate", "SC"),
    ("HawleyMO", "Josh Hawley", "H001089", "Senate", "MO"),
    ("timkaine", "Tim Kaine", "K000384", "Senate", "VA"),
    ("ChrisMurphyCT", "Christopher Murphy", "M001169", "Senate", "CT"),
    ("ossoff", "Jon Ossoff", "O000174", "Senate", "GA"),
    ("RandPaul", "Rand Paul", "P000603", "Senate", "KY"),
    ("marcorubio", "Marco Rubio", "R000595", "Senate", "FL"),
    ("CongresswomanSC", "Nancy Mace", "M000194", "House", "SC"),
    ("repdelgado", "Antonio Delgado", "D000630", "House", "NY"),
    ("Lancegooden", "Lance Gooden", "G000589", "House", "TX"),
    ("claudiatenney", "Claudia Tenney", "T000478", "House", "NY"),
    ("michaelgwaltz", "Michael Waltz", "W000823", "House", "FL"),
]
APPLY_CORRECTIONS = True
CORRECTIONS = {
    "CongresswomanSC": {
        "name": "Sheila Cherfilus-McCormick", "bioguide": "C001127", "chamber": "House", "state": "FL",
        "reason": "plan appendix maps CongresswomanSC to M000194 (Nancy Mace), but M000194 is already matched "
                  "automatically via RepNancyMace, also in the network (bioguide would appear twice). "
                  "CongresswomanSC is Sheila Cherfilus-McCormick (C001127, House FL-20, from 2022-01-18; 'SC' = "
                  "her initials); she has no Twitter entry in the 2022-05 social table and the account's "
                  "network neighbours are Democrats, mostly from Florida (see evidence)."},
}
CHAMBER = {"rep": "House", "sen": "Senate"}


def load_yaml(path):
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def person_row(p: dict) -> dict:
    t = p["terms"][-1]
    nm = p["name"]
    return {"name": nm.get("official_full") or f"{nm.get('first', '')} {nm.get('last', '')}".strip(),
            "last": nm.get("last", ""), "chamber": CHAMBER.get(t["type"], t["type"]), "state": t["state"],
            "party": t.get("party"), "terms": p["terms"]}


def serves_in(terms, lo: str, hi: str) -> bool:
    return any(t["start"] <= hi and t["end"] >= lo for t in terms)


def build(current, social, historical, network) -> tuple[list[dict], dict]:
    net = network[0]
    names = net["usernameList"]
    people = {}
    for src, rows in (("current", current), ("historical", historical)):
        for p in rows:
            b = p["id"].get("bioguide")
            if b and b not in people:
                people[b] = (src, p)
    h2b = collections.defaultdict(set)
    for s in social:
        tw = (s.get("social") or {}).get("twitter")
        if tw:
            h2b[tw.lower()].add(s["id"]["bioguide"])
    ambiguous = {h: sorted(b) for h, b in h2b.items() if len(b) > 1}
    if ambiguous:
        raise SystemExit(f"handles registered for several legislators: {ambiguous}")

    rows, problems = [], []
    manual = {h: (n, b, c, s) for h, n, b, c, s in MANUAL_PLAN}
    for node, handle in enumerate(names):
        hit = h2b.get(handle.lower())
        if hit:
            b = next(iter(hit))
            src, p = people[b]
            pr = person_row(p)
            rows.append({"handle": handle, "node": node, "bioguide": b, "name": pr["name"], "chamber": pr["chamber"],
                         "state": pr["state"], "method": "auto", "note": ""})
        elif handle in manual:
            n, b, c, s = manual[handle]
            rows.append({"handle": handle, "node": node, "bioguide": b, "name": n, "chamber": c, "state": s,
                         "method": "manual", "note": "plan appendix"})
        else:
            problems.append(f"{handle}: neither matched automatically nor in the manual table")
    # the appendix as written
    dup_plan = [b for b, c in collections.Counter(r["bioguide"] for r in rows).items() if c > 1]
    report = {"plan_literal_duplicate_bioguides": {
        b: [r["handle"] for r in rows if r["bioguide"] == b] for b in dup_plan}}
    corrections = []
    if APPLY_CORRECTIONS:
        for r in rows:
            c = CORRECTIONS.get(r["handle"])
            if c and r["method"] == "manual":
                corrections.append({"handle": r["handle"], "plan": {"bioguide": r["bioguide"], "name": r["name"],
                                                                     "chamber": r["chamber"], "state": r["state"]},
                                    "used": {k: c[k] for k in ("bioguide", "name", "chamber", "state")},
                                    "reason": c["reason"]})
                r.update({k: c[k] for k in ("bioguide", "name", "chamber", "state")})
                r["note"] = "corrected from plan appendix (M000194 is RepNancyMace)"
    report["corrections"] = corrections

    # verify manual rows against the YAML
    checks = []
    for r in rows:
        if r["method"] != "manual":
            continue
        ok = r["bioguide"] in people
        msg = []
        if ok:
            src, p = people[r["bioguide"]]
            pr = person_row(p)
            if pr["last"].lower() not in r["name"].lower():
                ok = False
                msg.append(f"last name {pr['last']!r} not in {r['name']!r}")
            if pr["state"] != r["state"]:
                ok = False
                msg.append(f"state {pr['state']} != {r['state']}")
            if pr["chamber"] != r["chamber"]:
                ok = False
                msg.append(f"chamber {pr['chamber']} != {r['chamber']}")
            if not serves_in(p["terms"], *TWEET_PERIOD):
                ok = False
                msg.append("no term overlapping the tweet period")
            checks.append({"handle": r["handle"], "bioguide": r["bioguide"], "yaml_file": src,
                           "yaml_name": pr["name"], "yaml_chamber": pr["chamber"], "yaml_state": pr["state"],
                           "ok": ok, "problems": msg})
        else:
            checks.append({"handle": r["handle"], "bioguide": r["bioguide"], "ok": False,
                           "problems": ["bioguide not in the YAML files"]})
        if not ok:
            problems.append(f"manual row {r['handle']} -> {r['bioguide']}: {'; '.join(msg) or 'not found'}")
    report["manual_checks"] = checks

    # evidence for the correction: party and state of the account's automatically matched neighbours
    ev = {}
    for h in CORRECTIONS:
        if h in names:
            i = names.index(h)
            nb = set(net["outList"][i]) | set(net["inList"][i])
            cnt = collections.Counter()
            for j in nb:
                hit = h2b.get(names[j].lower())
                if hit:
                    pr = person_row(people[next(iter(hit))][1])
                    cnt[f"{pr['state']}-{(pr['party'] or '?')[0]}"] += 1
            ev[h] = {"neighbours": len(nb), "matched_neighbours_by_state_party": dict(cnt.most_common())}
    report["correction_evidence"] = ev

    if len(rows) != len(names):
        problems.append(f"{len(rows)} rows for {len(names)} network nodes")
    dup = [b for b, c in collections.Counter(r["bioguide"] for r in rows).items() if c > 1]
    if dup:
        problems.append(f"bioguide appears more than once: {dup}")
    report["problems"] = problems
    report["counts"] = {"rows": len(rows), "auto": sum(r["method"] == "auto" for r in rows),
                        "manual": sum(r["method"] == "manual" for r in rows),
                        "unique_bioguides": len({r["bioguide"] for r in rows})}
    return rows, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", default=str(E6DATA / "current_2022-06.yaml"))
    ap.add_argument("--social", default=str(E6DATA / "social_2022-06.yaml"))
    ap.add_argument("--historical", default=str(E6DATA / "historical_2022-06.yaml"))
    ap.add_argument("--network", default=str(E6DATA / "CongressionalTwitterNetwork" / "congress_network_data.json"))
    ap.add_argument("--license", default=None, help="network repository LICENSE (default: next to --network)")
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net_sha = sha256_file(args.network)
    if net_sha != NETWORK_SHA256:
        print(f"[crosswalk] network sha256 {net_sha} differs from the plan's {NETWORK_SHA256}", file=sys.stderr)
        return 1
    with open(args.network, encoding="utf-8") as f:
        network = json.load(f)
    current, social, historical = load_yaml(args.current), load_yaml(args.social), load_yaml(args.historical)
    rows, report = build(current, social, historical, network)
    for k, v in report["plan_literal_duplicate_bioguides"].items():
        print(f"[crosswalk] plan appendix as written: {k} appears for {v}")
    for c in report["corrections"]:
        print(f"[crosswalk] correction: {c['handle']} {c['plan']['bioguide']} -> {c['used']['bioguide']} "
              f"({c['used']['name']}); evidence {report['correction_evidence'].get(c['handle'])}")
    if report["problems"]:
        for p in report["problems"]:
            print(f"[crosswalk] PROBLEM: {p}", file=sys.stderr)
        return 1
    cols = ["handle", "node", "bioguide", "name", "chamber", "state", "method", "note"]
    with open(out / "crosswalk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    shutil.copyfile(args.network, out / "congress_network_data.json")
    lic = Path(args.license) if args.license else Path(args.network).parent / "LICENSE.txt"
    files = {}
    if lic.exists():
        shutil.copyfile(lic, out / "LICENSE_CongressionalTwitterNetwork.txt")
        files["LICENSE_CongressionalTwitterNetwork.txt"] = {
            "sha256": sha256_file(out / "LICENSE_CongressionalTwitterNetwork.txt"),
            "source": f"{NETWORK_REPO} commit {NETWORK_COMMIT}, LICENSE.txt"}
    net = network[0]
    files["congress_network_data.json"] = {
        "sha256": sha256_file(out / "congress_network_data.json"),
        "size": (out / "congress_network_data.json").stat().st_size,
        "source": NETWORK_REPO, "commit": NETWORK_COMMIT, "path_in_repo": "congress_network_data.json",
        "license": "MIT (Copyright (c) 2023 Gina Sprint); see LICENSE_CongressionalTwitterNetwork.txt",
        "references": ["Fink et al., A centrality measure for quantifying spread on weighted, directed networks, "
                       "Physica A (2023), doi:10.1016/j.physa.2023.129083",
                       "Fink et al., A Congressional Twitter network dataset quantifying pairwise probability "
                       "of influence, Data in Brief (2023)"],
        "accounts": len(net["usernameList"]), "directed_edges": sum(len(x) for x in net["outList"]),
        "tweet_period": list(TWEET_PERIOD),
        "edge_semantics": "outList[i] = nodes receiving connections FROM i: edge i->j means j retweeted, quoted, "
                          "replied to or mentioned i; weight = count / number of tweets by i"}
    files["crosswalk.csv"] = {
        "sha256": sha256_file(out / "crosswalk.csv"), "size": (out / "crosswalk.csv").stat().st_size,
        "generated_by": "e6/make_crosswalk.py", **report["counts"],
        "sources": {"repo": LEGIS_REPO, "commit": LEGIS_COMMIT, "commit_date": "2022-05-26",
                    "license": "CC0-1.0 (public domain)",
                    "files": {"legislators-current.yaml": sha256_file(args.current),
                              "legislators-social-media.yaml": sha256_file(args.social),
                              "legislators-historical.yaml": sha256_file(args.historical)}},
        "method": "auto = handle equals social.twitter (case-insensitive) in legislators-social-media.yaml; "
                  "manual = plan appendix, checked against the YAML (name, state, chamber, term in the tweet "
                  "period)"}
    write_json({"created": now_iso(), "files": files, "plan_literal_duplicate_bioguides":
                report["plan_literal_duplicate_bioguides"], "corrections": report["corrections"],
                "correction_evidence": report["correction_evidence"], "manual_checks": report["manual_checks"]},
               out / "MANIFEST.json")
    print(f"[crosswalk] {report['counts']} -> {out / 'crosswalk.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
