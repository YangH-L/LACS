#!/usr/bin/env python3
"""E6 report: E6_REPORT.md + e6_report.json from everything that exists.

  python -m e6.report --run runs/e6 [--results runs/e6/results] [--data data/e6]

Reads (each part is optional; a missing part is reported as missing, not as a failure):
  RUN/public/e6_build.json, RUN/geo/meta.json            data counts, data gate, crosswalk (build)
  RESULTS/e6_structure.json                              plan section 4 (e6.structure)
  RESULTS/e6_premise.json                                plan section 5, Q1 (e6.premise)
  RESULTS/e6_static.json                                 plan section 6, Q2 (e6.static)
  RESULTS/e1x_summary.csv                                plan section 7, Q3 (written by the dynamics code;
      columns dataset, method, variant, w, groups, share_broken, ci_lo, ci_hi, share_cond_special,
      share_cond_refined, theorem_violations)
  RESULTS/geo_e2.json (and RESULTS/<subdir>/geo_e2.json) plan section 8, Q4 (lacs geo-e2): its summary
      and paired differences are printed as they are
The reading rules of the plan (section 4 structure rule, section 5 verdict, section 7 theorem check)
are restated next to the numbers; nothing is re-interpreted here.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import DATA_DIR, now_iso, read_json, setup_logging, write_json  # noqa: E402

LOG = logging.getLogger("e6.report")
MISSING = "缺失（尚未运行或文件不存在）"
Q3_COLS = ["dataset", "method", "variant", "w", "groups", "share_broken", "ci_lo", "ci_hi", "share_cond_special",
           "share_cond_refined", "theorem_violations"]


def _num(x, d: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "–"
        if x != 0 and abs(x) < 10 ** (-d):
            return f"{x:.2e}"
        return f"{x:.{d}f}"
    return str(x)


def _ci(c, d: int = 4) -> str:
    return f"[{_num(c[0], d)}, {_num(c[1], d)}]" if c else "–"


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(_num(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return out


def _load(path: Path):
    try:
        return read_json(path) if path.exists() else None
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning("%s unreadable: %s", path, e)
        return None


# ---------------------------------------------------------------- sections --

def _fetch_line(fm: dict) -> str:
    tot = fm.get("totals") or {}
    per = "；".join(f"{t} 目录列出 XML {r.get('listed_xml')}、压缩包 {r.get('listed_zip')}，取得 {r.get('obtained_xml')}，"
                   f"失败 {len(r.get('failures') or [])}" for t, r in (fm.get("billstatus") or {}).items())
    return (f"- 下载清单 fetch_manifest.json（{fm.get('mode')}，{fm.get('finished')}）：完整 = {_num(fm.get('complete'))}；"
            f"{per}；合计取得 {tot.get('obtained_xml')}；Voteview `{(fm.get('voteview') or {}).get('url')}` "
            f"sha256 `{(fm.get('voteview') or {}).get('sha256')}`")


def sec_data(build, meta, L: list[str], J: dict, fman: dict | None = None) -> None:
    L.append("## 1. 数据与关卡（第 2–3 节）")
    if fman is not None:
        J["fetch_manifest"] = {k: fman.get(k) for k in ("mode", "complete", "finished", "totals",
                                                        "listing_unverified_types")}
    if build is None:
        L += [""] + ([_fetch_line(fman)] if fman else []) + ["", MISSING, ""]
        J["data"] = None
        return
    gate = build.get("gate", {})
    mem = build.get("members", {})
    inp = build.get("inputs", {})
    bs = build.get("billstatus", {})
    vv = inp.get("voteview", {})
    net = inp.get("network", {})
    L.append("")
    L.append(f"- **数据关卡**：可用议员 {gate.get('eligible')} 人，要求 ≥ {gate.get('required')}："
             f"{'通过' if gate.get('pass') else '**未通过 —— 按第 2 节停下汇报，后续分析不做**'}")
    fetch = build.get("fetch")
    if fman is not None:
        L.append(_fetch_line(fman))
        if fman.get("listing_unverified_types"):
            L.append(f"  - 无目录清单、完整性未核对的类型：{fman['listing_unverified_types']}")
    elif fetch:
        per = "；".join(f"{t} 取得 {r.get('obtained_xml')}（目录列出 XML {r.get('listed_xml')}、压缩包 "
                       f"{r.get('listed_zip')}；失败 {r.get('failures')}）"
                       for t, r in (fetch.get("billstatus") or {}).items())
        L.append(f"- 下载（e6.fetch，{fetch.get('mode')}）：完整 = {_num(fetch.get('complete'))}；{per}；"
                 f"Voteview sha256 `{(fetch.get('voteview') or {}).get('sha256')}`")
        if fetch.get("listing_unverified_types"):
            L.append(f"  - 无目录清单、完整性未核对的类型：{fetch['listing_unverified_types']}")
    else:
        L.append("- 下载清单（fetch_manifest.json）不存在：完整性未核对")
    L.append(f"- Voteview：{vv.get('rows')} 行，含 dim1 的议员 {vv.get('bioguides_with_dim1')} 人"
             f"（多行议员 {vv.get('bioguides_with_several_rows')} 人，取投票数最多的众/参院行）")
    L.append(f"- BILLSTATUS：{bs.get('files')} 个文件、{bs.get('bills')} 项议案 {bs.get('bills_by_type')}；"
             f"事件 {bs.get('events')}（{bs.get('events_by_role')}），撤回联署 {bs.get('cosponsorships_withdrawn')}，"
             f"解析失败 {bs.get('n_parse_errors')}，无提案人议案 {bs.get('bills_without_sponsor')}")
    L.append(f"- 事件按时期：{bs.get('events_by_period')}；时期外 {bs.get('events_outside_periods')}")
    L.append(f"- Twitter 网络：{net.get('accounts')} 个账号、{net.get('directed_edges')} 条有向边；"
             f"议员之间：{(build.get('graph') or {}).get('directed_edges')} 条有向边，"
             f"无向图 {(build.get('graph') or {}).get('undirected_edges')} 条边")
    L.append(f"- 议员：账号 {mem.get('network_accounts')}，对上 Voteview（有 dim1）{mem.get('matched_voteview_dim1')}，"
             f"可用 {mem.get('eligible')}；原因 {mem.get('reasons')}；党团 × 院 {mem.get('eligible_by_caucus_chamber')}")
    med = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (mem.get("theta_median") or {}).items()}
    L.append(f"- 接受域中位数（可用议员）：{med}")
    L.append(f"- 触及下限 0.02 的人数：{mem.get('floor_hits')}")
    if build.get("joins"):
        jn = build["joins"]
        L.append(f"- 议员加入议案（不含撤回）：{jn.get('member_joins_by_period')}；测试期活跃议员 "
                 f"{jn.get('test_active_members')}；好友对共同加入 {jn.get('covisit_friend_pairs')}")
        nt = jn.get("test_notop") or {}
        L.append(f"- 稳健性版本（去掉测试期加入人数最多的 1% 议案）：阈值 {nt.get('threshold')} 人"
                 f"（99 分位，inverted_cdf），去掉 {nt.get('bills_removed')} / {nt.get('bills_with_joins')} 项议案、"
                 f"{nt.get('joins_removed')} 次加入")
    L.append("")
    J["data"] = {"gate": gate, "members": mem, "billstatus": {k: bs.get(k) for k in
                                                              ("files", "bills", "bills_by_type", "events",
                                                               "events_by_role", "cosponsorships_withdrawn",
                                                               "n_parse_errors", "events_by_period")},
                 "voteview": {k: vv.get(k) for k in ("sha256", "rows", "bioguides_with_dim1")},
                 "network": net, "graph": build.get("graph"), "joins": build.get("joins"), "fetch": fetch,
                 "meta": meta}


def sec_crosswalk(build, manifest, L: list[str], J: dict) -> None:
    L.append("## 2. 账号对照")
    cw = ((build or {}).get("inputs") or {}).get("crosswalk")
    files = (manifest or {}).get("files", {})
    shipped = files.get("crosswalk.csv", {})
    if cw is None and not shipped:
        L += ["", MISSING, ""]
        J["crosswalk"] = None
        return
    info = cw or shipped
    # the MANIFEST (corrections, evidence) describes the shipped crosswalk only
    same = cw is None or (shipped.get("sha256") is not None and shipped.get("sha256") == cw.get("sha256"))
    L.append("")
    L.append(f"- 对照表 {info.get('rows')} 行：自动 {info.get('auto')}、人工 {info.get('manual')}；"
             f"sha256 `{info.get('sha256')}`" + ("" if same else "（不是随包附带的对照表）"))
    corrections = (manifest or {}).get("corrections", []) if same else []
    for c in corrections:
        L.append(f"- **对附录的更正**：{c['handle']}：附录 {c['plan']['bioguide']}（{c['plan']['name']}）→ "
                 f"{c['used']['bioguide']}（{c['used']['name']}，{c['used']['chamber']} {c['used']['state']}）。"
                 f"理由：{c['reason']}")
    ev = (manifest or {}).get("correction_evidence") if same else None
    if ev:
        L.append(f"- 更正的证据（该账号已对上的网络邻居，州-党）：{json.dumps(ev, ensure_ascii=False)}")
    for nt in (cw or {}).get("notes") or []:
        if not corrections:
            L.append(f"- 对照表注记：{nt}")
    L.append("")
    J["crosswalk"] = {"counts": {k: info.get(k) for k in ("rows", "auto", "manual", "unique_bioguides")},
                      "shipped": same, "corrections": corrections, "evidence": ev}


def sec_structure(st, L: list[str], J: dict) -> bool | None:
    L.append("## 3. 结构检查（第 4 节）")
    if st is None:
        L += ["", MISSING, ""]
        J["structure"] = None
        return None
    fp = st.get("friend_pairs", {})
    lcc = st.get("lcc", {})
    L.append("")
    L.append(f"- 好友对（图的边）{fp.get('edges')} 对，互相在接受域内（{st.get('tolerance')}）的比例："
             f"{_num(fp.get('share_mutual_within'))}（仅一方 {_num(fp.get('share_one_side_only'))}，"
             f"双方都不 {_num(fp.get('share_neither'))}）")
    flag = None
    if "missing" in lcc:
        L.append(f"- LCC 部分：{MISSING}（{lcc['missing']}）")
    else:
        L.append(f"- 查询 {lcc.get('queries')} 个，LCC 非空比例 {_num(lcc.get('share_lcc_nonempty'))}"
                 f"（状态 {lcc.get('lcc_status_counts')}）")
        r = lcc.get("ratio") or {}
        if "missing" in r:
            L.append(f"- LCC / k-core：{MISSING}（{r['missing']}）")
        else:
            flag = lcc.get("no_filtering")
            L.append(f"- 非空查询上 LCC 规模 / 所在 k-core 连通块规模：中位数 {_num(r.get('median'))}"
                     f"（{r.get('queries')} 个查询；四分位 {_ci(r.get('q25_q75'), 3)}；"
                     f"LCC 中位规模 {_num(r.get('lcc_size_median'), 1)}，k-core 中位规模 {_num(r.get('kcore_size_median'), 1)}）")
            if flag:
                L.append("- **规则触发（中位数 ≥ 0.9）**：接受域起不到筛选作用；Q2–Q4 只作描述，并如实写成结构性观察。")
            elif flag is not None:
                L.append("- 规则（中位数 ≥ 0.9）未触发：接受域有筛选作用。")
    L.append("")
    J["structure"] = {"friend_pairs": fp, "lcc": lcc, "no_filtering": flag}
    return flag


def sec_q1(pr, L: list[str], J: dict) -> str | None:
    L.append("## 4. Q1：前提检验（第 5 节，主假设）")
    if pr is None:
        L += ["", MISSING, ""]
        J["q1"] = None
        return None
    L.append("")
    L.append(f"- 样本：{pr.get('n_members')} 名议员的 {pr.get('n_dyads')} 个有序对，y = 1 的 {pr.get('n_positive')} 个"
             f"（{_num(pr.get('positive_rate'), 4)}）；固定效应 L2 罚 λ = {pr.get('lambda')}；"
             f"置换 {pr.get('n_perm_primary')} 次（次要变体 {pr.get('n_perm_secondary')} 次），5 折留出，"
             f"bootstrap {pr.get('n_boot')} 次")
    L.append("")
    rows = []
    for name, v in (pr.get("variants") or {}).items():
        if "skipped" in v:
            rows.append([name, "–", "–", "–", "–", "–", v["skipped"]])
            continue
        p = v.get("perm") or {}
        o = v.get("oof") or {}
        rows.append([name, v.get("term"), _num(v.get("coef"), 4),
                     f"{_num(p.get('p_one_sided'), 4)}（{p.get('direction')}，{p.get('n')} 次）",
                     _num(o.get("mean_diff"), 5), _ci(o.get("ci95"), 5), ""])
    L += _table(["变体", "项", "系数", "置换 p（单侧）", "留出对数损失 M1−M0", "95% CI（按发送者聚类）", "备注"], rows)
    ml = pr.get("member_level") or {}
    L.append("")
    if "spearman" in ml:
        L.append(f"- 成员级：θ_u 与 r_u 的 Spearman = {_num(ml['spearman'])} {_ci(ml.get('spearman_ci95'), 3)}，"
                 f"控制党团 × 院与 m_u 后的偏相关 = {_num(ml.get('partial'))} {_ci(ml.get('partial_ci95'), 3)}"
                 f"（{ml.get('n_members')} 人，至少 {ml.get('min_partners')} 个互动对象）")
    else:
        L.append(f"- 成员级：{ml.get('note', MISSING)}")
    vd = pr.get("verdict") or {}
    L.append(f"- **判读（主变体 primary_1d_q50_indicator）**：{vd.get('text')}"
             f"（γ = {_num(vd.get('gamma'), 4)}，p = {_num(vd.get('p_perm'), 4)}，留出之差 CI {_ci(vd.get('oof_diff_ci95'), 5)}）")
    if vd.get("note"):
        L.append(f"  - {vd['note']}")
    pc = vd.get("position_check")
    if pc:
        L.append(f"  - 位置检查（10-25 事先写定的 posadj_1d_q50_indicator）：{pc.get('text')}"
                 f"（γ = {_num(pc.get('coef'), 4)}，p = {_num(pc.get('p_one_sided'), 4)}）")
    L.append("")
    J["q1"] = {"variants": pr.get("variants"), "member_level": ml, "verdict": vd}
    return vd.get("code")


def sec_q2(sc, L: list[str], J: dict) -> None:
    L.append("## 5. Q2：静态越界（第 6 节，描述）")
    if sc is None:
        L += ["", MISSING, ""]
        J["q2"] = None
        return
    L.append("")
    for b in sc.get("results", []):
        L.append(f"- 数据集 {b.get('dataset')}，设置 {b.get('setting')}，k = {b.get('k')}：查询 {b.get('queries')} 个，"
                 f"LCC 非空 {b.get('queries_lcc_nonempty')} 个")
        for scope, title in (("all", "全部查询"), ("lcc_nonempty", "LCC 非空的查询")):
            L.append("")
            L.append(f"{title}：")
            L.append("")
            rows = []
            for m, s in b.get("methods", {}).items():
                x = s.get(scope) or {}
                rows.append([m, x.get("groups"), _num(x.get("share_iii")), _num(x.get("share_members_ecc_gt_theta")),
                             _num(x.get("share_pairs_rejected")), _num(x.get("mean_group_share_members")),
                             _num(x.get("mean_group_share_pairs")), x.get("groups_without_members")])
            L += _table(["方法", "群数（≥2 人）", "满足 (iii)", "成员离心距离 > θ_u", "至少一方拒绝的成员对",
                         "（每群平均）成员", "（每群平均）成员对", "无成员记录"], rows)
        L.append("")
    J["q2"] = sc.get("results")


def sec_q3(path: Path, L: list[str], J: dict) -> None:
    L.append("## 6. Q3：动力学（第 7 节）")
    if not path.exists():
        L += ["", MISSING + f"：{path.name}", ""]
        J["q3"] = None
        return
    df = pd.read_csv(path)
    miss = [c for c in Q3_COLS if c not in df.columns]
    L.append("")
    if miss:
        L.append(f"- 注意：{path.name} 缺少列 {miss}")
    viol = pd.to_numeric(df.get("theorem_violations", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nviol = int(viol.sum())
    if nviol > 0:
        L.append(f"- **定理核对失败：出现 {nviol} 次不应出现的交互丢失。按第 7 节这是代码或数值问题：停下汇报，不解读 Q3 的结果。**")
    else:
        L.append("- 定理核对：0 次违反（封闭 LCC 群与满足 (a)+(b) 的群在所有变体下均未失去交互）")
    L.append("")
    cols = [c for c in Q3_COLS if c in df.columns] + [c for c in df.columns if c not in Q3_COLS]
    L += _table(cols, df[cols].astype(object).where(df[cols].notna(), None).values.tolist())
    L.append("")
    J["q3"] = {"rows": df.to_dict(orient="records"), "theorem_violations": nviol}


def _q4_block(name: str, g: dict, L: list[str]) -> dict:
    L.append(f"### {name}")
    L.append("")
    L.append(f"- k = {g.get('k')}，接受域 {g.get('tolerance')}，查询 {g.get('queries')}（合格 {g.get('eligible')}，"
             f"{g.get('query_mode')}）")
    summ = g.get("summary") or []
    if summ:
        cols = []
        for r in summ:
            for c in r:
                if c not in cols:
                    cols.append(c)
        L.append("")
        L += _table(cols, [[r.get(c) for c in cols] for r in summ])
    paired = {k: v for k, v in g.items() if k.startswith("paired_")}
    for key, block in paired.items():
        if not block:
            continue
        L.append("")
        L.append(f"{key}：")
        L.append("")
        rows = []
        for m, v in block.items():
            if isinstance(v, dict) and "mean_diff" in v:
                rows.append([m, v.get("n"), _num(v.get("mean_diff"), 5), _ci(v.get("ci95"), 5), _num(v.get("p_boot"), 4)])
            elif isinstance(v, dict):  # nested: metric -> method -> stats
                for m2, v2 in v.items():
                    if isinstance(v2, dict):
                        rows.append([f"{m} / {m2}", v2.get("n"), _num(v2.get("mean_diff"), 5), _ci(v2.get("ci95"), 5),
                                     _num(v2.get("p_boot"), 4)])
        L += _table(["方法", "n", "差（LCC − 方法）", "95% CI", "p"], rows)
    L.append("")
    return {"summary": summ, **paired}


def sec_q4(res: Path, L: list[str], J: dict) -> None:
    L.append("## 7. Q4：应用（第 8 节，次要）")
    files = ([res / "geo_e2.json"] if (res / "geo_e2.json").exists() else []) + \
        sorted(p for p in res.glob("*/geo_e2.json"))
    if not files:
        L += ["", MISSING + "：geo_e2.json", ""]
        J["q4"] = None
        return
    L.append("")
    L.append("（Q4 只报告数字。）")
    L.append("")
    J["q4"] = {}
    for p in files:
        g = _load(p)
        if g is None:
            continue
        name = "geo_e2.json" if p.parent == res else f"{p.parent.name}/geo_e2.json"
        J["q4"][name] = _q4_block(name, g, L)


def run(args) -> int:
    run_dir = Path(args.run)
    res = Path(args.results) if args.results else run_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    setup_logging(res / "e6_report.log")
    build = _load(run_dir / "public" / "e6_build.json")
    meta = _load(run_dir / "geo" / "meta.json")
    manifest = _load(Path(args.manifest)) if args.manifest else _load(DATA_DIR / "MANIFEST.json")
    L = ["# E6 报告：国会 Twitter 网络 + DW-NOMINATE", "",
         f"生成：{now_iso()}；运行目录 `{run_dir}`；方案 lacs_e6_plan_20261025（docs/PLANS.md）（预注册，第 4–6 节由 e6 包实现，"
         "第 7–8 节的结果由其他代码写入并在此列出）。", ""]
    J = {"created": now_iso(), "run": str(run_dir), "results": str(res)}
    fman = _load(Path(args.data) / "fetch_manifest.json") if args.data else None
    sec_data(build, meta, L, J, fman)
    sec_crosswalk(build, manifest, L, J)
    flag = sec_structure(_load(res / "e6_structure.json"), L, J)
    verdict = sec_q1(_load(res / "e6_premise.json"), L, J)
    if flag:
        L.append("> 结构检查规则已触发：以下 Q2–Q4 只作描述（结构性观察）。")
        L.append("")
    if verdict == "no_effect":
        L.append("> Q1 判读为“没有可测的作用”：Q2–Q4 照常报告，但正文写明这一接受域估计在 Twitter 上没有得到验证。")
        L.append("")
    sec_q2(_load(res / "e6_static.json"), L, J)
    sec_q3(res / "e1x_summary.csv", L, J)
    sec_q4(res, L, J)
    L.append("## 8. 写作限制（第 9 节）")
    L.append("")
    L.append("- 正文主张限于第 5–8 节的判读；位置来自唱票，接受域来自联署行为，不是测得的态度；")
    L.append("- 国会有很强的政党纪律，结论只代表这一情境。")
    L.append("")
    J["status"] = {k: ("present" if J.get(k) is not None else "missing")
                   for k in ("data", "crosswalk", "structure", "q1", "q2", "q3", "q4")}
    J["flags"] = {"gate_pass": ((build or {}).get("gate") or {}).get("pass"), "structure_no_filtering": flag,
                  "q1_verdict": verdict,
                  "q3_theorem_violations": (J.get("q3") or {}).get("theorem_violations") if J.get("q3") else None}
    (res / "E6_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(J, res / "e6_report.json")
    LOG.info("report: %s; parts %s", res / "E6_REPORT.md", J["status"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", "--geo", dest="run", required=True, help="geo run directory (e.g. runs/e6)")
    ap.add_argument("--results", "--out", dest="results", default=None, help="default: RUN/results")
    ap.add_argument("--data", default=None, help="E6 data directory: adds its fetch_manifest.json summary")
    ap.add_argument("--manifest", default=None, help="crosswalk MANIFEST.json (default: e6/data/MANIFEST.json)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
