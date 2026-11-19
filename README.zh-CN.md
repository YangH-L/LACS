# LACS：成员个性化容忍度下的社区搜索

[English](README.md)

本仓库提供以下论文的代码、实验脚本和分析脚本。

> **Community Search under Member-Specific Tolerances**（已投 ICDE 2027）。技术报告见 [arXiv:2611.05951](https://arxiv.org/abs/2611.05951)。

给定查询顶点和参数 *k*，**接受域一致社区**（latitude-consistent community，LCC）是包含该顶点的连通子图，子图的最小度至少为 *k*。在成员所处的位置空间中，社区的直径（任意两名成员之间的最大距离）不能超过社区内最小的成员容忍度。仓库实现了 LatSweep 的 BallCore-F、Greedy、BallCore-R 和精确分支定界算法，以及基线方法、意见动力学模拟和下文列出的论文第 VI 节实验流程。

仓库随附采用 MIT 许可证的美国国会 Twitter 网络（`e6/data/`）。其他数据集需从各自的官方来源获取，详见[数据](#数据)。**本仓库不包含或再分发 Yelp Open Dataset 的任何原始数据**。仓库提供处理脚本，技术报告列出汇总结果。

## 技术报告

[下载完整技术报告（PDF）](full-technical-report/LACS_technical_report.pdf)

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `cpp/` | C++17 搜索引擎。`lacs_search` 实现各搜索算法；`lacs_convert` 将边列表转换为二进制 CSR 格式，支持针对 Friendster 的低内存模式；`lacs_lpa` 运行带规模上限的标签传播，`lacs_project` 将二部图投影为用户之间的图。 |
| `lacs/` | Python 包，入口为 `python -m lacs <command>`。包含图的读写、合成位置与容忍度的生成、搜索引擎接口、效率实验（`e3`）和线下聚会应用（`geo-*`）。 |
| `p0/` | 意见动力学计算内核（C++ 实现，经 ctypes 调用）和实验脚本，用于研究搜索所得群组的动力学、容忍度异质性、求解器表现，以及合成设置下的稳健性。 |
| `e1x/` | 开放群组实验：引入群组外朋友的影响，并静态检验命题 3（开放群组）的条件。 |
| `e5/` | Yelp 数据准备、可行性检查及线下聚会结果分析。 |
| `e6/` | 美国国会实验：下载数据、确定成员立场、根据议员联署法案时与提案人的立场距离估计容忍度、检验研究前提、统计违反容忍度约束的情况并生成报告。 |
| `e12/` | 欧洲议会实验：根据唱名表决估计立场，根据转推估计容忍度，并检验研究前提。 |
| `p2/` | 容忍度缩放（`e11.py`）、容忍度噪声实验，以及用当前引擎重跑合成设置（`e4v05.py`）。 |
| `p2x/` | Friendster 上的 k-core 参考计时，对应表 II 的 Friendster 行、k-core 列。 |
| `analysis/p2/` | 分析无 LCC 的查询、开放群组的群组级结果、容忍度缩放及合成设置的重跑结果。 |
| `analysis/paper/` | 生成部分实验图表，涵盖效率、求解器、动力学、线下聚会和美国国会实验。 |
| `configs/` | 实验配置。 |
| `scripts/` | 实验运行脚本。各脚本按阶段运行，支持中断后续跑，并报告失败的阶段。 |
| `tests/`、`*/tests/` | 单元测试与自检。 |
| `docs/PLANS.md` | 实验代码引用的分析计划索引。 |

`lacs/model`、`lacs/features`、`configs/debagreement.yaml`、`configs/cmv.yaml`、`scripts/run_cmv.sh` 和 `scripts/run_debagreement.sh` 属于早期基于文本的项目版本，本论文的实验未使用这些模块。

## 运行环境

- Linux、Python ≥ 3.10、g++ ≥ 9（C++17）。CMake 为可选工具。
- 运行 `pip install -r requirements.txt` 安装 NumPy、SciPy、pandas、scikit-learn、NetworkX、PyYAML、Matplotlib 和 joblib。
- Friendster 实验需要 64 GB 内存，其中图本身约占 15 GB；其他实验使用 16 GB 即可。
- 所有实验均在 CPU 上运行，每个引擎进程使用单线程。计时时设置 `OMP_NUM_THREADS=1`，其他 BLAS 线程数变量也设为 1。

构建并运行测试：

```bash
python -m lacs build-cpp                 # 构建 cpp/build/{lacs_search,lacs_convert,lacs_lpa,lacs_project}
bash p0/build.sh                         # 构建 p0/p0kit/_lib/libp0.so
bash scripts/run_tests.sh                # 运行 lacs/ 的单元测试
```

分阶段运行的脚本还提供 `selftest`。例如，`STAGES="build selftest" bash scripts/run_p0.sh` 会在运行实验前，用小规模实例的穷举结果或定理结论核对计算内核。

## 数据

请按下表放置原始文件。表中路径与配置文件一致，`data/` 和 `runs*/` 目录不纳入 Git 版本管理。

| 数据集 | 来源与获取方式 | 存放位置 |
| --- | --- | --- |
| Gowalla、Brightkite（好友关系与签到） | 从 <https://snap.stanford.edu/data/> 下载 `loc-gowalla_edges.txt.gz`、`loc-gowalla_totalCheckins.txt.gz`、`loc-brightkite_edges.txt.gz` 和 `loc-brightkite_totalCheckins.txt.gz`。 | `data/raw/snap/` |
| LiveJournal、Orkut | 从 SNAP 下载 `com-lj.ungraph.txt.gz` 和 `com-orkut.ungraph.txt.gz`。 | 解压到 `data/raw/snap/` |
| DBLP、YouTube、Pokec、Friendster | 运行 `bash scripts/fetch_snap_w2.sh` 可从 SNAP 下载并解压。设置 `FRIENDSTER=0` 可跳过 31 GB 的 Friendster 文件。 | `data/raw/snap/` |
| Yelp Open Dataset（2022 年版） | 按 Yelp 数据集使用协议，从 <https://www.yelp.com/dataset> 自行下载 `Yelp-JSON.zip`，不得再分发。 | `data/yelp/Yelp-JSON.zip` |
| 美国第 117 届国会（议员、法案、Twitter 网络） | 运行 `python -m e6.fetch --data data/e6` 下载 Voteview 议员数据和 GovInfo BILLSTATUS 法案数据。Fink 等人整理的国会 Twitter 网络及其 MIT 许可证已随附于 `e6/data/`。 | `data/e6/` |
| 欧洲议会（MEP 1.0，Cherepnalkoski 等） | 从 CLARIN.SI（<http://hdl.handle.net/11356/1071>）获取 `Readme.txt`、`mep_info.json`、`rcv_info.json`、`action_info.csv`、`mep_rcv_vote.csv` 和 `retweets.csv`。 | 任意目录，运行时通过 `DATA=` 指定 |

## 复现论文实验

以下命令均从仓库根目录运行。每个脚本开头的注释列出了步骤与选项。支持 `STAGES` 的脚本（`run_w2.sh`、`run_p0.sh`、`run_e5_feas.sh`、`run_e5_main.sh`、`run_p2.sh`）中断后可重新运行以继续，并会在结束时列出失败的阶段。

| 论文中的结果 | 命令 | 主要输出 |
| --- | --- | --- |
| Gowalla 和 Brightkite 的线下聚会实验（第 VI-D 节）。为图 3 及技术报告表 XV 中这两个数据集的列提供输入。 | 先运行 `bash scripts/run_round3.sh`（准备数据、检查条件，并在 LiveJournal 和 Orkut 上运行 E3），再运行 `bash scripts/run_round4.sh`（覆盖所有符合条件的查询，并扫描 k 和容忍度）。 | `runs/{gowalla,brightkite}/`、`runs/efficiency_round3/` |
| 表 II（效率）、第 VI-B 节的可扩展性与消融实验。技术报告表 XIV 列出未能证明最优性的查询。 | `bash scripts/run_w2.sh`。A 组为 Gowalla、Brightkite、LiveJournal、Orkut，B 组为 DBLP、YouTube、Pokec、Friendster。 | `runs/efficiency_w2*/results/` |
| 表 II 的 Friendster 行、k-core 列 | B 组运行结束后执行 `python p2x/fkcore.py run`。脚本先在 Pokec 上核对，再对相同的 200 个 Friendster 查询测量 k-core 参考用时。换机器时设置 `P2X_SKIP_TIMING=1`，保留规模检查，跳过用时范围检查。 | `runs_p2/p2x/p2x_summary.json` |
| 表 III（求解器比较）、表 IV（搜索所得群组的动力学），以及容忍度异质性实验（第 VI-B 节、技术报告表 X）。 | `bash scripts/run_p0.sh` | `runs_p0/` |
| 合成设置下的稳健性（第 VI-B 节、技术报告表 XI） | `STAGES="env build selftest p25 bundle" bash scripts/run_p2.sh` | `runs_p2/e4v05/` |
| 开放群组（第 VI-C 节）。Gowalla 和 Brightkite 的输出用于技术报告表 XIX。 | `STAGES="e1sweep" bash scripts/run_e5_main.sh`，使用 `run_p0.sh` 导出的群组。 | `runs/e1_open/` |
| Yelp 线下聚会实验，以 2019 年为测试期（第 VI-D 节）。结果见图 3 和技术报告表 XV 的 Yelp 列。 | 先运行 `bash scripts/run_e5_feas.sh`，再运行 `STAGES="env build selftest e5prep e5gates e5pilot e5main e5sweep e5analysis" bash scripts/run_e5_main.sh`。 | `runs/e5_main/results/` |
| 容忍度缩放（第 VI-D 节、技术报告附录 A-J 表 XVIII）及容忍度噪声检查（附录 A-J）。技术报告表 XVII 中无 LCC 查询的分类见下表的 `p21_unsolvable.py`。 | `STAGES="env build selftest p24pilot p24gate p24 p24b p24sum bundle" bash scripts/run_p2.sh` | `runs_p2/e11/`、`runs_p2/p24b/` |
| 美国国会（技术报告第 VI-E 节、表 V）。这些阶段为图 4 准备输入，绘图脚本为 `p23_case.py`。 | `STAGES="env build selftest e6fetch e6build e6geo e6dyn e6premise e6report" bash scripts/run_e5_main.sh` | `runs/e6/` |
| 欧洲议会（技术报告第 VI-E 节） | 先运行 `python3 -I analysis/p2/p26_data_gate.py /path/to/MEP_files` 检查数据，再运行 `DATA=/path/to/MEP_files RUN=runs/e12 bash e12/run_e12.sh`。 | `runs/e12/results/` |

`bash analysis/make_workdir.sh` 会在 `work/` 下创建指向各结果目录的链接。脚本开头的注释列出了分析命令示例。

| 分析脚本 | 对应的论文结果 |
| --- | --- |
| `analysis/paper/w2_numbers.py`、`make_tables.py` | `w2_numbers.py` 汇总第 VI-B 节的 W2 效率实验（表 II）。`make_tables.py` 根据各实验的结果文件生成技术报告的表 II 至表 V 和表 XV。 |
| `analysis/paper/p0_e3_numbers.py`、`p0_e1_checks.py` | 前者计算表 III 的求解器统计量，后者逐组核对第 VI-C 节的动力学结果。`make_tables.py` 使用 `e1_summary.csv` 生成表 IV。 |
| `analysis/paper/geo_size_check.py`、`geo_posthoc_check.py` | 两个脚本为 Gowalla 和 Brightkite 输出 JSON，给出三类差异的估计值和自助法置信区间：规模相同但成员不同的群组之间的共同到访率差异、按群组规模校正的成员对共同到访率差异，以及校正规模后有无 LCC 的查询之间 SAC 成员对共同到访率的差异。其中的规模对照结果用于两版论文的图 3 和技术报告表 XV。 |
| `analysis/paper/make_fig_meetup.py` | 生成两版论文的图 3，读取与 `make_tables.py` 位于同一目录的 `meetup_values.json`。图中估计值列于技术报告表 XV。 |
| `analysis/paper/p23_case.py` | 生成技术报告图 4（美国国会案例）。 |
| `analysis/p2/p21_unsolvable.py`、`p22_open_groups.py`、`p24_analysis.py`、`p25_analysis.py` | 依次对应技术报告的无 LCC 查询（表 XVII）、开放群组的群组级结果（表 XIX）、容忍度缩放（表 XVIII）、合成设置下的稳健性（表 XI）。 |

## 复现说明

- **随机性。** 对每个数据集，从配置中的种子派生独立的随机流，用于抽取查询、生成合成位置和容忍度。因此，无论同时运行哪些其他数据集，重跑同一数据集时均得到相同的查询。
- **计时。** 论文中的测量使用 Intel Xeon Gold 6530 和 64 GB 内存，每次测量运行一个引擎进程。耗时取决于机器。Friendster 的 k-core 参考用时单独测量；在同一台机器上复核 Pokec 时，耗时为原测量的 1.20 倍。
- **分析计划。** `docs/PLANS.md` 列出实验代码引用的分析计划及其章节标识。
- **源文件校验。** `scripts/run_p2.sh` 的 `env` 阶段运行 `python -I p2/manifest.py check`，校验 `p2/MANIFEST_SOURCES.sha256` 中列出的文件。
- **Yelp。** E5 主实验的结果包包含 `runs/e5_main/results/`。Yelp 原始数据和派生的 `geo/`、`graph/`、`work/` 目录保留在运行分析的机器上。

## 许可证

本仓库采用 MIT 许可证（见 `LICENSE`）。`e6/data/` 中的美国国会 Twitter 网络使用其自身的 MIT 许可证（见 `e6/data/LICENSE_CongressionalTwitterNetwork.txt`）。

## 引用

```bibtex
@misc{liu2026community,
  title         = {Community Search under Member-Specific Tolerances},
  author        = {Yanghao Liu and Fangda Guo and Yuxuan Zhang and Xueqi Cheng},
  year          = {2026},
  eprint        = {2611.05951},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  url           = {https://arxiv.org/abs/2611.05951}
}
```

会议版录用后，我们将更新此条目。
