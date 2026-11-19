# LACS: Community Search under Member-Specific Tolerances

[简体中文](README.zh-CN.md)

This repository contains the code, experiment drivers and analysis scripts for the paper

> **Community Search under Member-Specific Tolerances** (ICDE 2027 submission). Technical report: [arXiv:2611.05951](https://arxiv.org/abs/2611.05951).

A *latitude-consistent community* (LCC) is a connected subgraph of minimum degree at least *k* that contains a query vertex and whose diameter, in the space of the members' positions, is at most the smallest tolerance of its members. The repository implements LatSweep (BallCore-F, Greedy, BallCore-R and the exact branch and bound), the baselines, the opinion-dynamics simulations and the Section VI experiment workflows listed below.

No dataset is included except the Congressional Twitter network (MIT licence, `e6/data/`). Every other dataset is downloaded from its official source (see [Data](#data)). In particular, **this repository does not contain or redistribute any part of the Yelp Open Dataset**; the processing scripts are included here, and the technical report presents aggregate results.

## Technical report

The [full technical report](full-technical-report/LACS_technical_report.pdf) is available here.

## Layout

| Path | Contents |
| --- | --- |
| `cpp/` | C++17 search engine: `lacs_search` (all search algorithms), `lacs_convert` (edge list to binary CSR, with a low-memory mode for Friendster), `lacs_lpa` (size-capped label propagation), `lacs_project` |
| `lacs/` | Python package (`python -m lacs <command>`): graph I/O, synthetic positions and tolerances, the engine wrapper, the efficiency experiment (`e3`), the meetup application (`geo-*`) |
| `p0/` | Opinion dynamics kernels (C++, via ctypes) and drivers: dynamics on returned groups, heterogeneity, the solver comparison, robustness of the synthetic setting |
| `e1x/` | Open groups: dynamics with friends outside the group, and the static conditions of Proposition 3 (Open groups) |
| `e5/` | Yelp: preparation, feasibility check, analysis of the meetup comparison |
| `e6/` | U.S. Congress: data download, positions, tolerances from co-sponsorship, premise test, static violations, report |
| `e12/` | European Parliament: positions from roll calls, tolerances from retweets, premise test |
| `p2/` | Tolerance scaling (`e11.py`), tolerance noise, and the synthetic-setting rerun with the current engine (`e4v05.py`) |
| `p2x/` | Friendster k-core reference timing (Table II, Friendster row, k-core column) |
| `analysis/p2/` | Analyses of the queries without an LCC, open groups at group level, tolerance scaling and the synthetic-setting rerun |
| `analysis/paper/` | Scripts for selected experimental tables and figures, including the efficiency, solver, dynamics, meetup and Congress results |
| `configs/` | Experiment configurations |
| `scripts/` | Run scripts (each runs its stages in order, resumes after interruption, and reports failed stages) |
| `tests/`, `*/tests/` | Unit tests and self-tests |
| `docs/PLANS.md` | Index of the analysis plans referenced by the code |

Some modules (`lacs/model`, `lacs/features`, `configs/debagreement.yaml`, `configs/cmv.yaml`, `scripts/run_cmv.sh`, `scripts/run_debagreement.sh`) belong to an earlier text-based variant of the project and are not used in the paper.

## Requirements

- Linux, Python ≥ 3.10, g++ ≥ 9 (C++17). CMake is optional.
- `pip install -r requirements.txt` (NumPy, SciPy, pandas, scikit-learn, NetworkX, PyYAML, Matplotlib, joblib).
- Memory: 64 GB for Friendster (the graph alone takes about 15 GB); 16 GB suffice for every other experiment.
- Everything runs on the CPU, single-threaded per engine process. Set `OMP_NUM_THREADS=1` (and the other BLAS thread variables) for timings.

Build and test:

```bash
python -m lacs build-cpp                 # builds cpp/build/{lacs_search,lacs_convert,lacs_lpa,lacs_project}
bash p0/build.sh                         # builds p0/p0kit/_lib/libp0.so
bash scripts/run_tests.sh                # unit tests of lacs/
```

The staged run scripts also have a `selftest` stage (for example `STAGES="build selftest" bash scripts/run_p0.sh`) that checks their kernels against brute force or the theorems on small instances before any experiment runs.

## Data

Place the raw files as below (the paths are those of the configurations; `data/` and `runs*/` are not tracked).

| Dataset | Source | Where |
| --- | --- | --- |
| Gowalla, Brightkite (friendships and check-ins) | SNAP: `loc-gowalla_edges.txt.gz`, `loc-gowalla_totalCheckins.txt.gz`, `loc-brightkite_edges.txt.gz`, `loc-brightkite_totalCheckins.txt.gz` from <https://snap.stanford.edu/data/> | `data/raw/snap/` |
| LiveJournal, Orkut | SNAP: `com-lj.ungraph.txt.gz`, `com-orkut.ungraph.txt.gz` | `data/raw/snap/` (unpacked) |
| DBLP, YouTube, Pokec, Friendster | SNAP; `bash scripts/fetch_snap_w2.sh` downloads and unpacks them (`FRIENDSTER=0` skips the 31 GB Friendster file) | `data/raw/snap/` |
| Yelp Open Dataset (2022 release) | <https://www.yelp.com/dataset>, under Yelp's Dataset User Agreement. Download `Yelp-JSON.zip` yourself; it must not be redistributed | `data/yelp/Yelp-JSON.zip` |
| U.S. Congress (117th): members, bills, Twitter network | `python -m e6.fetch --data data/e6` downloads Voteview members and GovInfo BILLSTATUS; the Congressional Twitter network (Fink et al.) is included in `e6/data/` with its MIT licence | `data/e6/` |
| European Parliament (MEP 1.0, Cherepnalkoski et al.) | CLARIN.SI, <http://hdl.handle.net/11356/1071>: `Readme.txt`, `mep_info.json`, `rcv_info.json`, `action_info.csv`, `mep_rcv_vote.csv`, `retweets.csv` | any directory, passed as `DATA=` |

## Reproducing the paper

Run every command from the repository root. The comments at the top of each script list its steps and options; the scripts with `STAGES` (`run_w2.sh`, `run_p0.sh`, `run_e5_feas.sh`, `run_e5_main.sh`, `run_p2.sh`) resume after an interruption when rerun and report failed stages at the end.

| Paper | Commands | Main outputs |
| --- | --- | --- |
| Gowalla and Brightkite meetups (Section VI-D): inputs for Fig. 3 and the two datasets’ columns in Table XV of the technical report | `bash scripts/run_round3.sh` (preparation, gates, E3 on LiveJournal and Orkut), then `bash scripts/run_round4.sh` (all eligible queries, k and tolerance sweeps) | `runs/{gowalla,brightkite}/`, `runs/efficiency_round3/` |
| Table II (efficiency), scalability and ablation in Section VI-B. Table XIV of the technical report lists the unproven queries | `bash scripts/run_w2.sh` (run A: Gowalla, Brightkite, LiveJournal, Orkut; run B: DBLP, YouTube, Pokec, Friendster) | `runs/efficiency_w2*/results/` |
| Table II, Friendster row, k-core column | `python p2x/fkcore.py run` (after run B; times the k-core reference on the same 200 queries, after a check on Pokec; on another machine set `P2X_SKIP_TIMING=1`, which keeps the size check and skips the time band) | `runs_p2/p2x/p2x_summary.json` |
| Table III (solver comparison), Table IV (dynamics on returned groups), and the heterogeneity sweep (Section VI-B and Table X of the technical report) | `bash scripts/run_p0.sh` | `runs_p0/` |
| Synthetic robustness (Section VI-B and Table XI of the technical report) | `STAGES="env build selftest p25 bundle" bash scripts/run_p2.sh` | `runs_p2/e4v05/` |
| Open groups (Section VI-C): Gowalla and Brightkite inputs for Table XIX of the technical report | `STAGES="e1sweep" bash scripts/run_e5_main.sh` (uses the groups exported by `run_p0.sh`) | `runs/e1_open/` |
| Yelp meetups, 2019 test split (Section VI-D): Yelp results in Fig. 3 and in Table XV of the technical report | `bash scripts/run_e5_feas.sh`, then `STAGES="env build selftest e5prep e5gates e5pilot e5main e5sweep e5analysis" bash scripts/run_e5_main.sh` | `runs/e5_main/results/` |
| Tolerance scaling (Section VI-D and Table XVIII in Appendix A-J of the technical report) and tolerance-noise checks (Appendix A-J). For the empty-query classification in Table XVII of the technical report, see `p21_unsolvable.py` below | `STAGES="env build selftest p24pilot p24gate p24 p24b p24sum bundle" bash scripts/run_p2.sh` | `runs_p2/e11/`, `runs_p2/p24b/` |
| U.S. Congress (Section VI-E and Table V of the technical report). These stages prepare the inputs for Fig. 4, plotted by `p23_case.py` | `STAGES="env build selftest e6fetch e6build e6geo e6dyn e6premise e6report" bash scripts/run_e5_main.sh` | `runs/e6/` |
| European Parliament results (Section VI-E of the technical report) | `python3 -I analysis/p2/p26_data_gate.py /path/to/MEP_files` (data check), then `DATA=/path/to/MEP_files RUN=runs/e12 bash e12/run_e12.sh` | `runs/e12/results/` |

`bash analysis/make_workdir.sh` links the result directories under `work/`. The comments at the top of the script give example analysis commands.

| Script | Paper |
| --- | --- |
| `analysis/paper/w2_numbers.py`, `make_tables.py` | `w2_numbers.py` summarizes the W2 efficiency runs in Section VI-B (Table II). `make_tables.py` writes Tables II–V and XV of the technical report from their respective result files. |
| `analysis/paper/p0_e3_numbers.py`, `p0_e1_checks.py` | Solver statistics for Table III and per-group dynamics checks for Section VI-C. `make_tables.py` generates Table IV from `e1_summary.csv`. |
| `analysis/paper/geo_size_check.py`, `geo_posthoc_check.py` | For Gowalla and Brightkite, the scripts write JSON estimates and bootstrap intervals for co-visit differences between equal-sized groups, size-adjusted pair-rate differences, and SAC pair-rate differences between queries with and without an LCC. The size comparisons contribute to Fig. 3 in both versions and to Table XV of the technical report. |
| `analysis/paper/make_fig_meetup.py` | Fig. 3 in both versions. The script reads `meetup_values.json`, written alongside `make_tables.py`. Table XV of the technical report lists the plotted estimates. |
| `analysis/paper/p23_case.py` | Fig. 4 of the technical report (U.S. Congress case study) |
| `analysis/p2/p21_unsolvable.py`, `p22_open_groups.py`, `p24_analysis.py`, `p25_analysis.py` | Technical report: queries without an LCC (Table XVII), open groups at group level (Table XIX), tolerance scaling (Table XVIII), and synthetic robustness (Table XI), respectively |

## Notes on reproducibility

- **Randomness.** Queries, synthetic positions and tolerances are drawn from per-dataset random streams derived from the configuration seed, so rerunning a dataset gives the same queries regardless of which other datasets are listed.
- **Timings** depend on the machine. Our numbers come from an Intel Xeon Gold 6530 with 64 GB of memory, one engine process per measurement. The k-core reference on Friendster was measured in a separate run; on the same machine, its check on Pokec took 1.20 times the original time.
- **Analysis plans.** `docs/PLANS.md` lists the analysis plans and section identifiers referenced by the experiment code.
- **Source check.** The `env` stage of `scripts/run_p2.sh` runs `python -I p2/manifest.py check` to verify the files listed in `p2/MANIFEST_SOURCES.sha256`.
- **Yelp.** The E5 main result bundle includes `runs/e5_main/results/`. The raw Yelp data and the derived `geo/`, `graph/` and `work/` directories stay on the machine used for the analysis.

## Licence

MIT (see `LICENSE`). The Congressional Twitter network in `e6/data/` keeps its own MIT licence (`e6/data/LICENSE_CongressionalTwitterNetwork.txt`).

## Citation

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

We will update this citation if the conference version is accepted.
