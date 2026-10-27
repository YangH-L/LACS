# LACS: Community Search under Member-Specific Tolerances

This repository contains the code, experiment drivers and analysis scripts for the paper

> **Community Search under Member-Specific Tolerances** (ICDE 2027 submission; technical report on arXiv, link to be added).

A *latitude-consistent community* (LCC) is a connected subgraph of minimum degree at least *k* that contains a query vertex and whose diameter, in the space of the members' positions, is at most the smallest tolerance of its members. The repository implements LatSweep (BallCore-F, Greedy, BallCore-R and the exact branch and bound), the baselines, the opinion-dynamics simulations and every experiment of Section VI.

No dataset is included except the Congressional Twitter network (MIT licence, `e6/data/`). Every other dataset is downloaded from its official source (see [Data](#data)). In particular, **this repository does not contain or redistribute any part of the Yelp Open Dataset**; it contains only the scripts that process it.

## Layout

| Path | Contents |
| --- | --- |
| `cpp/` | C++17 search engine: `lacs_search` (all search algorithms), `lacs_convert` (edge list to binary CSR, with a low-memory mode for Friendster), `lacs_lpa` (size-capped label propagation), `lacs_project` |
| `lacs/` | Python package (`python -m lacs <command>`): graph I/O, synthetic positions and tolerances, the engine wrapper, the efficiency experiment (`e3`), the meetup application (`geo-*`) |
| `p0/` | Opinion dynamics kernels (C++, via ctypes) and drivers: dynamics on returned groups, heterogeneity, the solver comparison, robustness of the synthetic setting |
| `e1x/` | Open groups: dynamics with friends outside the group, and the static conditions of the open-group proposition |
| `e5/` | Yelp: preparation, feasibility check, analysis of the meetup comparison |
| `e6/` | U.S. Congress: data download, positions, tolerances from co-sponsorship, premise test, static violations, report |
| `e12/` | European Parliament: positions from roll calls, tolerances from retweets, premise test |
| `p2/` | Tolerance scaling (`e11.py`), tolerance noise, and the synthetic-setting rerun with the current engine (`e4v05.py`) |
| `p2x/` | The k-core reference time on Friendster (Table I) |
| `analysis/p2/` | Analyses of the queries without an LCC, open groups at group level, tolerance scaling and the synthetic-setting rerun |
| `analysis/paper/` | Scripts that write every table and figure of the paper from the result files (no number is typed by hand) |
| `configs/` | Experiment configurations |
| `scripts/` | Run scripts (each runs its stages in order, resumes after interruption, and reports failed stages) |
| `tests/`, `*/tests/` | Unit tests and self-tests |
| `docs/PLANS.md` | The analysis plans that the code refers to |

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
| Meetup application on Gowalla and Brightkite (inputs of most later steps) | `bash scripts/run_round3.sh` (preparation, gates, E3 on LiveJournal and Orkut), then `bash scripts/run_round4.sh` (all eligible queries, k and tolerance sweeps) | `runs/{gowalla,brightkite}/`, `runs/efficiency_round3/` |
| Table I (efficiency), unproven queries, scalability, ablation | `bash scripts/run_w2.sh` (run A: Gowalla, Brightkite, LiveJournal, Orkut; run B: DBLP, YouTube, Pokec, Friendster) | `runs/efficiency_w2*/results/` |
| Table I, k-core column on Friendster | `python p2x/fkcore.py run` (after run B; times the k-core reference on the same 200 queries, after a check on Pokec; on another machine set `P2X_SKIP_TIMING=1`, which keeps the size check and skips the time band) | `runs_p2/p2x/p2x_summary.json` |
| Table II (solver comparison), Table III (dynamics on returned groups), heterogeneity | `bash scripts/run_p0.sh` | `runs_p0/` |
| Limits of the synthetic setting | `STAGES="env build selftest p25 bundle" bash scripts/run_p2.sh` | `runs_p2/e4v05/` |
| Open groups (Section VI-C) | `STAGES="e1sweep" bash scripts/run_e5_main.sh` (uses the groups exported by `run_p0.sh`) | `runs/e1_open/` |
| Yelp (Section VI-D, Fig. 3, Table IV) | `bash scripts/run_e5_feas.sh`, then `STAGES="env build selftest e5prep e5gates e5pilot e5main e5sweep e5analysis" bash scripts/run_e5_main.sh` | `runs/e5_main/results/` |
| Queries without an LCC; tolerance scaling and noise (Section VI-D) | `STAGES="env build selftest p24pilot p24gate p24 p24b p24sum bundle" bash scripts/run_p2.sh` | `runs_p2/e11/` |
| U.S. Congress (Section VI-E, Table V, Fig. 4) | `STAGES="env build selftest e6fetch e6build e6geo e6dyn e6premise e6report" bash scripts/run_e5_main.sh` | `runs/e6/` |
| European Parliament (Section VI-E) | `python3 -I analysis/p2/p26_data_gate.py /path/to/MEP_files` (data check), then `DATA=/path/to/MEP_files RUN=runs/e12 bash e12/run_e12.sh` | `runs/e12/results/` |

Analyses and paper tables: `bash analysis/make_workdir.sh` links the result directories into `work/` in the layout that the analysis scripts expect; the comments at its top give the commands in order. Every analysis script documents its inputs in its docstring:

| Script | Paper |
| --- | --- |
| `analysis/paper/w2_numbers.py`, `make_tables.py` | Table I and the numbers of Section VI-B; `make_tables.py` writes every table |
| `analysis/paper/p0_e3_numbers.py`, `p0_e1_checks.py` | Tables II and III |
| `analysis/paper/geo_size_check.py`, `geo_posthoc_check.py` | post hoc size-controlled comparisons on Gowalla and Brightkite (Fig. 3, Table IV) |
| `analysis/paper/make_fig_meetup.py` | Fig. 3 (reads `meetup_values.json`, which `make_tables.py` writes next to itself) |
| `analysis/paper/p23_case.py` | Fig. 4 |
| `analysis/p2/p21_unsolvable.py`, `p22_open_groups.py`, `p24_analysis.py`, `p25_analysis.py` | queries without an LCC, open groups at group level, tolerance scaling, limits of the synthetic setting |

## Notes on reproducibility

- **Randomness.** Queries, synthetic positions and tolerances are drawn from per-dataset random streams derived from the configuration seed, so rerunning a dataset gives the same queries regardless of which other datasets are listed.
- **Timings** depend on the machine. Our numbers come from an Intel Xeon Gold 6530 with 64 GB of memory, one engine process per measurement. The k-core reference on Friendster was measured in a separate run; on the same machine, its check on Pokec took 1.20 times the original time.
- **Analysis plans.** The analyses of Yelp, Congress, the European Parliament and the supplementary experiments follow plans written before their results were available (`docs/PLANS.md`); the code refers to their sections.
- **Source check.** The `env` stage of `scripts/run_p2.sh` compares every source file with `p2/MANIFEST_SOURCES.sha256`; after editing any file, run `python -I p2/manifest.py write` first.
- **Yelp.** Under the Dataset User Agreement, the data stay with the user who downloaded them; the run scripts never copy user-level Yelp data into result bundles.

## Licence

MIT (see `LICENSE`). The Congressional Twitter network in `e6/data/` keeps its own MIT licence (`e6/data/LICENSE_CongressionalTwitterNetwork.txt`).

## Citation

To be added after publication.
