# Analysis plans

Comments in the code cite the analysis plans below by identifier and section, for example `docs/PLANS.md (lacs_p2_plan_20261026) section 4`. Each plan fixed the data, definitions, measures, analyses and decision rules of an experiment before its results were available; later changes were added as dated sections and never replaced the original text. The paper reports the experiments as these plans prescribe.

| Identifier | Experiment | Fixes |
| --- | --- | --- |
| round-4 plan | Meetup application on Gowalla and Brightkite, re-evaluation on all eligible queries (`scripts/run_round4.sh`) | queries, methods, measures; the size-controlled comparisons on these two datasets were added post hoc, as the paper states |
| `lacs_e5_prereg_draft_20261025` | Yelp replication of the meetup application (`e5/`); its section 8 also covers the open-group sweep (`e1x/`) | time split, users, graph, homes and tolerances; primary and secondary measures; H1 (size-matched comparisons a and b), H2, H3; decision rules (section 6); open-group weights |
| `lacs_e1_open_groups_20261025` | Open groups (`e1x/conditions.py`) | the open-group proposition and the conditions checked statically |
| `lacs_e6_plan_20261025` | U.S. Congress (`e6/`) | positions (DW-NOMINATE), tolerances from co-sponsorship, network; Q1 premise test (penalised fixed-effects logit, permutation test, out-of-fold log loss, decision table of section 5), Q2 static violations, Q3 dynamics, Q4 meetup-style comparison |
| `lacs_p2_plan_20261026` | Supplementary experiments (`p2/`, `p2x/`, `analysis/p2/`) | section 1: classes of queries without an LCC; section 2: open groups at group level; section 3: the case figure; section 4: tolerance scaling and noise; section 5: the synthetic-setting rerun with the current engine; section 6: the data gate of a second parliament |
| `lacs_e12_plan_20261029` | European Parliament (`e12/`); finalised on 2026-10-26 despite the date in its name | positions from roll calls (classical MDS), tolerances from training-period retweets, nodes and graph, test events; Q1 premise test and its variants (section 5); Q2–Q4 as for Congress |

The plan documents themselves are working notes of the authors and are available on request.
