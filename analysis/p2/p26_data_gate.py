"""P2-6 data gate (plan section 6.3): structure and size only, no outcome statistics."""
import json, sys
import numpy as np, pandas as pd
d = sys.argv[1]
mep = json.load(open(f"{d}/mep_info.json"))
rcv = json.load(open(f"{d}/rcv_info.json"))
V = pd.read_csv(f"{d}/mep_rcv_vote.csv", index_col=0)
rt = pd.read_csv(f"{d}/retweets.csv", dtype=str)
print("MEPs in mep_info:", len(mep), " RCVs in rcv_info:", len(rcv))
print("vote matrix:", V.shape, " values:", pd.Series(V.to_numpy().ravel()).value_counts(dropna=False).to_dict())
print("retweet rows:", len(rt))
# timestamps
t = pd.to_datetime(rt["retweetCreatedAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
print("retweet time parse failures:", int(t.isna().sum()), " range:", t.min(), t.max())
o = pd.to_datetime(rt["origCreatedAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
print("orig time parse failures:", int(o.isna().sum()))
# ids
rt["src"] = rt["retweetMepId"]; rt["dst"] = rt["origMepId"]
print("missing mep ids:", int(rt["src"].isna().sum()), int(rt["dst"].isna().sum()), " self-retweets:", int((rt.src == rt.dst).sum()))
ids_vote = set(V.index.astype(str))
ids_mep = set(mep.keys())
print("vote-matrix ids in mep_info:", len(ids_vote & ids_mep), "/", len(ids_vote))
print("retweet ids in mep_info:", len(set(rt.src) & ids_mep), len(set(rt.dst) & ids_mep), "of", rt.src.nunique(), rt.dst.nunique())
# positions: >= 20 FOR/AGAINST votes (wnominate default minvotes = 20, yea/nay only)
nv = ((V == 0) | (V == 1)).sum(axis=1)
has_pos = set(nv.index[nv >= 20].astype(str))
print("MEPs with >= 20 FOR/AGAINST votes:", len(has_pos))
tr = (t >= pd.Timestamp("2014-10-01", tz="UTC")) & (t < pd.Timestamp("2015-07-01", tz="UTC"))
te = (t >= pd.Timestamp("2015-07-01", tz="UTC")) & (t < pd.Timestamp("2016-03-02", tz="UTC"))
print("retweets in training / test window:", int(tr.sum()), int(te.sum()), " outside:", int((~tr & ~te).sum()))
x = rt[tr & (rt.src != rt.dst)]
x = x[x.src.isin(has_pos) & x.dst.isin(has_pos)]
nd = x.groupby("src").dst.nunique()
print("MEPs with position and >= 5 distinct retweeted MEPs (training):", int((nd >= 5).sum()))
e = {tuple(sorted(p)) for p in zip(x.src, x.dst)}
print("training undirected edges among MEPs with position:", len(e))
xa = rt[tr & (rt.src != rt.dst)]
print("training undirected edges, all MEPs:", len({tuple(sorted(p)) for p in zip(xa.src, xa.dst)}))
