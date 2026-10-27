"""Optional E1 baseline: Signed GCN (Derr et al., ICDM 2018) via PyTorch Geometric.

NOT RUN in the packaging environment (no torch / torch_geometric there). Written
against the API used in PyG's examples/signed_gcn.py; check it once on the
server before relying on the numbers.

Task: sign prediction on judgements, accept = +1 / reject = -1 (noncommit
dropped). Train on the training split, report AUC/F1 on the test split, i.e.
comparable to `auc_reject_vs_accept` in results/e1_metrics.csv.

    pip install torch_geometric
    python scripts/optional/sgcn_baseline.py --run runs/debagreement
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import SignedGCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    root = Path(args.run)
    F = np.load(root / "features" / "pairs.npz", allow_pickle=False)
    meta = json.loads((root / "data" / "meta.json").read_text())
    K = int(meta["n_classes"])
    y, split, recv, src = F["y"], F["split"], F["recv"], F["src"]
    pos_lab, neg_lab = 0, K - 1          # accept / reject (binary tasks: accept / not)
    n = int(max(recv.max(), src.max())) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def edges(mask, lab):
        m = mask & (y == lab)
        return torch.tensor(np.stack([recv[m], src[m]]), dtype=torch.long, device=device)

    tr, te = split == 0, split == 2
    tr_pos, tr_neg = edges(tr, pos_lab), edges(tr, neg_lab)
    te_pos, te_neg = edges(te, pos_lab), edges(te, neg_lab)
    model = SignedGCN(args.dim, args.dim, num_layers=2, lamb=5).to(device)
    x = model.create_spectral_features(tr_pos, tr_neg, num_nodes=n)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        z = model(x, tr_pos, tr_neg)
        loss = model.loss(z, tr_pos, tr_neg)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model(x, tr_pos, tr_neg)
                auc, f1 = model.test(z, te_pos, te_neg)
            best = {"epoch": epoch + 1, "auc": float(auc), "f1": float(f1), "loss": float(loss)}
            print(best)
    out = root / "results" / "e1_sgcn.json"
    out.write_text(json.dumps(best, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
