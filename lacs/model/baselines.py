"""Baselines for acceptance prediction (E1). None of them sees the reply text
except ReplyTextReference, which is reported only as an upper reference."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .sjtlat import ord2_loss, ord3_loss, softplus, softplus_inv


class Prior:
    """Training-set class frequencies."""

    def fit(self, y, K):
        self.p = np.bincount(y, minlength=K) / len(y)
        return self

    def predict_proba(self, n):
        return np.clip(np.tile(self.p, (n, 1)), 1e-12, 1)


def _lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


class DistLogit:
    """Multinomial logistic regression on the raw distance and receiver
    features: same inputs as SJT-Lat, no ordinal/latitude structure."""

    def features(self, d_raw, Zr, recv_hist, src_hist):
        return np.column_stack([d_raw, d_raw ** 2, np.log1p(recv_hist), np.log1p(src_hist), Zr,
                                Zr * d_raw[:, None]])

    def fit(self, X, y, C=1.0):
        self.m = _lr(C).fit(X, y)
        self.classes = self.m.classes_
        return self

    def predict_proba(self, X, K):
        P = np.full((X.shape[0], K), 1e-12)
        P[:, self.classes] = self.m.predict_proba(X)
        return np.clip(P / P.sum(1, keepdims=True), 1e-12, 1)


class MFOrdinal:
    """Collaborative baseline: s = <P_v, Q_u> + bias_v + bias_u with ordinal
    thresholds. Learns who-accepts-whom from ids only."""

    def __init__(self, rank=8, l2=1.0, max_iter=300, seed=0):
        self.rank, self.l2, self.max_iter, self.seed = rank, l2, max_iter, seed

    def _unpack(self, th):
        U, r = self.U, self.rank
        i = 0
        P = th[i:i + U * r].reshape(U, r); i += U * r
        Q = th[i:i + U * r].reshape(U, r); i += U * r
        bv = th[i:i + U]; i += U
        bu = th[i:i + U]; i += U
        c = th[i:]
        return P, Q, bv, bu, c

    def _obj(self, th, v, u, y):
        P, Q, bv, bu, c = self._unpack(th)
        s = (P[v] * Q[u]).sum(1) + bv[v] + bu[u]
        n = len(y)
        if self.K == 3:
            c1 = c[0] + softplus(c[1])
            L, g0, g1 = ord3_loss(c[0] - s, c1 - s, y)
            ds = -(g0 + g1)
            gc = np.array([g0.sum() + g1.sum(), g1.sum() * (1 / (1 + np.exp(-c[1])))])
        else:
            L, g0 = ord2_loss(c[0] - s, y)
            ds = -g0
            gc = np.array([g0.sum()])
        gP = np.zeros_like(P); gQ = np.zeros_like(Q)
        np.add.at(gP, v, ds[:, None] * Q[u])
        np.add.at(gQ, u, ds[:, None] * P[v])
        gbv = np.bincount(v, weights=ds, minlength=self.U)
        gbu = np.bincount(u, weights=ds, minlength=self.U)
        reg = 0.5 * self.l2 * ((P ** 2).sum() + (Q ** 2).sum() + (bv ** 2).sum() + (bu ** 2).sum())
        g = np.concatenate([(gP + self.l2 * P).ravel(), (gQ + self.l2 * Q).ravel(), gbv + self.l2 * bv,
                            gbu + self.l2 * bu, gc])
        return (L.sum() + reg) / n, g / n

    def fit(self, v, u, y, n_users, K):
        self.U, self.K = n_users, K
        rng = np.random.default_rng(self.seed)
        init_c = [0.0, softplus_inv(1.0)] if K == 3 else [0.0]
        th0 = np.concatenate([0.01 * rng.normal(size=2 * n_users * self.rank), np.zeros(2 * n_users), init_c])
        res = minimize(self._obj, th0, args=(v, u, y), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.th = res.x
        return self

    def predict_proba(self, v, u):
        from scipy.special import expit

        P, Q, bv, bu, c = self._unpack(self.th)
        s = (P[v] * Q[u]).sum(1) + bv[v] + bu[u]
        if self.K == 3:
            c0 = expit(c[0] - s)
            c1 = expit(c[0] + softplus(c[1]) - s)
            out = np.stack([c0, c1 - c0, 1 - c1], 1)
        else:
            p = expit(c[0] - s)
            out = np.stack([p, 1 - p], 1)
        return np.clip(out, 1e-12, 1)


class BalanceFeatures:
    """Signed-network baseline in the spirit of Leskovec et al. (WWW 2010):
    degree and triad features computed from TRAINING judgements only
    (accept = +1, noncommit = 0, reject = -1), then logistic regression."""

    def fit_graph(self, v, u, y, K):
        sign = np.where(y == 0, 1, np.where(y == K - 1, -1, 0)) if K == 3 else np.where(y == 0, 1, -1)
        self.out_cnt, self.in_cnt, self.pair_cnt = {}, {}, {}
        self.nbr: dict[int, dict[int, int]] = {}
        for a, b, s in zip(v, u, sign):
            self.out_cnt.setdefault(a, np.zeros(3))[s + 1] += 1
            self.in_cnt.setdefault(b, np.zeros(3))[s + 1] += 1
            self.pair_cnt.setdefault((a, b), np.zeros(3))[s + 1] += 1
            for x, y_ in ((a, b), (b, a)):
                d = self.nbr.setdefault(x, {})
                d[y_] = d.get(y_, 0) + s
        return self

    def features(self, v, u):
        rows = []
        z3 = np.zeros(3)
        for a, b in zip(v, u):
            o = self.out_cnt.get(a, z3)
            i = self.in_cnt.get(b, z3)
            pr = self.pair_cnt.get((a, b), z3)
            rp = self.pair_cnt.get((b, a), z3)
            na, nb = self.nbr.get(a, {}), self.nbr.get(b, {})
            common = set(na) & set(nb)
            tri = np.zeros(4)
            for w in common:
                sa, sb = np.sign(na[w]), np.sign(nb[w])
                if sa != 0 and sb != 0:
                    tri[(sa < 0) * 2 + (sb < 0)] += 1
            rows.append(np.concatenate([np.log1p(o), o / max(o.sum(), 1), np.log1p(i), i / max(i.sum(), 1),
                                        np.log1p(pr), np.log1p(rp), np.log1p(tri), [np.log1p(len(common))]]))
        return np.array(rows)

    def fit(self, X, y, C=1.0):
        self.m = _lr(C).fit(X, y)
        self.classes = self.m.classes_
        return self

    def predict_proba(self, X, K):
        P = np.full((X.shape[0], K), 1e-12)
        P[:, self.classes] = self.m.predict_proba(X)
        return np.clip(P / P.sum(1, keepdims=True), 1e-12, 1)


class ReplyTextReference(DistLogit):
    """Upper reference that SEES the reply text (judged message + reply
    embeddings). Not comparable to the other models; it shows how much of the
    label is directly readable from the reply."""

    def features(self, e_src, e_dst):
        return np.hstack([e_src, e_dst, e_src * e_dst, np.abs(e_src - e_dst)])
