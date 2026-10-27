"""SJT-Lat: analytic gradient vs finite differences, and parameter recovery.

Run: python -m unittest tests.test_sjtlat -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacs.model.sjtlat import SJTLat, band_loss, ord3_loss  # noqa: E402


def toy(rng, n=300, D=6, F=3, U=25, K=3):
    dX = rng.normal(size=(n, D))
    Z = rng.normal(size=(U, F))
    r = rng.integers(0, U, size=n)
    y = rng.integers(0, K, size=n)
    return dX, Z, r, y


def fd_check(model, dX, Z, r, y, U, rng, eps=1e-6):
    model.init_params(dX, Z, r, y, U)
    # move away from zero so every parameter matters
    for k, v in model.params.items():
        model.params[k] = v + 0.3 * rng.normal(size=v.shape)
    names = model._free()
    theta = model._pack(names)
    sw = rng.uniform(0.5, 2.0, size=len(y))
    args = (names, dX, Z[r], r, y, sw, sw.sum())
    f0, g = model._objective(theta.copy(), *args)
    idx = rng.choice(len(theta), size=min(40, len(theta)), replace=False)
    num = np.zeros(len(idx))
    for j, i in enumerate(idx):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        num[j] = (model._objective(tp, *args)[0] - model._objective(tm, *args)[0]) / (2 * eps)
    return g[idx], num


class TestGradients(unittest.TestCase):
    def test_loss_derivatives(self):
        rng = np.random.default_rng(0)
        s0 = rng.normal(size=200)
        s1 = s0 + rng.uniform(0.01, 3.0, size=200)
        y = rng.integers(0, 3, size=200)
        L, g0, g1 = ord3_loss(s0, s1, y)
        e = 1e-6
        Lp, _, _ = ord3_loss(s0 + e, s1, y)
        Lm, _, _ = ord3_loss(s0 - e, s1, y)
        np.testing.assert_allclose(g0, (Lp - Lm) / (2 * e), rtol=1e-4, atol=1e-6)
        Lp, _, _ = ord3_loss(s0, s1 + e, y)
        Lm, _, _ = ord3_loss(s0, s1 - e, y)
        np.testing.assert_allclose(g1, (Lp - Lm) / (2 * e), rtol=1e-4, atol=1e-6)
        # probabilities sum to one
        c0, c1 = expit(s0), expit(s1)
        P = np.stack([c0, c1 - c0, 1 - c1], 1)
        np.testing.assert_allclose(-np.log(P[np.arange(200), y]), L, rtol=1e-8, atol=1e-10)
        ya = rng.integers(0, 2, size=200)
        sa, sb = rng.normal(size=200), rng.normal(size=200)
        L, ga, gb = band_loss(sa, sb, ya)
        Lp, _, _ = band_loss(sa + e, sb, ya)
        Lm, _, _ = band_loss(sa - e, sb, ya)
        np.testing.assert_allclose(ga, (Lp - Lm) / (2 * e), rtol=1e-4, atol=1e-6)
        Lp, _, _ = band_loss(sa, sb + e, ya)
        Lm, _, _ = band_loss(sa, sb - e, ya)
        np.testing.assert_allclose(gb, (Lp - Lm) / (2 * e), rtol=1e-4, atol=1e-6)

    def test_model_gradients(self):
        rng = np.random.default_rng(1)
        for K, band in ((3, False), (2, False), (2, True)):
            for variant in ("global", "feat", "resid", "full"):
                dX, Z, r, y = toy(rng, K=K)
                m = SJTLat(n_classes=K, proj_dim=4, variant=variant, band=band, l2_W=0.7, l2_w=0.3, l2_b=2.0)
                ga, gn = fd_check(m, dX, Z, r, y, Z.shape[0], rng)
                np.testing.assert_allclose(ga, gn, rtol=2e-4, atol=2e-6,
                                           err_msg=f"K={K} band={band} variant={variant}")


class TestRecovery(unittest.TestCase):
    def test_personalised_latitudes_are_recovered(self):
        rng = np.random.default_rng(3)
        U, D, F, n = 150, 8, 3, 12000
        Z = rng.normal(size=(U, F))
        W_true = rng.normal(size=(2, D)) / np.sqrt(D)
        acc = np.log1p(np.exp(0.0 + 0.7 * Z[:, 0]))          # feature 0 widens acceptance
        gap = np.log1p(np.exp(0.2 - 0.5 * Z[:, 1]))          # feature 1 narrows noncommitment
        r = rng.integers(0, U, size=n)
        dX = rng.normal(size=(n, D)) * 1.5
        d = np.linalg.norm(dX @ W_true.T, axis=1)
        T = 0.25
        c0 = expit((acc[r] - d) / T)
        c1 = expit((acc[r] + gap[r] - d) / T)
        u = rng.random(n)
        y = np.where(u < c0, 0, np.where(u < c1, 1, 2))
        tr, te = np.arange(n) < 9000, np.arange(n) >= 9000
        nll = {}
        models = {}
        warm = None
        for v in ("global", "feat", "full"):
            m = SJTLat(n_classes=3, proj_dim=2, variant=v, l2_W=0.1, l2_w=0.1, l2_b=5.0, max_iter=600)
            m.fit(dX[tr], Z, r[tr], y[tr], U, warm=warm)
            warm = m if v == "global" else warm
            P = m.predict_proba(dX[te], Z, r[te])
            nll[v] = float(-np.log(P[np.arange(te.sum()), y[te]]).mean())
            models[v] = m
        self.assertLess(nll["feat"], nll["global"] - 0.01, msg=str(nll))
        self.assertLess(nll["full"], nll["global"] - 0.01, msg=str(nll))
        w = models["feat"].params
        self.assertGreater(w["w_a"][0], 0.0)    # acceptance widens with feature 0
        self.assertLess(w["w_g"][1], 0.0)       # noncommitment narrows with feature 1
        acc_hat, rej_hat = models["feat"].latitudes(Z)
        self.assertGreater(np.corrcoef(acc_hat, acc)[0, 1], 0.8)
        self.assertTrue(np.all(rej_hat > acc_hat))


if __name__ == "__main__":
    unittest.main()
