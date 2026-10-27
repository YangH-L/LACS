"""SJT-Lat: ordinal Social-Judgment model with personalised latitudes.

For a judgement of message/source u by receiver v with representation gap
dX = x_v - s_u (PCA space):
    d     = sqrt(||W dX||^2 + eps)                       learned metric
    a_v   = softplus(w_a . z_v + c_a + b_a[v])           latitude of acceptance
    g_v   = softplus(w_g . z_v + c_g + b_g[v]) + g_min   width of noncommitment
    theta_acc = a_v,  theta_rej = a_v + g_v
3 classes (accept / noncommit / reject):
    P(y <= 0) = sigmoid((a - d) / T),  P(y <= 1) = sigmoid((a + g - d) / T)
2 classes (accept / not):
    P(y = 0) = sigmoid((a - d) / T)                      [band: * sigmoid((d - delta) / T)]

Variants: global (no w, no b), feat (w), resid (b), full (w + b).
Objective (MAP): [sum_i w_i l_i + l2_W/2 |W|^2 + l2_w/2 |w|^2 + l2_b/2 |b|^2] / sum_i w_i,
optimised with L-BFGS using the analytic gradient below (checked numerically in
tests/test_sjtlat.py). T is fixed (the scale of W absorbs it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

LOG = logging.getLogger("lacs.model")


def softplus(x):
    return np.logaddexp(0.0, x)


def softplus_inv(y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-6)
    return np.where(y > 30, y, np.log(np.expm1(y)))


def ord3_loss(s0, s1, y):
    """Per-sample loss and derivatives w.r.t. s0, s1 for the 3-class model."""
    L = np.empty_like(s0)
    g0 = np.zeros_like(s0)
    g1 = np.zeros_like(s0)
    m0, m1, m2 = y == 0, y == 1, y == 2
    L[m0] = -log_expit(s0[m0])
    g0[m0] = -expit(-s0[m0])
    L[m2] = -log_expit(-s1[m2])
    g1[m2] = expit(s1[m2])
    if m1.any():
        a, b = s0[m1], s1[m1]
        D = b - a
        with np.errstate(over="ignore"):
            inv = 1.0 / np.expm1(D)
        L[m1] = -log_expit(b) - log_expit(-a) - np.log(-np.expm1(-D))
        g0[m1] = expit(a) + inv
        g1[m1] = -expit(-b) - inv
    return L, g0, g1


def ord2_loss(s0, y):
    L = np.where(y == 0, -log_expit(s0), -log_expit(-s0))
    g0 = np.where(y == 0, -expit(-s0), expit(s0))
    return L, g0


def band_loss(sa, sb, y):
    """P(accept) = sigmoid(sa) * sigmoid(sb)."""
    acc = y == 0
    L = np.empty_like(sa)
    ga = np.empty_like(sa)
    gb = np.empty_like(sa)
    L[acc] = -log_expit(sa[acc]) - log_expit(sb[acc])
    ga[acc] = -expit(-sa[acc])
    gb[acc] = -expit(-sb[acc])
    a, b = sa[~acc], sb[~acc]
    lse = np.logaddexp(0.0, np.logaddexp(a, b))
    L[~acc] = softplus(a) + softplus(b) - lse
    ga[~acc] = expit(a) - np.exp(a - lse)
    gb[~acc] = expit(b) - np.exp(b - lse)
    return L, ga, gb


VARIANT_PARAMS = {
    "global": {"W", "c_a", "c_g", "delta"},
    "feat": {"W", "c_a", "c_g", "delta", "w_a", "w_g"},
    "resid": {"W", "c_a", "c_g", "delta", "b_a", "b_g"},
    "full": {"W", "c_a", "c_g", "delta", "w_a", "w_g", "b_a", "b_g"},
}


@dataclass
class SJTLat:
    n_classes: int = 3
    proj_dim: int = 16
    variant: str = "full"
    l2_W: float = 1.0
    l2_w: float = 1.0
    l2_b: float = 10.0
    band: bool = False
    temperature: float = 1.0
    max_iter: int = 400
    eps: float = 1e-6
    g_min: float = 1e-3
    learn_W: bool = True
    seed: int = 0
    params: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    # ------------------------------------------------------------ helpers --
    def _shapes(self, D: int, F: int, U: int) -> dict[str, tuple]:
        s = {"W": (self.proj_dim, D), "c_a": (1,), "w_a": (F,), "b_a": (U,)}
        if self.n_classes == 3:
            s.update({"c_g": (1,), "w_g": (F,), "b_g": (U,)})
        if self.band:
            s["delta"] = (1,)
        return s

    def _free(self) -> list[str]:
        allowed = set(VARIANT_PARAMS[self.variant])
        if not self.learn_W:
            allowed.discard("W")
        return [k for k in self._shapes_cache if k in allowed]

    def _pack(self, names):
        return np.concatenate([self.params[k].ravel() for k in names]) if names else np.zeros(0)

    def _unpack(self, theta, names):
        i = 0
        for k in names:
            n = int(np.prod(self._shapes_cache[k]))
            self.params[k] = theta[i:i + n].reshape(self._shapes_cache[k])
            i += n

    # ------------------------------------------------------- forward pass --
    def _forward(self, dX, Zr, r):
        p = self.params
        P = dX @ p["W"].T
        d = np.sqrt((P * P).sum(axis=1) + self.eps)
        eta_a = Zr @ p["w_a"] + p["c_a"][0] + p["b_a"][r]
        a = softplus(eta_a)
        out = {"P": P, "d": d, "eta_a": eta_a, "a": a}
        if self.n_classes == 3:
            eta_g = Zr @ p["w_g"] + p["c_g"][0] + p["b_g"][r]
            out["eta_g"] = eta_g
            out["g"] = softplus(eta_g) + self.g_min
        return out

    def _objective(self, theta, names, dX, Zr, r, y, sw, wsum):
        self._unpack(theta, names)
        p = self.params
        T = self.temperature
        f = self._forward(dX, Zr, r)
        d, a = f["d"], f["a"]
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if self.n_classes == 3:
            s0 = (a - d) / T
            s1 = (a + f["g"] - d) / T
            L, g0, g1 = ord3_loss(s0, s1, y)
            dLda = (g0 + g1) / T
            dLdg = g1 / T
            dLdd = -(g0 + g1) / T
        elif self.band:
            delta = softplus(p["delta"][0])
            sa = (a - d) / T
            sb = (d - delta) / T
            L, ga, gb = band_loss(sa, sb, y)
            dLda = ga / T
            dLdd = (-ga + gb) / T
            grads["delta"][0] = (sw * (-gb / T)).sum() * expit(p["delta"][0])
        else:
            s0 = (a - d) / T
            L, g0 = ord2_loss(s0, y)
            dLda = g0 / T
            dLdd = -g0 / T
        loss = (sw * L).sum()
        ga_eta = sw * dLda * expit(f["eta_a"])
        grads["w_a"] = Zr.T @ ga_eta
        grads["c_a"][0] = ga_eta.sum()
        grads["b_a"] = np.bincount(r, weights=ga_eta, minlength=p["b_a"].shape[0])
        if self.n_classes == 3:
            gg_eta = sw * dLdg * expit(f["eta_g"])
            grads["w_g"] = Zr.T @ gg_eta
            grads["c_g"][0] = gg_eta.sum()
            grads["b_g"] = np.bincount(r, weights=gg_eta, minlength=p["b_g"].shape[0])
        coef = sw * dLdd / d
        grads["W"] = (coef[:, None] * f["P"]).T @ dX
        # penalties
        loss += 0.5 * self.l2_W * (p["W"] ** 2).sum()
        grads["W"] = grads["W"] + self.l2_W * p["W"]
        for k in ("w_a", "w_g"):
            if k in p:
                loss += 0.5 * self.l2_w * (p[k] ** 2).sum()
                grads[k] = grads[k] + self.l2_w * p[k]
        for k in ("b_a", "b_g"):
            if k in p:
                loss += 0.5 * self.l2_b * (p[k] ** 2).sum()
                grads[k] = grads[k] + self.l2_b * p[k]
        g = np.concatenate([grads[k].ravel() for k in names]) if names else np.zeros(0)
        return loss / wsum, g / wsum

    # ---------------------------------------------------------------- API --
    def init_params(self, dX, Z, r, y, n_users, warm: "SJTLat | None" = None):
        D, F = dX.shape[1], Z.shape[1]
        self._shapes_cache = self._shapes(D, F, n_users)
        rng = np.random.default_rng(self.seed)
        p = {k: np.zeros(s) for k, s in self._shapes_cache.items()}
        if warm is not None and warm.params.get("W") is not None and warm.params["W"].shape == p["W"].shape:
            for k in p:
                if k in warm.params and warm.params[k].shape == p[k].shape:
                    p[k] = warm.params[k].copy()
            self.params = p
            return
        idx = rng.choice(len(dX), size=min(len(dX), 20000), replace=False)
        _, _, Vt = np.linalg.svd(dX[idx] - dX[idx].mean(0), full_matrices=False)
        m = min(self.proj_dim, Vt.shape[0])
        W = np.zeros((self.proj_dim, D))
        W[:m] = Vt[:m]
        d0 = np.linalg.norm(dX @ W.T, axis=1)
        W *= 3.0 / max(np.median(d0), 1e-9)
        d0 = np.linalg.norm(dX @ W.T, axis=1)
        p["W"] = W
        q0 = np.mean(y == 0)
        a0 = np.quantile(d0, np.clip(q0, 0.02, 0.98))
        p["c_a"][0] = softplus_inv(a0)
        if self.n_classes == 3:
            q1 = np.mean(y <= 1)
            g0 = max(np.quantile(d0, np.clip(q1, 0.02, 0.98)) - a0, 0.2)
            p["c_g"][0] = softplus_inv(g0)
        if self.band:
            p["delta"][0] = softplus_inv(np.quantile(d0, 0.02))
        self.params = p

    def fit(self, dX, Z, r, y, n_users, sample_weight=None, warm: "SJTLat | None" = None, verbose=False):
        dX = np.asarray(dX, dtype=np.float64)
        Zr = np.asarray(Z, dtype=np.float64)[r]
        y = np.asarray(y, dtype=int)
        r = np.asarray(r, dtype=int)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.init_params(dX, Z, r, y, n_users, warm)
        names = self._free()
        x0 = self._pack(names)
        res = minimize(self._objective, x0, args=(names, dX, Zr, r, y, sw, sw.sum()), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter, "maxfun": 2 * self.max_iter})
        self._unpack(res.x, names)
        self.history = {"loss": float(res.fun), "nit": int(res.nit), "success": bool(res.success),
                        "message": str(res.message)}
        if verbose:
            LOG.info("SJTLat[%s] loss=%.4f nit=%d %s", self.variant, res.fun, res.nit, res.message)
        return self

    def predict_proba(self, dX, Z, r):
        dX = np.asarray(dX, dtype=np.float64)
        Zr = np.asarray(Z, dtype=np.float64)[r]
        f = self._forward(dX, Zr, np.asarray(r, dtype=int))
        T = self.temperature
        if self.n_classes == 3:
            c0 = expit((f["a"] - f["d"]) / T)
            c1 = expit((f["a"] + f["g"] - f["d"]) / T)
            P = np.stack([c0, c1 - c0, 1.0 - c1], axis=1)
        elif self.band:
            delta = softplus(self.params["delta"][0])
            pa = expit((f["a"] - f["d"]) / T) * expit((f["d"] - delta) / T)
            P = np.stack([pa, 1.0 - pa], axis=1)
        else:
            pa = expit((f["a"] - f["d"]) / T)
            P = np.stack([pa, 1.0 - pa], axis=1)
        return np.clip(P, 1e-12, 1.0)

    def distances(self, dX):
        P = np.asarray(dX, dtype=np.float64) @ self.params["W"].T
        return np.sqrt((P * P).sum(axis=1) + self.eps)

    def project(self, X):
        """Positions in the learned metric space: ||proj(x)-proj(y)|| = d."""
        return np.asarray(X, dtype=np.float64) @ self.params["W"].T

    def latitudes(self, Z, users=None):
        Z = np.asarray(Z, dtype=np.float64)
        idx = np.arange(Z.shape[0]) if users is None else np.asarray(users)
        p = self.params
        acc = softplus(Z[idx] @ p["w_a"] + p["c_a"][0] + p["b_a"][idx])
        if self.n_classes == 3:
            rej = acc + softplus(Z[idx] @ p["w_g"] + p["c_g"][0] + p["b_g"][idx]) + self.g_min
        else:
            rej = np.full_like(acc, np.nan)
        return acc, rej

    def save(self, path):
        np.savez(path, **{f"p_{k}": v for k, v in self.params.items()},
                 config=np.array([repr({k: getattr(self, k) for k in (
                     "n_classes", "proj_dim", "variant", "l2_W", "l2_w", "l2_b", "band", "temperature",
                     "max_iter", "eps", "g_min", "learn_W", "seed")})]))

    @classmethod
    def load(cls, path):
        import ast

        z = np.load(path, allow_pickle=False)
        cfg = ast.literal_eval(str(z["config"][0]))
        m = cls(**cfg)
        m.params = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
        m._shapes_cache = {k: v.shape for k, v in m.params.items()}
        return m
