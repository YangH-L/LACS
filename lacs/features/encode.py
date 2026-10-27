"""Message embeddings (GPU sentence encoder or CPU TF-IDF fallback) + PCA.

Both the TF-IDF vocabulary/SVD and the PCA are fitted on training-period
messages only, so nothing about later text leaks into the representation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("lacs.features")


def _fingerprint(cids: list[str], enc_cfg: dict) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({k: enc_cfg.get(k) for k in sorted(enc_cfg)}, sort_keys=True, default=str).encode())
    for c in cids:
        h.update(c.encode())
        h.update(b"\0")
    return h.hexdigest()


def encode_sbert(texts: list[str], enc: dict) -> np.ndarray:
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - depends on the server
        raise ImportError("encoder.backend=sbert needs torch and sentence-transformers "
                          "Use encoder.backend=tfidf for a CPU fallback.") from e
    import torch

    device = enc.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA not available, encoding on CPU (slow)")
        device = "cpu"
    model = SentenceTransformer(enc["model"], device=device)
    model.max_seq_length = int(enc.get("max_seq_length", 256))
    if device.startswith("cuda") and enc.get("fp16", True):
        model = model.half()
    emb = model.encode(texts, batch_size=int(enc.get("batch_size", 256)), convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def encode_tfidf(texts: list[str], fit_mask: np.ndarray, enc: dict, seed: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(min_df=2, max_features=100000, sublinear_tf=True)
    fit_texts = [t for t, m in zip(texts, fit_mask) if m]
    vec.fit(fit_texts)
    X = vec.transform(texts)
    dim = int(min(enc.get("tfidf_dim", 256), X.shape[1] - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    svd.fit(X[np.flatnonzero(fit_mask)])
    Z = svd.transform(X)
    return normalize(Z).astype(np.float32)


def fit_pca(E: np.ndarray, fit_mask: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    X = E[fit_mask].astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    # economical SVD on a sample if very large
    if Xc.shape[0] > 200000:
        idx = np.random.default_rng(0).choice(Xc.shape[0], 200000, replace=False)
        Xc = Xc[idx]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[: min(dim, Vt.shape[0])]
    return mu.astype(np.float32), comps.astype(np.float32)


def run(cfg: dict, paths: dict) -> dict:
    comments = pd.read_pickle(paths["data"] / "comments.pkl")
    meta = json.loads((paths["data"] / "meta.json").read_text())
    enc = cfg["encoder"]
    cids = comments["cid"].astype(str).tolist()
    texts = comments["text"].fillna("").astype(str).tolist()
    valid = np.array([len(t) > 0 for t in texts])
    fit_mask = (comments["time"].to_numpy() <= meta["t_train_end"]) & valid
    fp = _fingerprint(cids, {**enc, "pca_dim": cfg["pca_dim"]})
    out = paths["emb"]
    stamp = out / "fingerprint.txt"
    if stamp.exists() and stamp.read_text().strip() == fp and (out / "emb_pca.npy").exists():
        LOG.info("embeddings up to date (%s)", out)
        return {"cached": True}
    if enc["backend"] == "sbert":
        E = encode_sbert([t if t else " " for t in texts], enc)
    elif enc["backend"] == "tfidf":
        E = encode_tfidf(texts, fit_mask, enc, cfg["seed"])
    else:
        raise ValueError(f"unknown encoder backend {enc['backend']}")
    mu, comps = fit_pca(E, fit_mask, int(cfg["pca_dim"]))
    P = ((E - mu) @ comps.T).astype(np.float32)
    np.save(out / "emb_raw.npy", E)
    np.save(out / "emb_pca.npy", P)
    np.save(out / "valid.npy", valid)
    np.savez(out / "pca.npz", mean=mu, components=comps)
    (out / "cids.txt").write_text("\n".join(cids))
    stamp.write_text(fp)
    LOG.info("encoded %d messages (%d empty) -> raw dim %d, pca dim %d", len(cids), int((~valid).sum()),
             E.shape[1], P.shape[1])
    return {"cached": False, "dim_raw": int(E.shape[1]), "dim_pca": int(P.shape[1])}
