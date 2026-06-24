"""Semi-supervised tabular regression baselines.

The implementations are intentionally compact wrappers around common default
recipes so the benchmark can compare against VIME, RankUp, and SCARF without
turning these baselines into tuned competitors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _prepare_unlabeled(x_l: np.ndarray, x_u: Optional[np.ndarray], max_unlabeled: int, seed: int) -> np.ndarray:
    if x_u is None or len(x_u) == 0:
        return x_l
    x_u = _as_float_array(x_u)
    if max_unlabeled and len(x_u) > max_unlabeled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_u), size=max_unlabeled, replace=False)
        x_u = x_u[idx]
    return np.vstack([x_l, x_u]).astype(np.float32, copy=False)


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _inverse_y(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _VIMENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim=hidden_dim, depth=2, dropout=0.1)
        self.mask_head = nn.Linear(hidden_dim, input_dim)
        self.recon_head = nn.Linear(hidden_dim, input_dim)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward_pretext(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.mask_head(z), self.recon_head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reg_head(self.encoder(x)).squeeze(-1)


class _RankUpNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim=hidden_dim, depth=2, dropout=0.1)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.rank_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reg_head(self.encode(x)).squeeze(-1)

    def rank_logits(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        z_a = self.encode(x_a)
        z_b = self.encode(x_b)
        return self.rank_head(torch.cat([z_a, z_b], dim=1)).squeeze(-1)


class _SCARFNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim=hidden_dim, depth=2, dropout=0.1)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim))
        self.reg_head = nn.Linear(hidden_dim, 1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reg_head(self.encoder(x)).squeeze(-1)


class _UCVMENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim=hidden_dim, depth=2, dropout=dropout)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reg_head(self.encoder(x)).squeeze(-1)


def _corrupt_with_pool(x: torch.Tensor, pool: torch.Tensor, corruption_rate: float) -> torch.Tensor:
    mask = torch.rand_like(x) < corruption_rate
    row_idx = torch.randint(0, pool.shape[0], x.shape, device=x.device)
    col_idx = torch.arange(x.shape[1], device=x.device).view(1, -1).expand_as(row_idx)
    replacement = pool[row_idx, col_idx]
    return torch.where(mask, replacement, x)


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@dataclass
class _BaseNNConfig:
    hidden_dim: int = 128
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    pretrain_epochs: int = 20
    max_unlabeled: int = 5000
    random_state: int = 42


class VIMERegressorBaseline(BaseEstimator, RegressorMixin):
    """VIME-style mask/reconstruction pretraining followed by supervised tuning."""

    def __init__(
        self,
        hidden_dim: int = 128,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrain_epochs: int = 20,
        epochs: int = 100,
        mask_prob: float = 0.3,
        recon_weight: float = 2.0,
        max_unlabeled: int = 5000,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.mask_prob = mask_prob
        self.recon_weight = recon_weight
        self.max_unlabeled = max_unlabeled
        self.random_state = random_state

    def fit(self, X, y, X_unlabeled=None):
        _set_seed(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32)
        x_ssl = _prepare_unlabeled(x_l, X_unlabeled, self.max_unlabeled, self.random_state)
        self.x_scaler_ = StandardScaler().fit(x_ssl)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_ssl_s = self.x_scaler_.transform(x_ssl).astype(np.float32)
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        self.model_ = _VIMENet(x_l_s.shape[1], self.hidden_dim).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        x_ssl_t = _to_tensor(x_ssl_s)
        loader = DataLoader(TensorDataset(x_ssl_t), batch_size=self.batch_size, shuffle=True, drop_last=False)
        pool = x_ssl_t.to(DEVICE)
        self.model_.train()
        for _ in range(self.pretrain_epochs):
            for (xb_cpu,) in loader:
                xb = xb_cpu.to(DEVICE)
                mask = (torch.rand_like(xb) < self.mask_prob).float()
                x_corrupt = _corrupt_with_pool(xb, pool, self.mask_prob)
                mask_logits, recon = self.model_.forward_pretext(x_corrupt)
                loss_mask = F.binary_cross_entropy_with_logits(mask_logits, mask)
                loss_recon = F.mse_loss(recon * mask, xb * mask)
                loss = loss_mask + self.recon_weight * loss_recon
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        train_loader = DataLoader(
            TensorDataset(_to_tensor(x_l_s), _to_tensor(y_s)),
            batch_size=min(self.batch_size, max(1, len(x_l_s))),
            shuffle=True,
            drop_last=False,
        )
        for _ in range(self.epochs):
            for xb_cpu, yb_cpu in train_loader:
                xb = xb_cpu.to(DEVICE)
                yb = yb_cpu.to(DEVICE)
                pred = self.model_(xb)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                xb = _to_tensor(x[start : start + 4096]).to(DEVICE)
                preds.append(self.model_(xb).detach().cpu().numpy())
        return _inverse_y(np.concatenate(preds), self.y_scaler_)


class RankUpRegressorBaseline(BaseEstimator, RegressorMixin):
    """RankUp-style regression with auxiliary pairwise order constraints."""

    def __init__(
        self,
        hidden_dim: int = 128,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        ranking_weight: float = 1.0,
        consistency_weight: float = 1.0,
        pairs_per_batch: int = 64,
        max_unlabeled: int = 5000,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.ranking_weight = ranking_weight
        self.consistency_weight = consistency_weight
        self.pairs_per_batch = pairs_per_batch
        self.max_unlabeled = max_unlabeled
        self.random_state = random_state

    def fit(self, X, y, X_unlabeled=None):
        _set_seed(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32)
        x_u = _as_float_array(X_unlabeled) if X_unlabeled is not None and len(X_unlabeled) else np.empty((0, x_l.shape[1]), dtype=np.float32)
        if self.max_unlabeled and len(x_u) > self.max_unlabeled:
            rng = np.random.default_rng(self.random_state)
            x_u = x_u[rng.choice(len(x_u), size=self.max_unlabeled, replace=False)]
        x_fit = np.vstack([x_l, x_u]) if len(x_u) else x_l
        self.x_scaler_ = StandardScaler().fit(x_fit)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_u_s = self.x_scaler_.transform(x_u).astype(np.float32) if len(x_u) else x_u
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        self.model_ = _RankUpNet(x_l_s.shape[1], self.hidden_dim).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_l_t = _to_tensor(x_l_s).to(DEVICE)
        y_l_t = _to_tensor(y_s).to(DEVICE)
        x_u_t = _to_tensor(x_u_s).to(DEVICE) if len(x_u_s) else None

        n_l = len(x_l_t)
        batch = min(self.batch_size, max(1, n_l))
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_l, device=DEVICE)
            for start in range(0, n_l, batch):
                idx = perm[start : start + batch]
                xb = x_l_t[idx]
                yb = y_l_t[idx]
                pred = self.model_(xb)
                loss = F.mse_loss(pred, yb)

                if n_l >= 2 and self.pairs_per_batch > 0:
                    ia = torch.randint(0, n_l, (self.pairs_per_batch,), device=DEVICE)
                    ib = torch.randint(0, n_l, (self.pairs_per_batch,), device=DEVICE)
                    labels = (y_l_t[ia] > y_l_t[ib]).float()
                    logits = self.model_.rank_logits(x_l_t[ia], x_l_t[ib])
                    loss = loss + self.ranking_weight * F.binary_cross_entropy_with_logits(logits, labels)

                if x_u_t is not None and len(x_u_t) >= 2 and self.consistency_weight > 0:
                    iu = torch.randint(0, len(x_u_t), (self.pairs_per_batch,), device=DEVICE)
                    ju = torch.randint(0, len(x_u_t), (self.pairs_per_batch,), device=DEVICE)
                    logits_u = self.model_.rank_logits(x_u_t[iu], x_u_t[ju])
                    with torch.no_grad():
                        p_i = self.model_(x_u_t[iu])
                        p_j = self.model_(x_u_t[ju])
                        target = torch.sigmoid(p_i - p_j)
                    loss = loss + self.consistency_weight * F.binary_cross_entropy_with_logits(logits_u, target)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                xb = _to_tensor(x[start : start + 4096]).to(DEVICE)
                preds.append(self.model_(xb).detach().cpu().numpy())
        return _inverse_y(np.concatenate(preds), self.y_scaler_)


class SCARFRegressorBaseline(BaseEstimator, RegressorMixin):
    """SCARF-style contrastive pretraining followed by supervised regression."""

    def __init__(
        self,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrain_epochs: int = 20,
        epochs: int = 100,
        corruption_rate: float = 0.6,
        temperature: float = 1.0,
        max_unlabeled: int = 5000,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.corruption_rate = corruption_rate
        self.temperature = temperature
        self.max_unlabeled = max_unlabeled
        self.random_state = random_state

    def fit(self, X, y, X_unlabeled=None):
        _set_seed(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32)
        x_ssl = _prepare_unlabeled(x_l, X_unlabeled, self.max_unlabeled, self.random_state)
        self.x_scaler_ = StandardScaler().fit(x_ssl)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_ssl_s = self.x_scaler_.transform(x_ssl).astype(np.float32)
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        self.model_ = _SCARFNet(x_l_s.shape[1], self.hidden_dim, self.projection_dim).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_ssl_t = _to_tensor(x_ssl_s)
        loader = DataLoader(TensorDataset(x_ssl_t), batch_size=self.batch_size, shuffle=True, drop_last=False)
        pool = x_ssl_t.to(DEVICE)
        self.model_.train()
        for _ in range(self.pretrain_epochs):
            for (xb_cpu,) in loader:
                xb = xb_cpu.to(DEVICE)
                x1 = _corrupt_with_pool(xb, pool, self.corruption_rate)
                x2 = _corrupt_with_pool(xb, pool, self.corruption_rate)
                loss = _info_nce(self.model_.project(x1), self.model_.project(x2), self.temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        train_loader = DataLoader(
            TensorDataset(_to_tensor(x_l_s), _to_tensor(y_s)),
            batch_size=min(self.batch_size, max(1, len(x_l_s))),
            shuffle=True,
            drop_last=False,
        )
        for _ in range(self.epochs):
            for xb_cpu, yb_cpu in train_loader:
                xb = xb_cpu.to(DEVICE)
                yb = yb_cpu.to(DEVICE)
                loss = F.mse_loss(self.model_(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                xb = _to_tensor(x[start : start + 4096]).to(DEVICE)
                preds.append(self.model_(xb).detach().cpu().numpy())
        return _inverse_y(np.concatenate(preds), self.y_scaler_)


class LapBoostRegressorBaseline(BaseEstimator, RegressorMixin):
    """Graph-Laplacian boosting baseline for semi-supervised tabular regression."""

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1e-3,
        k_neighbors: int = 10,
        smoothing_steps: int = 20,
        smoothing_strength: float = 0.25,
        anchor_strength: float = 0.65,
        pseudo_ratio: float = 1.0,
        max_unlabeled: int = 5000,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.k_neighbors = k_neighbors
        self.smoothing_steps = smoothing_steps
        self.smoothing_strength = smoothing_strength
        self.anchor_strength = anchor_strength
        self.pseudo_ratio = pseudo_ratio
        self.max_unlabeled = max_unlabeled
        self.random_state = random_state

    def _make_model(self):
        return HistGradientBoostingRegressor(
            max_iter=int(self.max_iter),
            learning_rate=float(self.learning_rate),
            max_leaf_nodes=int(self.max_leaf_nodes),
            l2_regularization=float(self.l2_regularization),
            random_state=int(self.random_state),
        )

    def fit(self, X, y, X_unlabeled=None):
        rng = np.random.default_rng(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        x_u = (
            _as_float_array(X_unlabeled)
            if X_unlabeled is not None and len(X_unlabeled)
            else np.empty((0, x_l.shape[1]), dtype=np.float32)
        )
        if self.max_unlabeled and len(x_u) > self.max_unlabeled:
            idx = rng.choice(len(x_u), size=int(self.max_unlabeled), replace=False)
            x_u = x_u[idx]

        x_fit = np.vstack([x_l, x_u]) if len(x_u) else x_l
        self.x_scaler_ = StandardScaler().fit(x_fit)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_u_s = self.x_scaler_.transform(x_u).astype(np.float32) if len(x_u) else x_u
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        base = self._make_model()
        base.fit(x_l_s, y_s)
        if len(x_u_s) == 0:
            self.model_ = base
            self.pseudo_response_metrics_ = {"pseudo_count": 0, "used_unlabeled": False}
            return self

        x_all = np.vstack([x_l_s, x_u_s]).astype(np.float32)
        preds = base.predict(x_all).astype(np.float32)
        smoothed = preds.copy()
        n_neighbors = min(max(2, int(self.k_neighbors) + 1), len(x_all))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(x_all)
        distances, indices = nn.kneighbors(x_all, return_distance=True)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        sigma = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
        sigma = max(sigma, 1e-6)
        weights = np.exp(-(distances**2) / (2.0 * sigma**2)).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)

        labeled_anchor = np.zeros(len(x_all), dtype=np.float32)
        labeled_anchor[: len(x_l_s)] = y_s
        is_labeled = np.zeros(len(x_all), dtype=bool)
        is_labeled[: len(x_l_s)] = True
        alpha = float(np.clip(self.smoothing_strength, 0.0, 1.0))
        anchor = float(np.clip(self.anchor_strength, 0.0, 1.0))
        for _ in range(max(0, int(self.smoothing_steps))):
            neighbor_mean = np.sum(weights * smoothed[indices], axis=1)
            updated = (1.0 - alpha) * smoothed + alpha * neighbor_mean
            updated[is_labeled] = (1.0 - anchor) * updated[is_labeled] + anchor * labeled_anchor[is_labeled]
            smoothed = updated.astype(np.float32, copy=False)

        pseudo_y = smoothed[len(x_l_s) :]
        local_mean = np.sum(weights[len(x_l_s) :] * smoothed[indices[len(x_l_s) :]], axis=1)
        local_var = np.sum(weights[len(x_l_s) :] * (smoothed[indices[len(x_l_s) :]] - local_mean[:, None]) ** 2, axis=1)
        confidence = 1.0 / (1e-6 + local_var)
        confidence = confidence / max(float(np.mean(confidence)), 1e-8)
        keep = len(x_u_s)
        if self.pseudo_ratio < 1.0:
            keep = max(1, int(np.ceil(len(x_u_s) * float(self.pseudo_ratio))))
        order = np.argsort(-confidence)[:keep]
        sample_weight = np.concatenate(
            [
                np.ones(len(x_l_s), dtype=np.float32),
                np.clip(confidence[order], 0.05, 5.0).astype(np.float32),
            ]
        )
        final_x = np.vstack([x_l_s, x_u_s[order]]).astype(np.float32)
        final_y = np.concatenate([y_s, pseudo_y[order]]).astype(np.float32)
        self.model_ = self._make_model()
        self.model_.fit(final_x, final_y, sample_weight=sample_weight)
        self.pseudo_response_metrics_ = {
            "pseudo_count": int(len(order)),
            "used_unlabeled": True,
            "graph_k": int(self.k_neighbors),
            "smoothing_steps": int(self.smoothing_steps),
            "mean_pseudo_weight": float(np.mean(sample_weight[len(x_l_s) :])),
        }
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        pred_s = np.asarray(self.model_.predict(x), dtype=np.float32).reshape(-1)
        return _inverse_y(pred_s, self.y_scaler_)


class COREGRegressorBaseline(BaseEstimator, RegressorMixin):
    """Co-training KNN regression baseline following the COREG selection idea."""

    def __init__(
        self,
        k1: int = 3,
        k2: int = 5,
        max_iter: int = 20,
        trial_pool_size: int = 64,
        max_unlabeled: int = 1500,
        min_improvement: float = 1e-7,
        random_state: int = 42,
    ):
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.trial_pool_size = trial_pool_size
        self.max_unlabeled = max_unlabeled
        self.min_improvement = min_improvement
        self.random_state = random_state

    def _fit_knn(self, X, y, k):
        n_neighbors = max(1, min(int(k), len(X)))
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", p=2)
        model.fit(X, y)
        return model

    def _best_trial(self, learner, peer, X_train, y_train, X_pool, pool_indices, k):
        if len(X_pool) == 0:
            return None
        n_neighbors = max(1, min(int(k), len(X_train)))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_train)
        base_neighbor_indices = nn.kneighbors(X_pool, return_distance=False)
        base_preds = learner.predict(X_train)
        best = None
        for row_pos, global_idx in enumerate(pool_indices):
            neigh_idx = base_neighbor_indices[row_pos]
            base_loss = float(np.mean((y_train[neigh_idx] - base_preds[neigh_idx]) ** 2))
            pseudo = float(peer.predict(X_pool[row_pos : row_pos + 1])[0])
            x_aug = np.vstack([X_train, X_pool[row_pos : row_pos + 1]])
            y_aug = np.concatenate([y_train, np.array([pseudo], dtype=np.float32)])
            tmp = self._fit_knn(x_aug, y_aug, k)
            new_loss = float(np.mean((y_train[neigh_idx] - tmp.predict(X_train[neigh_idx])) ** 2))
            gain = base_loss - new_loss
            if best is None or gain > best[0]:
                best = (gain, int(global_idx), pseudo)
        if best is None or best[0] <= float(self.min_improvement):
            return None
        return best

    def fit(self, X, y, X_unlabeled=None):
        rng = np.random.default_rng(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        x_u = (
            _as_float_array(X_unlabeled)
            if X_unlabeled is not None and len(X_unlabeled)
            else np.empty((0, x_l.shape[1]), dtype=np.float32)
        )
        if self.max_unlabeled and len(x_u) > self.max_unlabeled:
            x_u = x_u[rng.choice(len(x_u), size=int(self.max_unlabeled), replace=False)]

        x_fit = np.vstack([x_l, x_u]) if len(x_u) else x_l
        self.x_scaler_ = StandardScaler().fit(x_fit)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_u_s = self.x_scaler_.transform(x_u).astype(np.float32) if len(x_u) else x_u
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        X1, y1 = x_l_s.copy(), y_s.copy()
        X2, y2 = x_l_s.copy(), y_s.copy()
        remaining = np.arange(len(x_u_s), dtype=int)
        added = 0
        for _ in range(max(0, int(self.max_iter))):
            learner1 = self._fit_knn(X1, y1, self.k1)
            learner2 = self._fit_knn(X2, y2, self.k2)
            if len(remaining) == 0:
                break
            if len(remaining) > int(self.trial_pool_size):
                pool_indices = rng.choice(remaining, size=int(self.trial_pool_size), replace=False)
            else:
                pool_indices = remaining.copy()
            X_pool = x_u_s[pool_indices]
            trial1 = self._best_trial(learner1, learner2, X1, y1, X_pool, pool_indices, self.k1)
            trial2 = self._best_trial(learner2, learner1, X2, y2, X_pool, pool_indices, self.k2)
            chosen = []
            if trial1 is not None:
                _, idx, pseudo = trial1
                X1 = np.vstack([X1, x_u_s[idx : idx + 1]])
                y1 = np.concatenate([y1, np.array([pseudo], dtype=np.float32)])
                chosen.append(idx)
                added += 1
            if trial2 is not None:
                _, idx, pseudo = trial2
                X2 = np.vstack([X2, x_u_s[idx : idx + 1]])
                y2 = np.concatenate([y2, np.array([pseudo], dtype=np.float32)])
                chosen.append(idx)
                added += 1
            if not chosen:
                break
            remaining = np.setdiff1d(remaining, np.asarray(chosen, dtype=int), assume_unique=False)

        self.model1_ = self._fit_knn(X1, y1, self.k1)
        self.model2_ = self._fit_knn(X2, y2, self.k2)
        self.pseudo_response_metrics_ = {
            "pseudo_count": int(added),
            "used_unlabeled": bool(added > 0),
            "remaining_unlabeled": int(len(remaining)),
        }
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        pred_s = 0.5 * (self.model1_.predict(x) + self.model2_.predict(x))
        return _inverse_y(np.asarray(pred_s, dtype=np.float32).reshape(-1), self.y_scaler_)


class UCVMERegressorBaseline(BaseEstimator, RegressorMixin):
    """Uncertainty-consistency variational model ensemble baseline."""

    def __init__(
        self,
        hidden_dim: int = 128,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        consistency_weight: float = 0.5,
        noise_std: float = 0.05,
        dropout: float = 0.15,
        ensemble_passes: int = 5,
        max_unlabeled: int = 5000,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.consistency_weight = consistency_weight
        self.noise_std = noise_std
        self.dropout = dropout
        self.ensemble_passes = ensemble_passes
        self.max_unlabeled = max_unlabeled
        self.random_state = random_state

    def fit(self, X, y, X_unlabeled=None):
        _set_seed(self.random_state)
        x_l = _as_float_array(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        x_u = (
            _as_float_array(X_unlabeled)
            if X_unlabeled is not None and len(X_unlabeled)
            else np.empty((0, x_l.shape[1]), dtype=np.float32)
        )
        if self.max_unlabeled and len(x_u) > self.max_unlabeled:
            rng = np.random.default_rng(self.random_state)
            x_u = x_u[rng.choice(len(x_u), size=int(self.max_unlabeled), replace=False)]
        x_fit = np.vstack([x_l, x_u]) if len(x_u) else x_l
        self.x_scaler_ = StandardScaler().fit(x_fit)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        x_l_s = self.x_scaler_.transform(x_l).astype(np.float32)
        x_u_s = self.x_scaler_.transform(x_u).astype(np.float32) if len(x_u) else x_u
        y_s = self.y_scaler_.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        self.model_ = _UCVMENet(x_l_s.shape[1], self.hidden_dim, self.dropout).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_l_t = _to_tensor(x_l_s).to(DEVICE)
        y_l_t = _to_tensor(y_s).to(DEVICE)
        x_u_t = _to_tensor(x_u_s).to(DEVICE) if len(x_u_s) else None
        n_l = len(x_l_t)
        batch = min(self.batch_size, max(1, n_l))
        self.model_.train()
        for _ in range(max(1, int(self.epochs))):
            perm = torch.randperm(n_l, device=DEVICE)
            for start in range(0, n_l, batch):
                idx = perm[start : start + batch]
                xb = x_l_t[idx]
                yb = y_l_t[idx]
                pred = self.model_(xb)
                loss = F.mse_loss(pred, yb)
                if x_u_t is not None and self.consistency_weight > 0:
                    iu = torch.randint(0, len(x_u_t), (len(idx),), device=DEVICE)
                    xu = x_u_t[iu]
                    noise1 = torch.randn_like(xu) * float(self.noise_std)
                    noise2 = torch.randn_like(xu) * float(self.noise_std)
                    stochastic = []
                    for _pass in range(max(2, int(self.ensemble_passes))):
                        stochastic.append(self.model_(xu + torch.randn_like(xu) * float(self.noise_std)))
                    stack = torch.stack(stochastic, dim=0)
                    uncertainty = torch.var(stack, dim=0, unbiased=False).detach()
                    weight = torch.exp(-uncertainty).clamp(0.05, 1.0)
                    p1 = self.model_(xu + noise1)
                    p2 = self.model_(xu + noise2)
                    consistency = torch.mean(weight * (p1 - p2.detach()) ** 2)
                    loss = loss + float(self.consistency_weight) * consistency
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.pseudo_response_metrics_ = {
            "pseudo_count": int(len(x_u_s)),
            "used_unlabeled": bool(len(x_u_s) > 0),
            "ensemble_passes": int(self.ensemble_passes),
        }
        return self

    def predict(self, X):
        x = self.x_scaler_.transform(_as_float_array(X)).astype(np.float32)
        preds = []
        self.model_.train()
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                xb = _to_tensor(x[start : start + 4096]).to(DEVICE)
                passes = []
                for _ in range(max(1, int(self.ensemble_passes))):
                    passes.append(self.model_(xb).detach().cpu().numpy())
                preds.append(np.mean(np.stack(passes, axis=0), axis=0))
        self.model_.eval()
        return _inverse_y(np.concatenate(preds).reshape(-1), self.y_scaler_)
