from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.distributions as D

from src.models.base import DiffusionModel


class MixtureOfGaussians(nn.Module):
    """
    Thin wrapper around torch.distributions.MixtureSameFamily for a MoG.

    Assumptions:
      x: [B, D]
      weights: [K]
      means: [K, D]
      covs: [K, D, D]
    """
    def __init__(self, weights, means, covs, device: str = "cpu"):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, device=device, dtype=torch.float32))
        self.register_buffer("means",   torch.as_tensor(means,   device=device, dtype=torch.float32))
        self.register_buffer("covs",    torch.as_tensor(covs,    device=device, dtype=torch.float32))

        # normalize weights once
        w = torch.clamp(self.weights, min=0.0)
        self.weights = w / (w.sum() + 1e-12)

    def dist(self, covs: torch.Tensor | None = None) -> D.Distribution:
        """
        Build MixtureSameFamily distribution.
        If covs is provided, uses that covariance (e.g. inflated by sigma^2 I).
        """
        covs = self.covs if covs is None else covs
        mix = D.Categorical(probs=self.weights)
        comp = D.MultivariateNormal(loc=self.means, covariance_matrix=covs)
        return D.MixtureSameFamily(mix, comp)

    @torch.no_grad()
    def score(self, x: torch.Tensor, covs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Fast analytic score for a MoG:
          score(x) = sum_k r_k(x) * (-Sigma_k^{-1}(x - mu_k))
        where r_k are responsibilities proportional to w_k * N_k(x).
        """
        covs = self.covs if covs is None else covs

        # Ensure float type/device consistency
        x = x.to(device=self.means.device, dtype=self.means.dtype)
        covs = covs.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        means = self.means.to(device=x.device, dtype=x.dtype)

        _, Ddim = x.shape

        # diff: [B, K, D]
        diff = x[:, None, :] - means[None, :, :]

        # Cholesky per component: [K, D, D]
        L = torch.linalg.cholesky(covs)

        # Compute Sigma^{-1} diff via cholesky_solve
        # rhs: [B, K, D, 1] -> result: [B, K, D, 1]
        rhs = diff.unsqueeze(-1)
        # torch.cholesky_solve expects (..., D, 1) with L as (..., D, D)
        # We need to broadcast L over B, so expand:
        Linv_diff = torch.cholesky_solve(rhs, L.unsqueeze(0))  # [B, K, D, 1]
        Linv_diff = Linv_diff.squeeze(-1)                     # [B, K, D]

        # score per component: -Sigma^{-1}(x - mu)
        score_k = -Linv_diff  # [B, K, D]

        # log N_k(x): use quadratic form + logdet
        quad = (diff * Linv_diff).sum(dim=-1)  # [B, K]
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # [K]
        log_norm = -0.5 * (Ddim * math.log(2.0 * math.pi) + logdet)  # [K]
        logp = log_norm[None, :] - 0.5 * quad  # [B, K]

        # responsibilities r_k ∝ w_k * exp(logp_k)
        logw = torch.log(weights + 1e-30)  # [K]
        logits = logw[None, :] + logp      # [B, K]
        r = torch.softmax(logits, dim=1)   # [B, K]

        # weighted sum of scores
        return (r.unsqueeze(-1) * score_k).sum(dim=1)  # [B, D]


class AnalyticDiffusionModel(DiffusionModel):
    """
    VE/EDM analytic diffusion where clean p(x0) is a MoG.

    Under additive Gaussian noise:
      p(x_t) = sum_k w_k N(x_t | mu_k, Sigma_k + sigma^2 I)

    So score(x_t, sigma) is the MoG score with inflated covariances.
    """
    domain = "toy2d"

    def __init__(
        self,
        cfg,
        device: str = "cpu",
        requires_grad: bool = False,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        mog_cfg = getattr(cfg, "mog")

        # If mog_cfg provides N, we synthesize an MoG.
        if hasattr(mog_cfg, "N") and getattr(mog_cfg, "N") is not None:
            N = int(getattr(mog_cfg, "N"))
            if N <= 0:
                raise ValueError(f"mog_cfg.N must be > 0, got {N}")

            tau = float(getattr(mog_cfg, "tau", 1.0))
            if tau <= 0:
                raise ValueError(f"mog_cfg.tau must be > 0, got {tau}")

            # Shrink mean support to avoid being too close to boundary.
            mean_margin = float(getattr(mog_cfg, "mean_margin", 0.0))
            if not (0.0 <= mean_margin < 1.0):
                raise ValueError(f"mog_cfg.mean_margin must be in [0, 1), got {mean_margin}")

            Ddim = 2
            low, high = -1.0 + mean_margin, 1.0 - mean_margin

            means = torch.empty((N, Ddim), device=device, dtype=torch.float32)
            means.uniform_(low, high, generator=generator)

            weights = torch.full((N,), 1.0 / N, device=device, dtype=torch.float32)

            I = torch.eye(Ddim, device=device, dtype=torch.float32)
            covs = (tau * tau) * I.unsqueeze(0).repeat(N, 1, 1)  # [N,2,2]

            self.mog = MixtureOfGaussians(
                weights=weights,
                means=means,
                covs=covs,
                device=device,
            ).to(device)

        else:
            # Backwards-compatible path: explicit weights/means/covs
            self.mog = MixtureOfGaussians(
                weights=getattr(mog_cfg, "weights"),
                means=getattr(mog_cfg, "means"),
                covs=getattr(mog_cfg, "covs"),
                device=device,
            ).to(device)

        self.eval()
        self.requires_grad_(requires_grad)

    def score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sig2 = sigma * sigma

        Ddim = x.shape[-1]
        I = torch.eye(Ddim, device=x.device, dtype=x.dtype)

        covs_t = self.mog.covs.to(device=x.device, dtype=x.dtype) + sig2 * I  # [K,D,D]
        return self.mog.score(x, covs=covs_t)

    def get_in_shape(self):
        return (int(self.mog.means.shape[1]),)
    
    def gt_sample(self, N: int):
        return self.mog.dist().sample((N,))