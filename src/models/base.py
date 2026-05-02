from __future__ import annotations

import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """
    Base diffusion model interface for this repo.

    The sampler interacts with models through either:
      - score(x, sigma):  ∇_x log p(x; sigma)
      - predict_x0(x, sigma): x0 estimate (a.k.a. Tweedie denoiser)

    You must override at least one of {score, predict_x0}. The other is provided
    via Tweedie's relationship:

        score(x, sigma)      = (predict_x0(x, sigma) - x) / sigma^2
        predict_x0(x, sigma) = x + score(x, sigma) * sigma^2

    Assumption:
      This relationship is valid when (x | x0, sigma) is additive Gaussian noise
      with standard deviation 'sigma' (VE/EDM-style). If you use a different
      parameterization, wrap the model (e.g., VPPrecond) so that the model
      conforms to this interface.
    """
    domain: str = "generic"  # e.g. "image", "toy2d", "audio", "molecule"

    def __init__(self):
        super().__init__()
        # Require at least one of score / predict_x0 to be overridden
        if (type(self).score is DiffusionModel.score) and (type(self).predict_x0 is DiffusionModel.predict_x0):
            raise NotImplementedError("Override either 'score' or 'predict_x0'.")

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """
        Standard nn.Module forward: returns x0 prediction by default.

        Args:
            x: Noisy input at time/noise level, shape [B, ...]
            sigma: Noise level (float or tensor broadcastable to x)
        """
        return self.predict_x0(x, sigma)

    def score(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """
        Compute score ∇_x log p(x; sigma).

        Default derives score from predict_x0 via Tweedie's formula.
        Override if your wrapped model provides score directly.
        """
        x0 = self.predict_x0(x, sigma)
        return (x0 - x) / (sigma ** 2)

    def predict_x0(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """
        Predict clean data x0 given noisy x at noise level sigma.

        Default derives x0 from score via Tweedie's formula.
        Override if your wrapped model provides x0 directly.
        """
        s = self.score(x, sigma)
        return x + s * (sigma ** 2)

    def get_in_shape(self):
        """Return the expected input data shape (excluding batch)."""
        raise NotImplementedError