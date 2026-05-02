import torch

def log_ess_from_logw(logw: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log effective sample size (log ESS) from **normalised** log-weights.
    Args:
        logw: Tensor of shape (B, N) (or any shape; ESS computed over `dim`).
              Contains **normalised** log weights.
        dim: Dimension over which to compute ESS (default: last dim).

    Returns:
        Tensor of shape logw.shape with `dim` removed (e.g. (B,)).
    """
    return  - torch.logsumexp(2.0 * logw, dim=dim)