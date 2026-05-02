import torch

from src.registry import DATASET
from .base import DiffusionData
from .utils2d import sample_posterior_px_given_y_abs_mog

@DATASET.register('toy2d')
class Toy2dDataset(DiffusionData):
    """
    Generate 2D toy data for experiments.
    Supports general (possibly non-diagonal) covariances.
    Exact p(x|y) sampling is only available when covariances are diagonal.
    """
    def __init__(self, model, N):
        self.data = model.gt_sample(N)  # expect (N,2) torch tensor

        # MoG parameters (torch tensors)
        self.weights = model.mog.weights          # (K,)
        self.means   = model.mog.means            # (K,2)
        self.covs    = model.mog.covs             # (K,2,2) can be non-diagonal

        # Basic checks
        K = self.means.shape[0]
        assert self.means.shape == (K, 2), f"means must be (K,2), got {self.means.shape}"
        assert self.covs.shape == (K, 2, 2), f"covs must be (K,2,2), got {self.covs.shape}"
        assert self.weights.shape == (K,), f"weights must be (K,), got {self.weights.shape}"

        assert isinstance(self.data, torch.Tensor) and self.data.ndim == 2 and self.data.shape[1] == 2, \
            f"data must be (N,2) torch tensor, got {type(self.data)} {getattr(self.data, 'shape', None)}"

        self.d = self.data.shape[1]  # 2

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.data.shape[0]

    def get_shape(self):
        return (self.d,)

    @torch.no_grad()
    def sample_px_given_y(self, y, sigma, generator, N=100000, *, eps=1e-7, diag_tol=1e-8):
        """
        Sample exactly from p(X|Y=y) for the absolute value forward operator.
        Requires diagonal covariances. Note that this is different from p(X|Y).
        To sample from p(X|Y) we just use the forward operator on self.data).
        """
        # Ensure y is tensor on same device/dtype as parameters
        y = torch.as_tensor(y, dtype=self.means.dtype, device=self.means.device).view(-1)
        assert y.numel() == 2, f"y must be shape (2,), got {tuple(y.shape)}"

        # Check diagonal covariances only here
        diag = self.covs.diagonal(dim1=-2, dim2=-1)              # (K,2)
        offdiag = self.covs - torch.diag_embed(diag)             # (K,2,2)
        max_offdiag = offdiag.abs().max().item()
        assert max_offdiag <= diag_tol, (
            f"Exact sampling requires diagonal covs; max |offdiag|={max_offdiag} > tol={diag_tol}"
        )

        samples, indices = sample_posterior_px_given_y_abs_mog(
            y=y,
            means=self.means,
            variances=diag,                 # use diag as variances
            weights=self.weights,
            sigma=float(sigma),
            n_samples=N,
            eps=eps,
            generator=generator,
        )
        return samples, indices
    
    @torch.no_grad()
    def build_ground_truth_conditionals(self, y, operator, rng_cpu, N):
        gt_list = []
        for i in range(y.shape[0]):
            gt_samples_i, _ = self.sample_px_given_y(
                y[i],
                sigma=operator.sigma,
                generator=rng_cpu,
                N=N,
            )
            gt_list.append(gt_samples_i)
        return torch.stack(gt_list, dim=0)