from dataclasses import dataclass
import torch


@dataclass
class RNG:
    """Independent RNG streams for different randomness sources."""
    init: torch.Generator       # initial noise x_T
    sde: torch.Generator        # SDE solver noise
    mcmc: torch.Generator       # x0_modifier noise (e.g., Langevin/MH)
    smc: torch.Generator        # resampling / proposal randomness
    operator: torch.Generator   # operator randomness (e.g., random masks, measurement noise)
    forward: torch.Generator    # forward process in sampler
    cpu: torch.Generator        # CPU-only randomness (toy2d, misc)
    model: torch.Generator      # model rng (e.g. means of mixtures)

def make_rng(seed: int, device: str = "cuda") -> RNG:
    """
    Create independent torch.Generator streams derived from a single seed.

    Separate streams make runs more stable under code changes: e.g., changing the
    number of MCMC steps won't change the initial noise stream.
    """
    def g(offset: int, dev: str) -> torch.Generator:
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + offset)
        return gen

    return RNG(
        init=g(0, device),
        sde=g(1, device),
        mcmc=g(2, device),
        smc=g(3, 'cpu'),
        operator=g(4, device),
        forward=g(5, device),
        cpu=g(6, 'cpu'),
        model=g(5, device),
    )
