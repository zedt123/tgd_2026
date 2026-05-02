from abc import ABC, abstractmethod

from src.registry import PROPOSAL_SCHEDULES

class ProposalSchedule(ABC):
    """
    Base class for standard deviation schedules for the approximation
    (proposal) of p(y|xt).
    """
    def __init__(self, sigma: float, strength: float):
        self.sigma = sigma
        self.alpha = strength

    @abstractmethod
    def __call__(self, t: float) -> float:
        raise NotImplementedError

@PROPOSAL_SCHEDULES.register('time_proposal')
class TimeProposal(ProposalSchedule):
    """
    Proposal std is the diffusion time.
    """
    def __init__(
        self,
        sigma: float,
        strength: float):
        super().__init__(sigma, strength)

    def __call__(self, t: float) -> float:
        return self.alpha * t + self.sigma