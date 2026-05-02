from ..types import DistributionMetric
from src.registry import DIST_METRICS

import torch
from ot.sliced import max_sliced_wasserstein_distance

@DIST_METRICS.register("swd")
class SlicedWD(DistributionMetric):
    """
    Sliced Wassertein Distance

    Contract
    --------
    compute(ref, gen) -> float
    """
    name = "swd"

    def __init__(self, n_projections: int = 100) -> None:
        self.n_projections = n_projections

    @torch.no_grad()
    def compute(
        self,
        ref: torch.Tensor,
        gen: torch.Tensor,
        *,
        device: str,
    ) -> float:
        
        ref = ref.to(device)
        gen = gen.to(device)

        dist = max_sliced_wasserstein_distance(
            X_s=ref,
            X_t=gen,
            n_projections=self.n_projections
        )
        return dist.detach().cpu().item()
    
    @torch.no_grad()
    def compute_batched(self, ref: torch.Tensor, gen: torch.Tensor, *, device: str) -> torch.Tensor:
        # ref: (B, N, D), gen: (B, M, D)
        ref = ref.to(device)
        gen = gen.to(device)

        if ref.ndim != 3 or gen.ndim != 3:
            raise ValueError("compute_batched expects (B,N,D) and (B,M,D)")
        if ref.shape[0] != gen.shape[0]:
            raise ValueError("Batch size mismatch")

        B = ref.shape[0]
        out = torch.empty((B,), device="cpu") 

        for b in range(B):
            dist = max_sliced_wasserstein_distance(
                X_s=ref[b],
                X_t=gen[b],
                n_projections=self.n_projections,
            )
            out[b] = dist.detach().cpu()

        return out

@DIST_METRICS.register("quad_score")
class QuadrantConditionalMixtureScore(DistributionMetric):
    """
    Conditional quadrant mixture score for p(x | y) when
    y = |x| + noise (componentwise).

    The key idea:
      - For a fixed y, the conditional distribution may occupy only a subset
        of the 4 quadrants.
      - We detect the active quadrants from the ground-truth conditional samples.
      - We compare generated samples to ground truth only on those active quadrants.

    Contract
    --------
    compute(ref, gen, device=...) -> float

    Inputs
    ------
    ref : (N, 2) tensor
        Ground-truth conditional samples from p(x | y).
    gen : (M, 2) tensor
        Generated conditional samples from the algorithm being evaluated.
    """

    name = "quad_cond_score"

    def __init__(
        self,
        *,
        w_mean: float = 1.0,
        w_cov: float = 1.0,
        w_weight: float = 0.2,
        cov_metric: str = "fro",                 # "fro" or "bures"
        aggregate: str = "gt_weighted_active",   # "gt_weighted_active" or "uniform_active"
        min_gt_weight: float = 1e-3,             # quadrant active if gt weight >= threshold
        min_gt_count: int = 10,                  # or gt count >= threshold
        empty_quadrant_penalty: float = 1e3,     # if gen misses an active GT quadrant
        eps: float = 1e-6,
    ) -> None:
        self.w_mean = w_mean
        self.w_cov = w_cov
        self.w_weight = w_weight
        self.cov_metric = cov_metric
        self.aggregate = aggregate
        self.min_gt_weight = min_gt_weight
        self.min_gt_count = min_gt_count
        self.empty_quadrant_penalty = empty_quadrant_penalty
        self.eps = eps

    @staticmethod
    @torch.no_grad()
    def quadrant_stats(X: torch.Tensor, eps: float = 1e-6):
        assert X.ndim == 2 and X.shape[1] == 2, "X must be (N,2)"
        N = X.shape[0]
        x1, x2 = X[:, 0], X[:, 1]

        masks = {
            "++": (x1 > 0) & (x2 > 0),
            "+-": (x1 > 0) & (x2 < 0),
            "-+": (x1 < 0) & (x2 > 0),
            "--": (x1 < 0) & (x2 < 0),
        }

        out = {}
        for k, m in masks.items():
            Xk = X[m]
            nk = Xk.shape[0]
            wk = nk / max(N, 1)

            if nk >= 2:
                muk = Xk.mean(dim=0)
                Z = Xk - muk
                covk = (Z.T @ Z) / nk
            elif nk == 1:
                muk = Xk[0]
                covk = torch.zeros((2, 2), device=X.device, dtype=X.dtype)
            else:
                muk = torch.full((2,), float("nan"), device=X.device, dtype=X.dtype)
                covk = torch.full((2, 2), float("nan"), device=X.device, dtype=X.dtype)

            if torch.isfinite(covk).all():
                covk = covk + eps * torch.eye(2, device=X.device, dtype=X.dtype)

            out[k] = {
                "weight": float(wk),
                "mean": muk,
                "cov": covk,
                "count": int(nk),
            }
        return out

    @staticmethod
    @torch.no_grad()
    def _cov_dist(C: torch.Tensor, Cg: torch.Tensor, cov_metric: str) -> torch.Tensor:
        if cov_metric == "fro":
            return torch.linalg.norm(C - Cg, ord="fro")

        if cov_metric == "bures":
            def sqrtm_psd(A):
                A = 0.5 * (A + A.T)
                w, V = torch.linalg.eigh(A)
                w = torch.clamp(w, min=1e-12)
                return (V * w.sqrt().unsqueeze(0)) @ V.T

            Cg_sqrt = sqrtm_psd(Cg)
            mid = Cg_sqrt @ C @ Cg_sqrt
            mid_sqrt = sqrtm_psd(mid)
            d2 = torch.trace(C) + torch.trace(Cg) - 2.0 * torch.trace(mid_sqrt)
            return torch.sqrt(torch.clamp(d2, min=0.0))

        raise ValueError("cov_metric must be 'fro' or 'bures'")

    @staticmethod
    @torch.no_grad()
    def conditional_quadrant_score(
        X: torch.Tensor,
        Xgt: torch.Tensor,
        *,
        w_mean: float = 1.0,
        w_cov: float = 1.0,
        w_weight: float = 0.2,
        cov_metric: str = "fro",
        aggregate: str = "gt_weighted_active",
        min_gt_weight: float = 1e-3,
        min_gt_count: int = 10,
        empty_quadrant_penalty: float = 1e3,
        eps: float = 1e-6,
    ):
        stats = QuadrantConditionalMixtureScore.quadrant_stats(X, eps=eps)
        gt = QuadrantConditionalMixtureScore.quadrant_stats(Xgt, eps=eps)

        quadrants = ["++", "+-", "-+", "--"]

        # Decide which quadrants are actually present in the GT conditional distribution.
        active = []
        inactive = []
        for k in quadrants:
            is_active = (
                gt[k]["count"] >= min_gt_count
                and gt[k]["weight"] >= min_gt_weight
            )
            if is_active:
                active.append(k)
            else:
                inactive.append(k)

        # Safety fallback: if thresholds are too strict, keep the single largest GT quadrant.
        if len(active) == 0:
            k_best = max(quadrants, key=lambda q: gt[q]["weight"])
            active = [k_best]
            inactive = [q for q in quadrants if q != k_best]

        per_quad = {}
        mean_errs, cov_errs, weight_errs, quad_scores = {}, {}, {}, {}

        for k in quadrants:
            wk = stats[k]["weight"]
            wkg = gt[k]["weight"]
            mk, mkg = stats[k]["mean"], gt[k]["mean"]
            Ck, Ckg = stats[k]["cov"], gt[k]["cov"]

            is_active = k in active

            if not is_active:
                per_quad[k] = {
                    "active": False,
                    "mean_err": 0.0,
                    "cov_err": 0.0,
                    "weight_err": 0.0,
                    "quad_score": 0.0,
                    "w": wk,
                    "w_gt": wkg,
                    "count": stats[k]["count"],
                    "count_gt": gt[k]["count"],
                }
                continue

            # Active GT quadrant: generator should match it.
            gen_missing = (not torch.isfinite(mk).all()) or (not torch.isfinite(Ck).all())

            if gen_missing:
                mean_err = torch.tensor(empty_quadrant_penalty, device=X.device, dtype=X.dtype)
                cov_err = torch.tensor(empty_quadrant_penalty, device=X.device, dtype=X.dtype)
            else:
                mean_err = torch.linalg.norm(mk - mkg, ord=2)
                cov_err = QuadrantConditionalMixtureScore._cov_dist(Ck, Ckg, cov_metric)

            weight_err = torch.tensor(abs(wk - wkg), device=X.device, dtype=X.dtype)
            quad_score = w_mean * mean_err + w_cov * cov_err + w_weight * weight_err

            per_quad[k] = {
                "active": True,
                "mean_err": float(mean_err),
                "cov_err": float(cov_err),
                "weight_err": float(weight_err),
                "quad_score": float(quad_score),
                "w": wk,
                "w_gt": wkg,
                "count": stats[k]["count"],
                "count_gt": gt[k]["count"],
            }

            mean_errs[k] = float(mean_err)
            cov_errs[k] = float(cov_err)
            weight_errs[k] = float(weight_err)
            quad_scores[k] = float(quad_score)

        # Active-only aggregation weights
        gt_w_active = {k: gt[k]["weight"] for k in active}
        gt_w_sum = sum(gt_w_active.values())
        gt_w_active = {k: v / max(gt_w_sum, 1e-12) for k, v in gt_w_active.items()}

        uni_w_active = {k: 1.0 / len(active) for k in active}

        def weighted_avg(values_dict, weights_dict):
            return sum(values_dict[k] * weights_dict[k] for k in weights_dict.keys())

        overall = {
            "uniform_active": {
                "mean_err": weighted_avg(mean_errs, uni_w_active),
                "cov_err": weighted_avg(cov_errs, uni_w_active),
                "weight_err": weighted_avg(weight_errs, uni_w_active),
                "quad_score": weighted_avg(quad_scores, uni_w_active),
            },
            "gt_weighted_active": {
                "mean_err": weighted_avg(mean_errs, gt_w_active),
                "cov_err": weighted_avg(cov_errs, gt_w_active),
                "weight_err": weighted_avg(weight_errs, gt_w_active),
                "quad_score": weighted_avg(quad_scores, gt_w_active),
            },
        }

        if aggregate == "gt_weighted_active":
            score = overall["gt_weighted_active"]["quad_score"]
        elif aggregate == "uniform_active":
            score = overall["uniform_active"]["quad_score"]
        else:
            raise ValueError("aggregate must be 'gt_weighted_active' or 'uniform_active'")

        details = {
            "score": float(score),
            "overall": overall,
            "per_quadrant": per_quad,
            "active_quadrants": active,
            "inactive_quadrants": inactive,
            "settings": {
                "w_mean": w_mean,
                "w_cov": w_cov,
                "w_weight": w_weight,
                "cov_metric": cov_metric,
                "aggregate": aggregate,
                "min_gt_weight": min_gt_weight,
                "min_gt_count": min_gt_count,
                "empty_quadrant_penalty": empty_quadrant_penalty,
            },
        }
        return float(score), details

    @torch.no_grad()
    def compute(
        self,
        ref: torch.Tensor,
        gen: torch.Tensor,
        *,
        device: str,
    ) -> float:
        ref = ref.to(device)
        gen = gen.to(device)

        score, _details = self.conditional_quadrant_score(
            X=gen,
            Xgt=ref,
            w_mean=self.w_mean,
            w_cov=self.w_cov,
            w_weight=self.w_weight,
            cov_metric=self.cov_metric,
            aggregate=self.aggregate,
            min_gt_weight=self.min_gt_weight,
            min_gt_count=self.min_gt_count,
            empty_quadrant_penalty=self.empty_quadrant_penalty,
            eps=self.eps,
        )
        return float(score)

# @DIST_METRICS.register("quad_score")
# class QuadrantMixtureScore(DistributionMetric):
#     """
#     Quadrant Mixture Score

#     Contract
#     --------
#     compute(ref, gen) -> float
#     """
#     name = "quad_score"

#     def __init__(
#         self,
#         *,
#         w_mean: float = 1.0,
#         w_cov: float = 1.0,
#         w_weight: float = 0.2,
#         cov_metric: str = "fro",          # "fro" or "bures"
#         aggregate: str = "gt_weighted",   # "gt_weighted" or "uniform"
#         eps: float = 1e-6,
#     ) -> None:
#         self.w_mean = w_mean
#         self.w_cov = w_cov
#         self.w_weight = w_weight
#         self.cov_metric = cov_metric
#         self.aggregate = aggregate
#         self.eps = eps

#     @staticmethod
#     @torch.no_grad()
#     def quadrant_stats(X: torch.Tensor, eps: float = 1e-6):
#         assert X.ndim == 2 and X.shape[1] == 2, "X must be (N,2)"
#         N = X.shape[0]
#         x1, x2 = X[:, 0], X[:, 1]

#         masks = {
#             "++": (x1 > 0) & (x2 > 0),
#             "+-": (x1 > 0) & (x2 < 0),
#             "-+": (x1 < 0) & (x2 > 0),
#             "--": (x1 < 0) & (x2 < 0),
#         }

#         out = {}
#         for k, m in masks.items():
#             Xk = X[m]
#             nk = Xk.shape[0]
#             wk = nk / max(N, 1)

#             if nk >= 2:
#                 muk = Xk.mean(dim=0)
#                 Z = Xk - muk
#                 covk = (Z.T @ Z) / nk
#             elif nk == 1:
#                 muk = Xk[0]
#                 covk = torch.zeros((2, 2), device=X.device, dtype=X.dtype)
#             else:
#                 muk = torch.full((2,), float("nan"), device=X.device, dtype=X.dtype)
#                 covk = torch.full((2, 2), float("nan"), device=X.device, dtype=X.dtype)

#             if torch.isfinite(covk).all():
#                 covk = covk + eps * torch.eye(2, device=X.device, dtype=X.dtype)

#             out[k] = {"weight": float(wk), "mean": muk, "cov": covk, "count": int(nk)}
#         return out

#     @staticmethod
#     @torch.no_grad()
#     def quadrant_mixture_score(
#         X: torch.Tensor,
#         Xgt: torch.Tensor,
#         *,
#         w_mean: float = 1.0,
#         w_cov: float = 1.0,
#         w_weight: float = 0.2,
#         cov_metric: str = "fro",
#         aggregate: str = "gt_weighted",
#         eps: float = 1e-6,
#     ):
#         stats = QuadrantMixtureScore.quadrant_stats(X, eps=eps)
#         gt = QuadrantMixtureScore.quadrant_stats(Xgt, eps=eps)

#         def cov_dist(C, Cg):
#             if cov_metric == "fro":
#                 return torch.linalg.norm(C - Cg, ord="fro")
#             elif cov_metric == "bures":
#                 def sqrtm_psd(A):
#                     A = 0.5 * (A + A.T)
#                     w, V = torch.linalg.eigh(A)
#                     w = torch.clamp(w, min=1e-12)
#                     return (V * w.sqrt().unsqueeze(0)) @ V.T

#                 Cg_sqrt = sqrtm_psd(Cg)
#                 mid = Cg_sqrt @ C @ Cg_sqrt
#                 mid_sqrt = sqrtm_psd(mid)
#                 d2 = torch.trace(C) + torch.trace(Cg) - 2.0 * torch.trace(mid_sqrt)
#                 return torch.sqrt(torch.clamp(d2, min=0.0))
#             else:
#                 raise ValueError("cov_metric must be 'fro' or 'bures'")

#         gt_w = {k: gt[k]["weight"] for k in ["++", "+-", "-+", "--"]}
#         uni_w = {k: 0.25 for k in ["++", "+-", "-+", "--"]}

#         per_quad = {}
#         mean_errs, cov_errs, weight_errs, quad_scores = {}, {}, {}, {}

#         for k in ["++", "+-", "-+", "--"]:
#             wk = stats[k]["weight"]
#             wkg = gt[k]["weight"]
#             mk, mkg = stats[k]["mean"], gt[k]["mean"]
#             Ck, Ckg = stats[k]["cov"], gt[k]["cov"]

#             if not torch.isfinite(mk).all() or not torch.isfinite(Ck).all():
#                 mean_err = torch.tensor(1e3, device=X.device)
#                 cov_err = torch.tensor(1e3, device=X.device)
#             else:
#                 mean_err = torch.linalg.norm(mk - mkg, ord=2)
#                 cov_err = cov_dist(Ck, Ckg)

#             weight_err = abs(wk - wkg)
#             quad_score = w_mean * mean_err + w_cov * cov_err + w_weight * weight_err

#             per_quad[k] = {
#                 "mean_err": float(mean_err),
#                 "cov_err": float(cov_err),
#                 "weight_err": float(weight_err),
#                 "quad_score": float(quad_score),
#                 "w": wk,
#                 "w_gt": wkg,
#                 "count": stats[k]["count"],
#                 "count_gt": gt[k]["count"],
#             }

#             mean_errs[k] = float(mean_err)
#             cov_errs[k] = float(cov_err)
#             weight_errs[k] = float(weight_err)
#             quad_scores[k] = float(quad_score)

#         def weighted_avg(values_dict, weights_dict):
#             return sum(values_dict[k] * weights_dict[k] for k in values_dict.keys())

#         overall = {
#             "uniform": {
#                 "mean_err": weighted_avg(mean_errs, uni_w),
#                 "cov_err": weighted_avg(cov_errs, uni_w),
#                 "weight_err": weighted_avg(weight_errs, uni_w),
#                 "quad_score": weighted_avg(quad_scores, uni_w),
#             },
#             "gt_weighted": {
#                 "mean_err": weighted_avg(mean_errs, gt_w),
#                 "cov_err": weighted_avg(cov_errs, gt_w),
#                 "weight_err": weighted_avg(weight_errs, gt_w),
#                 "quad_score": weighted_avg(quad_scores, gt_w),
#             },
#         }

#         if aggregate == "gt_weighted":
#             score = overall["gt_weighted"]["quad_score"]
#         elif aggregate == "uniform":
#             score = overall["uniform"]["quad_score"]
#         else:
#             raise ValueError("aggregate must be 'gt_weighted' or 'uniform'")

#         details = {
#             "score": float(score),
#             "overall": overall,
#             "per_quadrant": per_quad,
#             "settings": {
#                 "w_mean": w_mean,
#                 "w_cov": w_cov,
#                 "w_weight": w_weight,
#                 "cov_metric": cov_metric,
#                 "aggregate": aggregate,
#             },
#         }
#         return float(score), details

#     @torch.no_grad()
#     def compute(
#         self,
#         ref: torch.Tensor,
#         gen: torch.Tensor,
#         *,
#         device: str,
#     ) -> float:
#         ref = ref.to(device)
#         gen = gen.to(device)

#         score, _details = self.quadrant_mixture_score(
#             X=gen,
#             Xgt=ref,
#             w_mean=self.w_mean,
#             w_cov=self.w_cov,
#             w_weight=self.w_weight,
#             cov_metric=self.cov_metric,
#             aggregate=self.aggregate,
#             eps=self.eps,
#         )
#         return float(score)