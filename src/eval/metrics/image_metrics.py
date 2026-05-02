# src/eval/metrics/image_metrics.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..types import EvalBatch, PairedMetric, DistributionMetric
from src.registry import PAIRED_METRICS, DIST_METRICS

def _to01(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0:
        x = (x + 1) / 2
    return x.clamp(0, 1)

# -------------------------
# Paired image metrics
# -------------------------

@PAIRED_METRICS.register("psnr")
class PSNRMetric(PairedMetric):
    """
    PSNR over images.

    Assumptions
    -----------
    - batch.gt_flat and batch.gen_flat are (B*K, C, H, W)
    - values are already in [0, 1]
    """
    name = "psnr"
    cmp = "max"

    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = data_range
        from piq import psnr  
        self._psnr = psnr

    @torch.no_grad()
    def compute(
        self,
        batch: EvalBatch,
        *,
        device: str,        
    ) -> torch.Tensor:
        if batch.gt_flat is None:
            raise ValueError("PSNRMetric requires batch.gt_flat")
        gt = _to01(batch.gt_flat)
        gen = _to01(batch.gen_flat)
        v = self._psnr(gt, gen, data_range=self.data_range, reduction="none")
        v = v.reshape(batch.B, batch.K)  # (B,K)
        return v

@PAIRED_METRICS.register("ssim")
class SSIMMetric(PairedMetric):
    """
    SSIM over images.

    Assumptions
    -----------
    - batch.gt_flat and batch.gen_flat are (B*K, C, H, W)
    - values are already in [0, 1]
    """
    name = "ssim"
    cmp = "max"

    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = data_range
        from piq import ssim 
        self._ssim = ssim

    @torch.no_grad()
    def compute(
        self,
        batch: EvalBatch,
        *,
        device: str,        
    ) -> torch.Tensor:
        if batch.gt_flat is None:
            raise ValueError("SSIMMetric requires batch.gt_flat")
        gt = _to01(batch.gt_flat)
        gen = _to01(batch.gen_flat)
        v = self._ssim(gt, gen, data_range=self.data_range, reduction="none")
        v = v.reshape(batch.B, batch.K)  # (B,K)
        return v

@PAIRED_METRICS.register("lpips")
class LPIPSMetric(PairedMetric):
    """
    LPIPS over images.

    Assumptions
    -----------
    - batch.gt_flat and batch.gen_flat are (B*K, C, H, W)
    - values are already in [0, 1]
    """
    name = "lpips"
    cmp = "min"

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        from piq import LPIPS
        # Gives a warning because piq uses pretrained which is depricated.
        self._lpips = LPIPS(replace_pooling=True, reduction="none")

    @torch.no_grad()
    def compute(
        self,
        batch: EvalBatch,
        *,
        device: str,
    ) -> torch.Tensor:
        if batch.gt_flat is None:
            raise ValueError("LPIPSMetric requires batch.gt_flat")

        if next(self._lpips.parameters()).device != device:
            self._lpips.to(device)

        bs = self.batch_size

        gt = _to01(batch.gt_flat)
        gen = _to01(batch.gen_flat)

        outs = []
        for i in range(0, gt.shape[0], bs):
            o = self._lpips(gt[i : i + bs], gen[i : i + bs])
            outs.append(o)

        v = torch.cat(outs, dim=0)

        # piq LPIPS may return (N,1,1,1) or (N,)
        v = v.reshape(v.shape[0], -1).mean(dim=1)  # (N,)
        v = v.reshape(batch.B, batch.K)            # (B,K)
        return v

@PAIRED_METRICS.register("consistency_mse")
class ConsistencyMSE(PairedMetric):
    """
    Consistency / faithfulness metric in observation space:
        mse( y , operator.forward(x) )

    Assumptions
    -----------
    - operator.forward accepts (B*K, ...) and returns (B*K, ...) in same space as y_flat
    - y_flat exists and matches operator output shape (up to broadcasting in trailing dims)
    """
    name = "consistency_mse"
    cmp = "min"

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    @torch.no_grad()
    def compute(
        self,
        batch: EvalBatch,
        *,
        device: str,
    ) -> torch.Tensor:
        if batch.operator is None:
            raise ValueError("ConsistencyMSE requires batch.operator")
        if batch.y_flat is None:
            raise ValueError("ConsistencyMSE requires batch.y_flat")

        y = batch.y_flat
        y_hat = batch.operator.forward(batch.gen_flat)

        diff2 = (y_hat - y) ** 2
        if diff2.ndim == 1:
            v = diff2
        else:
            dims = tuple(range(1, diff2.ndim))
            v = diff2.mean(dim=dims) if self.reduction == "mean" else diff2.sum(dim=dims)  # (B*K,)

        return v.reshape(batch.B, batch.K)  # (B,K)


# -------------------------
# Distribution metric: FID
# -------------------------

@DIST_METRICS.register("fid")
class InceptionFID(DistributionMetric):
    """
    FID computed via InceptionV3 features + piq.FID.

    Contract
    --------
    compute(ref, gen) -> float

    Assumptions
    -----------
    - ref and gen are image sets shaped (N, 3, H, W) and already in [0,1]
    """
    name = "fid"

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        from piq import FID  
        self._fid = FID()

        try:
            from torchvision.models import inception_v3  
            self._inception_ctor = inception_v3
        except Exception as e:
            raise ImportError("InceptionFID requires torchvision.") from e

        self._model: Optional[torch.nn.Module] = None

    def _get_model(self, device: torch.device) -> torch.nn.Module:
        if self._model is None:
            # torchvision API varies: prefer weights="DEFAULT" when available
            try:
                model = self._inception_ctor(weights="DEFAULT", transform_input=False)
            except TypeError:
                model = self._inception_ctor(pretrained=True, transform_input=False)
            model.fc = torch.nn.Identity()
            model.eval().to(device)
            self._model = model
        return self._model

    def _preprocess_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        images01 = _to01(images)
        images = F.interpolate(images01, size=(299, 299), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return ((images - mean) / std).float()

    @torch.no_grad()
    def _features(self, x01: torch.Tensor, *, device: torch.device, bs: int) -> torch.Tensor:
        model = self._get_model(device)
        feats = []
        for i in range(0, x01.shape[0], bs):
            xb = self._preprocess_imagenet(x01[i : i + bs].to(device, non_blocking=True))
            feats.append(model(xb).detach().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def compute(
        self,
        ref: torch.Tensor,
        gen: torch.Tensor,
        *,
        device: str,
    ) -> float:
        bs = self.batch_size

        f_ref = self._features(ref, device=device, bs=bs)
        f_gen = self._features(gen, device=device, bs=bs)

        score = self._fid.compute_metric(f_ref, f_gen)  # scalar tensor
        return float(score.detach().cpu().item())
