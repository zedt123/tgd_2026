import math
import torch
from typing import Optional, Tuple

def systematic_resample_batched(
    *xs: torch.Tensor,                      # each (B*N, ...)
    log_w: torch.Tensor,                    # (B, N)
    do_resample: torch.Tensor,              # (B,) bool
    reshape: Tuple[int, int],               # (B, N)
    generator: Optional[torch.Generator] = None,
    return_ancestor: bool = False,
):
    B, N = reshape

    if log_w.shape != (B, N):
        raise ValueError(f"log_w must have shape {(B, N)}, got {tuple(log_w.shape)}")
    if do_resample.shape != (B,):
        raise ValueError(f"do_resample must have shape {(B,)}, got {tuple(do_resample.shape)}")
    if len(xs) == 0:
        raise ValueError("Need at least one tensor to resample.")

    device = log_w.device

    # Convert everything to (B, N, ...) internally, while remembering original layout.
    xs_bn = []
    layouts = []   # "flat" or "batched"
    orig_shapes = []

    for i, x in enumerate(xs):
        orig_shapes.append(tuple(x.shape))

        if x.device != device:
            raise ValueError(
                f"All tensors must be on the same device as log_w. "
                f"xs[{i}].device={x.device}, log_w.device={device}"
            )

        if x.ndim >= 2 and tuple(x.shape[:2]) == (B, N):
            # Already batched
            xs_bn.append(x)
            layouts.append("batched")
        elif x.shape[0] == B * N:
            # Flat particle layout
            xs_bn.append(x.reshape(B, N, *x.shape[1:]))
            layouts.append("flat")
        else:
            raise ValueError(
                f"xs[{i}] has incompatible shape {tuple(x.shape)}. "
                f"Expected either (B*N, ...) with B*N={B*N}, or (B, N, ...)=({B}, {N}, ...)."
            )


    # reshape flat particles -> (B, N, ...)
    # xs_bn = [x.reshape(B, N, *x.shape[1:]) for x in xs]     # each (B, N, ...)
    
    # Weights already normalised
    #log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)

    if not do_resample.any():
        # No resampling: identity ancestry
        ancestor = None
        if return_ancestor:
            ancestor = torch.arange(N, dtype=torch.long)[None, :].expand(B, N).clone()
        return tuple(xs), log_w, ancestor

    w = torch.exp(log_w)   

    # cumulative sum on CPU for reliability when N large
    cdf = torch.cumsum(w.double().cpu(), dim=-1)
    cdf[:, -1] = 1.0

    rows = do_resample.detach().cpu().nonzero(as_tuple=True)[0] # (m,)
    m = rows.numel()

    u0 = torch.rand((m,), device='cpu', generator=generator, dtype=torch.float64) / N   # uniform[0, 1/N)
    u = (torch.arange(N, device='cpu',  dtype=torch.float64)[None, :] / N) + u0[:, None] # (m, N)

    idx_sel = torch.searchsorted(cdf[rows], u).to(device)
    rows = rows.to(device)
    idx = torch.arange(N, device=device, dtype=idx_sel.dtype)[None, :].expand(B, N).clone()
    idx[rows] = idx_sel 

    # gather along particle dimension (dim=1)
    def gather_along_N(x_bn: torch.Tensor, idx_bn: torch.Tensor) -> torch.Tensor:
        # x_bn:   (B, N, ...)
        # idx_bn: (B, N)
        idx_expanded = idx_bn.view(B, N, *([1] * (x_bn.ndim - 2))).expand_as(x_bn)
        return torch.gather(x_bn, dim=1, index=idx_expanded)

    resampled = []
    for x_bn, layout, orig_shape in zip(xs_bn, layouts, orig_shapes):
        y_bn = gather_along_N(x_bn, idx)
        if layout == "flat":
            y = y_bn.reshape(B * N, *orig_shape[1:])
        else:
            y = y_bn
        resampled.append(y)

    # reset log weights only for resampled rows
    log_w_new = log_w.clone()                             
    log_w_new[rows.to(log_w.device)] = -math.log(N)

    ancestor = idx if return_ancestor else None
    return tuple(resampled), log_w_new, ancestor