import torch

def sample_posterior_px_given_y_abs(
    y, mu, var, sigma, n_samples, *, eps=1e-7, generator=None
):
    """
    y:  (d,)
    mu: (d,) or (n_samples,d)
    var:(d,) or (n_samples,d)
    returns: (n_samples,d)
    """
    y = y.view(1, -1)                         # (1,d)
    d = y.shape[1]
    sigma2 = sigma * sigma

    # Broadcast mu/var to (n_samples,d) if needed
    if mu.ndim == 1:
        mu = mu.view(1, d).expand(n_samples, d)
    if var.ndim == 1:
        var = var.view(1, d).expand(n_samples, d)

    v  = (sigma2 * var) / (sigma2 + var)
    sd = torch.sqrt(v)

    m_plus  = v * ( y/sigma2 + mu/var )
    m_minus = v * ( -y/sigma2 + mu/var )

    denom = var + sigma2

    # log N(mu; ±y, denom) up to constants that cancel between +/- for fixed denom
    logC_plus  = -0.5 * (mu - y)**2 / denom
    logC_minus = -0.5 * (mu + y)**2 / denom

    # truncation masses
    Phi_plus  = torch.special.ndtr(m_plus / sd).clamp_min(eps)
    Phi_minus = torch.special.ndtr((-m_minus) / sd).clamp_min(eps)

    logw_plus  = logC_plus  + torch.log(Phi_plus)
    logw_minus = logC_minus + torch.log(Phi_minus)

    p_plus = torch.sigmoid(logw_plus - logw_minus)

    u_sign = torch.rand((n_samples, d), device=y.device, generator=generator)
    sign_is_plus = (u_sign < p_plus)

    m = torch.where(sign_is_plus, m_plus, m_minus)
    s = sd

    # Truncation in z-space
    a = (0.0 - m) / s

    u = torch.rand((n_samples, d), device=y.device, generator=generator)
    lo = torch.special.ndtr(a)                         # Phi(a)

    u_plus  = lo + (1.0 - lo) * u                      # [Phi(a), 1)
    u_minus = lo * u                                   # (0, Phi(a)]

    u_trunc = torch.where(sign_is_plus, u_plus, u_minus).clamp(eps, 1.0 - eps)
    z = torch.special.ndtri(u_trunc)

    return m + s * z

def sample_posterior_px_given_y_abs_mog(
    y, means, variances, weights, sigma, n_samples,
    *, eps=1e-7, generator=None
):
    """
    y: (d,)
    means: (K,d)
    variances: (K,d)
    weights: (K,)
    returns x: (n_samples,d), k: (n_samples,)
    """
    K, d = means.shape
    sigma2 = sigma * sigma

    # log normalized weights
    logw = torch.log(weights)
    logw = logw - torch.logsumexp(logw, dim=0)

    yK = y.view(1, d).expand(K, d)
    denom = variances + sigma2
    v  = (sigma2 * variances) / (sigma2 + variances)
    sd = torch.sqrt(v)

    m_plus  = v * ( yK/sigma2 + means/variances )
    m_minus = v * ( -yK/sigma2 + means/variances )

    # Here denom varies across k => include -0.5 log denom term
    logN_plus  = -0.5 * (means - yK).pow(2) / denom - 0.5 * torch.log(denom)
    logN_minus = -0.5 * (means + yK).pow(2) / denom - 0.5 * torch.log(denom)

    Phi_plus  = torch.special.ndtr(m_plus / sd).clamp_min(eps)
    Phi_minus = torch.special.ndtr((-m_minus) / sd).clamp_min(eps)

    log_term_plus  = logN_plus  + torch.log(Phi_plus)
    log_term_minus = logN_minus + torch.log(Phi_minus)

    logZ_dim = torch.logsumexp(torch.stack([log_term_plus, log_term_minus], dim=0), dim=0)  # (K,d)
    log_py_given_k = logZ_dim.sum(dim=1)                                                    # (K,)

    logits_k = logw + log_py_given_k
    probs_k = torch.softmax(logits_k, dim=0)

    k_samples = torch.multinomial(probs_k, n_samples, replacement=True, generator=generator)  # (N,)

    mu_s  = means[k_samples]       # (N,d)
    var_s = variances[k_samples]   # (N,d)

    x = sample_posterior_px_given_y_abs(
        y=y, mu=mu_s, var=var_s, sigma=sigma, n_samples=n_samples, eps=eps, generator=generator
    )
    return x, k_samples