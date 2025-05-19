import torch
from math import comb

def bernstein_coeff_ordern_new(n, tmin, tmax, t_actual_batch):
    """
    Vectorized Bernstein basis and derivatives for a batch of time points.

    Args:
        n: Order of Bernstein basis (degree n)
        tmin, tmax: Time interval (scalars)
        t_actual_batch: Tensor of shape (num, 1) or (num,) — time points

    Returns:
        P     : (num, n+1)   — Bernstein basis
        Pdot  : (num, n+1)   — First derivative
        Pddot : (num, n+1)   — Second derivative
    """
    t_actual_batch = t_actual_batch.squeeze(-1)  # Ensure shape is (num,)
    num = t_actual_batch.shape[0]
    l = tmax - tmin
    t = (t_actual_batch - tmin) / l  # normalize to [0, 1], shape (num,)

    # Precompute binomial coefficients
    binom_coeffs = torch.tensor([comb(n, i) for i in range(n + 1)], dtype=t.dtype, device=t.device)

    # Compute powers of t and (1 - t) for all i
    t_powers = torch.stack([t**i for i in range(n + 1)], dim=1)                  # (num, n+1)
    one_minus_t_powers = torch.stack([(1 - t)**(n - i) for i in range(n + 1)], dim=1)

    P = binom_coeffs * one_minus_t_powers * t_powers                            # (num, n+1)

    # First derivative
    Pdot = []
    for i in range(n + 1):
        term1 = n * comb(n - 1, i - 1) * (1 - t)**(n - i) * t**(i - 1) if i > 0 else torch.zeros_like(t)
        term2 = n * comb(n - 1, i) * (1 - t)**(n - i - 1) * t**i if i < n else torch.zeros_like(t)
        Pdot.append(term1 - term2)
    Pdot = torch.stack(Pdot, dim=1) / l  # shape (num, n+1)

    # Second derivative
    Pddot = []
    for i in range(n + 1):
        term1 = comb(n - 2, i - 2) * (1 - t)**(n - i) * t**(i - 2) if i > 1 else torch.zeros_like(t)
        term2 = 2 * comb(n - 2, i - 1) * (1 - t)**(n - i - 1) * t**(i - 1) if 0 < i < n else torch.zeros_like(t)
        term3 = comb(n - 2, i) * (1 - t)**(n - i - 2) * t**i if i < n - 1 else torch.zeros_like(t)
        val = n * (n - 1) * (term1 - term2 + term3)
        Pddot.append(val)
    Pddot = torch.stack(Pddot, dim=1) / (l ** 2)  # shape (num, n+1)

    return P, Pdot, Pddot
