#!/usr/bin/env python3
"""Exact D=4 infinite-chain variance density from the accompanying research draft.

This *analytical* formula is independent of the Monte Carlo sampling code.
At depth D=4, v_inf(p) = p * Q(p) / (4 * 233280000000).
"""

# Q polynomial coefficients in ASCENDING powers of p, checked against
# the manuscript's explicit polynomial (Appendix E, Eq. E14).
Q_COEFFS = (
    232626816000,
    -191121545088,
    128513835360,
    -70449107184,
    30986183880,
    -10835143470,
    2935613475,
    -597691575,
    88172775,
    -8898600,
    549375,
    -15625,
)
DENOMINATOR = 4 * 233280000000


def variance_density_d4_infinite(p):
    """Compute v_4^(infinity)(p), valid for 0 <= p <= 1."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    q = 0
    for c in reversed(Q_COEFFS):
        q = q * p + c
    return p * q / DENOMINATOR


if __name__ == "__main__":
    for i in range(11):
        p = i / 10
        print(f"pT={p:.1f}: {variance_density_d4_infinite(p):.12f}")
