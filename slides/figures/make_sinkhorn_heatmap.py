"""Generate Sinkhorn transport plan heatmaps at small vs large epsilon.

Produces sinkhorn_heatmap.png: two side-by-side heatmaps of P_ij for
the same source/target on a 1D-sorted scattered point set so the
sharp diagonal vs blurry blob contrast is visible.

Run:  python make_sinkhorn_heatmap.py
"""

import numpy as np
import matplotlib.pyplot as plt


def sinkhorn(a, b, C, eps, n_iters=3000, tol=1e-10):
    """Vanilla Sinkhorn in log domain. Returns P (N, N)."""
    log_a = np.log(a)
    log_b = np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    for _ in range(n_iters):
        M = (g[None, :] - C) / eps + log_b[None, :]
        Mmax = M.max(axis=1, keepdims=True)
        f_new = -eps * (np.log(np.exp(M - Mmax).sum(axis=1)) + Mmax.flatten())
        M = (f_new[:, None] - C) / eps + log_a[:, None]
        Mmax = M.max(axis=0, keepdims=True)
        g_new = -eps * (np.log(np.exp(M - Mmax).sum(axis=0)) + Mmax.flatten())
        if np.max(np.abs(f_new - f)) < tol and np.max(np.abs(g_new - g)) < tol:
            f, g = f_new, g_new
            break
        f, g = f_new, g_new
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return P


def main():
    rng = np.random.default_rng(42)
    N = 50

    # Particles on the line, weights log-normal then normalised; target uniform
    x = np.sort(rng.standard_normal(N))
    log_w = 0.7 * rng.standard_normal(N)
    a = np.exp(log_w)
    a /= a.sum()
    b = np.full(N, 1.0 / N)

    # squared cost on the same point set, normalised by mean
    diff = x[:, None] - x[None, :]
    C = 0.5 * diff ** 2
    C = C / C.mean()

    P_small = sinkhorn(a, b, C, eps=0.01)
    P_large = sinkhorn(a, b, C, eps=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Per-panel vmax so the visual structure is visible in both
    im0 = axes[0].imshow(P_small, cmap='viridis',
                         vmin=0, vmax=np.percentile(P_small, 99.5),
                         interpolation='nearest', aspect='equal')
    axes[0].set_title(r'$\varepsilon = 0.01$ : sharp', fontsize=14)
    axes[0].set_xlabel('target index $j$ (sorted by position)')
    axes[0].set_ylabel('source index $i$ (sorted by position)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r'$P_{ij}$')

    im1 = axes[1].imshow(P_large, cmap='viridis',
                         vmin=0, vmax=np.percentile(P_large, 99.5),
                         interpolation='nearest', aspect='equal')
    axes[1].set_title(r'$\varepsilon = 0.5$ : blurry', fontsize=14)
    axes[1].set_xlabel('target index $j$ (sorted by position)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r'$P_{ij}$')

    fig.suptitle(r'Transport plan $P_{ij}$ on the same source/target ($N=50$, particles sorted by position)',
                 fontsize=13)
    fig.tight_layout()

    out = 'sinkhorn_heatmap.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
