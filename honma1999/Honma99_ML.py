"""Reproduce Honma (1999, ApJ 516, 693): maximum-likelihood M/L of binary
galaxies with optical-pair contamination.

Method (his sec. 3):
  - Bound pairs: point masses; separation vector distributed as the VOLUME
    density nu(R) ~ R^-2.6 (his Jeans-equation nu; confirmed: projecting the
    scalar p(R) ~ R^{2-gamma} = R^-0.6 matches his observed R_p at KS p = 0.96
    with the KS peaking at gamma = 2.6, the scalar reading p ~ 1e-29),
    R >= Rmin = 10 kpc. Velocities from the Jeans equation for a point mass,
    G' = 4.30091e-6 * 1e10 = 43009.1 in luminosity-corrected units:
        isotropic (beta=0): sigma_r^2 = G'(M/L) / ((gamma+1) R) = G'(M/L)/(3.6 R)
        circular (beta=-inf): |V|^2 = G'(M/L)/R, V perpendicular to R,
                              one shared line of sight for R and V
        radial (beta=1):      sigma_r^2 = G'(M/L) / ((gamma-1) R), V along R
    Honma states only the Jeans dispersions, not a distribution function.
    DF = 'bound' (default) draws an isotropic 3-D Gaussian and rejects speeds
    above escape, v^2 > 2G'(M/L)/R -- scale-free, since v^2/v_esc^2 =
    chi2_3/(2(gamma+1)) is independent of M/L. DF = 'gauss' keeps the
    unbounded Gaussian. 'bound' is the physically consistent choice for a
    bound-pair ensemble and moves every isotropic result 10-16% toward his.
  - Optical pairs: p_opt ~ [1 + xi(r)] R_p on the window (uniform in V_p for
    q = 0; falls with V_p for q = 1.8), xi = (r0/r)^q, r0 = 10 Mpc,
    r^2 = r_p^2 + (dv/H0)^2 in physical units. Per-pair L10 is used to
    convert; a universal p_opt (L10 = 1) gives identical results because
    xi >> 1 across the window.
  - Likelihood (his eqs. 17-19), an EXPECTED log-likelihood:
        log L(M/L, f) = sum_i INT g_i(V) log[ f p_bin(R_p,i, V) + (1-f) p_opt ] dV
    the OBSERVED pair spread as a Gaussian cloud g_i in V_p with its error,
    integrated against log p. This is the literal reading of eq. 17
    (sum n log p over the phase space) with eq. 18 (n = sum of g_i). The
    decisive evidence that it is what he computed is the circular-orbit case:
    the model has a hard cutoff at V_c, so the convolved form log INT g p dV
    gives 12.0 against his 28, the expected-log form 27.5.
    WEIGHT = 'unit' (each cloud sums to 1) is used. Eq. 19 as printed has a
    (2 pi sigma_i)^(-1/2) prefactor, which with eq. 18's collective
    normalisation would weight pair i by sqrt(sigma_i); that reading
    ('sqrtsig') overshoots his values when combined with the bound DF and is
    treated as a typo.

M/L enters only through the velocity scale sqrt(M/L), so each ensemble is
simulated once at M/L = 1 and rescaled.

Outputs the eight Table-2 comparisons (samples I and III), the radial case
(his sec. 3.3: 42 +34-7), and his sec. 4.1 separation subgroups (27/12/18
pairs at R_p < 100, 100-200, > 200 kpc: 36 +10-5, 37 +31-17, 25 +29-13),
each with the true-pair fraction f, the 1-D profile interval
(delta log L = 0.5) and the 2-D-contour convention he plots (delta log L =
1.15, both parameters).
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

rng = np.random.default_rng(11)

DF = sys.argv[1] if len(sys.argv) > 1 else 'bound'        # 'bound' | 'gauss'
WEIGHT = sys.argv[2] if len(sys.argv) > 2 else 'unit'     # 'unit' | 'sqrtsig'

GAMMA = 2.6
RMIN, RMAX = 10.0, 2000.0     # RMAX unstated by Honma; results insensitive 400-5000
WINDOW = 400.0
GP = 4.30091e-6 * 1e10
H0 = 50.0
R0_MPC = 10.0
N_MC = 4_000_000               # his: one million
H_R = 5.0                      # R_p slab half-width for the density estimate (insensitive 5-50)
V_STEP = 2.0
V_GRID = np.arange(V_STEP / 2, WINDOW, V_STEP)
H_V = 3.0                      # km/s smoothing of the model column (insensitive 1-3)

obs = pd.read_csv('honma99_pairs.csv')

# "Pure spiral pairs" (his sample III, N = 30): both members anything but
# E*, S0/SB0, Pec -- so Sa AND the irregulars (Im, IBm, Sm) count. Confirmed
# three ways: N = 30 (unique among 64 exclusion combinations), his stated 70%
# spiral fraction of sample I (ours 70.2%), and his stated mean raw
# separation of sample III, ~206 kpc (ours 205.8).
NONSPIRAL = {'E', 'E0', 'E1', 'E2', 'E3', 'E5', 'E6', 'S0', 'SB0', 'Pec'}
def is_spiral(t):
    return str(t).strip() not in NONSPIRAL

sampleIII = obs[[is_spiral(a) and is_spiral(b) for a, b in zip(obs.type1, obs.type2)]]
Rp = obs.Rp_recomputed          # unrounded; his 27/12/18 subgroup split reproduces exactly
bins = {'Rp<100': obs[Rp < 100], '100-200': obs[(Rp >= 100) & (Rp < 200)], '>200': obs[Rp >= 200]}


def simulate(model, n=N_MC):
    a = 3.0 - GAMMA
    R = (RMIN ** a + rng.random(n) * (RMAX ** a - RMIN ** a)) ** (1 / a)
    if model == 'iso':
        sigma = np.sqrt(GP / ((GAMMA + 1.0) * R))
        vx, vy, vz = rng.normal(0, 1, (3, n)) * sigma
        if DF == 'bound':
            keep = (vx ** 2 + vy ** 2 + vz ** 2) < 2 * GP / R
            while keep.sum() < n:
                m = ~keep
                vx[m], vy[m], vz[m] = rng.normal(0, 1, (3, m.sum())) * sigma[m]
                keep = (vx ** 2 + vy ** 2 + vz ** 2) < 2 * GP / R
        return R * np.sqrt(1.0 - rng.random(n) ** 2), np.abs(vz)
    if model == 'radial':
        sigma = np.sqrt(GP / ((GAMMA - 1.0) * R))
        v = rng.normal(0, sigma)
        ct = rng.uniform(-1, 1, n)
        return R * np.sqrt(1 - ct ** 2), np.abs(v * ct)
    # circular: r_hat isotropic, v_hat uniform in the perpendicular plane
    Vc = np.sqrt(GP / R)
    ct = rng.uniform(-1, 1, n); st = np.sqrt(1 - ct ** 2)
    psi = rng.uniform(0, 2 * np.pi, n)
    return R * st, Vc * np.abs(np.sin(psi) * st)


def p_opt_column(Rp_i, L10, q, v_grid):
    if q == 0:
        return np.full_like(v_grid, Rp_i / (WINDOW ** 2 / 2 * WINDOW))
    Lc = L10 ** (1 / 3.)
    gr = np.linspace(0.5, WINDOW, 800); gv = np.linspace(0.5, WINDOW, 800)
    RR, VV = np.meshgrid(gr, gv, indexing='ij')
    r_mpc = np.sqrt((RR * Lc / 1000.) ** 2 + (VV * Lc / H0) ** 2)
    dens = RR * (1.0 + (R0_MPC / r_mpc) ** q)
    norm = np.trapz(np.trapz(dens, gv, axis=1), gr)
    r_v = np.sqrt((Rp_i * Lc / 1000.) ** 2 + (v_grid * Lc / H0) ** 2)
    return Rp_i * (1.0 + (R0_MPC / np.maximum(r_v, 1e-3)) ** q) / norm


def fit(sample, Rp_mc, Vp_mc, q, ml_grid, f_grid):
    Rp_i = sample.Rp.values; Vp_i = sample.Vp.values
    sig_i = np.maximum(sample.sigma_Vp.values, 1.0)   # one pair has e_Vel = 0; floor is immaterial otherwise
    L_i = sample.L10.values
    N = len(Rp_i)
    G_obs = np.exp(-0.5 * ((V_GRID[None, :] - Vp_i[:, None]) / sig_i[:, None]) ** 2)
    G_obs /= G_obs.sum(axis=1, keepdims=True)
    if WEIGHT == 'sqrtsig':
        w = np.sqrt(sig_i); G_obs *= (w / w.mean())[:, None]
    popt_col = np.stack([p_opt_column(Rp_i[k], L_i[k], q, V_GRID) for k in range(N)])
    slabs = [Vp_mc[np.abs(Rp_mc - r) < H_R] for r in Rp_i]
    in_win_R = Rp_mc < WINDOW
    n_tot = len(Rp_mc)
    edges = np.arange(0.0, WINDOW + V_STEP, V_STEP)
    logL = np.full((len(ml_grid), len(f_grid)), -np.inf)
    for a, ml in enumerate(ml_grid):
        s = np.sqrt(ml)
        frac = max(np.mean(in_win_R & (Vp_mc * s < WINDOW)), 1e-12)
        pbin_col = np.empty((N, len(V_GRID)))
        for k in range(N):
            h, _ = np.histogram(slabs[k] * s, bins=edges)
            pad = h[:8][::-1]
            hs = gaussian_filter1d(np.concatenate([pad, h]).astype(float), sigma=H_V / V_STEP)[len(pad):]
            pbin_col[k] = hs / V_STEP / (n_tot * 2 * H_R) / frac
        for b, f in enumerate(f_grid):
            p = f * pbin_col + (1 - f) * popt_col
            if np.all(p > 0):
                logL[a, b] = (G_obs * np.log(p)).sum()
    return logL


def summarise(ml_grid, f_grid, logL):
    ia, ib = np.unravel_index(logL.argmax(), logL.shape)
    prof = logL.max(axis=1)
    ok1 = prof >= prof.max() - 0.5            # 1-D profile, 68%
    ok2 = prof >= prof.max() - 1.15           # 2-D contour extent, 68% (his figures)
    return (ml_grid[ia], f_grid[ib], ml_grid[ok1].min(), ml_grid[ok1].max(),
            ml_grid[ok2].min(), ml_grid[ok2].max())


if __name__ == '__main__':
    t0 = time.perf_counter()
    ml_grid = np.geomspace(2, 150, 90)
    f_grid = np.linspace(0.30, 1.00, 71)
    ens = {m: simulate(m) for m in ('iso', 'circ', 'radial')}
    print('DF = %s, weighting = %s' % (DF, WEIGHT))
    print('%-22s %-9s %-4s %6s %5s %12s %12s   %s' %
          ('sample', 'orbits', 'q', 'M/L', 'f', '68% (1-D)', '68% (2-D)', 'Honma'))
    cases = [('I  (57 pairs)', obs, 'iso', 0, '35 +7-5   f=0.88'),
             ('I  (57 pairs)', obs, 'iso', 1.8, '36 +8-4   f=0.71'),
             ('I  (57 pairs)', obs, 'circ', 0, '28 +5-3   f=0.88'),
             ('I  (57 pairs)', obs, 'circ', 1.8, '30 +6-3   f=0.73'),
             ('III (30 spirals)', sampleIII, 'iso', 0, '15 +5-3   f=0.95'),
             ('III (30 spirals)', sampleIII, 'iso', 1.8, '16 +4-4   f=0.84'),
             ('III (30 spirals)', sampleIII, 'circ', 0, '12 +4-3   f=0.95'),
             ('III (30 spirals)', sampleIII, 'circ', 1.8, '12 +3-3   f=0.86'),
             ('I  radial beta=1', obs, 'radial', 0, '42 +34-7'),
             ('I  Rp<100 (27)', bins['Rp<100'], 'iso', 0, '36 +10-5'),
             ('I  100-200 (12)', bins['100-200'], 'iso', 0, '37 +31-17'),
             ('I  >200 (18)', bins['>200'], 'iso', 0, '25 +29-13')]
    rows = []
    for name, sample, model, q, H in cases:
        r = summarise(ml_grid, f_grid, fit(sample, *ens[model], q, ml_grid, f_grid))
        print('%-22s %-9s %-4s %6.1f %5.2f %5.1f-%-6.1f %5.1f-%-6.1f   %s'
              % (name, model, q, r[0], r[1], r[2], r[3], r[4], r[5], H))
        rows.append(dict(sample=name, orbits=model, q=q, ML=r[0], f=r[1],
                         lo1=r[2], hi1=r[3], lo2=r[4], hi2=r[5], honma=H, DF=DF, weight=WEIGHT))
    pd.DataFrame(rows).to_csv('honma99_ML_results.csv', index=False)
    print('\nElapsed: %.0f s' % (time.perf_counter() - t0))
