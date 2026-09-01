"""Reproduce Honma (1999, ApJ 516, 693): maximum-likelihood M/L of binary
galaxies, with optical-pair contamination.

Method (his sec. 3):
  - Bound pairs: point masses; separation vector distributed as nu(R) ~ R^-2.6
    (volume density -- verified against his observed R_p distribution:
    p(scalar R) ~ R^{2-gamma} = R^-0.6 gives KS p = 0.96, the scalar reading
    p~0) with R >= Rmin = 10 kpc. Velocities from the Jeans equation for a
    point mass:
        beta = 0    (isotropic): sigma_r^2 = G'(M/L) / ((gamma+1) R),
                    V_los ~ N(0, sigma_r) -- one component of an isotropic
                    Gaussian, independent of the separation orientation.
        beta = -inf (circular):  |V|^2 = G'(M/L)/R, perpendicular to R;
                    projected with the SAME line of sight as R.
    G' = 4.30091e-6 * 1e10 = 43009.1 kpc (km/s)^2 / (1e10 Lsun) since all
    quantities are luminosity-corrected: R_p = r_p/L10^{1/3}, V_p = |dv|/L10^{1/3}.
  - Optical pairs: p_opt ~ [1 + xi(r)] R_p on the window, uniform in V_p;
    xi = (r0/r)^q with q = 0 (none) or q = 1.8, r0 = 10 Mpc;
    r^2 = r_p^2 + (dv/H0)^2 in physical units (per-pair L10 used to convert).
  - Likelihood over the window R_p, V_p in [0, 400] (his eqs. 17-19):
        log L(M/L, f) = sum_i INT g_i(V) log[ f p_bin(R_p,i, V) + (1-f) p_opt ] dV
    i.e. each OBSERVED pair is replaced by a Gaussian cloud g_i in V_p with its
    measurement error, and the cloud is integrated against the LOG of the
    model density (an expected log-likelihood). This is NOT equivalent to
    convolving the model and taking log of the integral: by Jensen's
    inequality the expected-log form penalises models that are small anywhere
    under the cloud, which pushes M/L up relative to the convolved form
    (a first implementation using the convolved form landed systematically
    at ~0.6x his values; a moment-matching cross-check on the median
    projected (M/L)_p, which is estimator-independent, agrees with HIS
    numbers -- so the expected-log reading of eqs. 17-18 is the right one).

M/L enters only through the velocity scale s = sqrt(M/L), so the ensemble is
simulated ONCE at M/L = 1 and rescaled.

Targets (his Tables): sample I (57 pairs) beta=0: 35+7-5 (q=0), 36+8-4 (q=1.8);
beta=-inf: 28+5-3, 30+6-3; f = 0.88 / 0.71. Sample III (30 pure spiral pairs):
15+5-3 / 16+4-4 (beta=0), 12 (beta=-inf).
"""
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

rng = np.random.default_rng(11)

GAMMA = 2.6
RMIN, RMAX = 10.0, 2000.0     # kpc; RMAX unspecified by Honma -- see sensitivity
WINDOW = 400.0                # kpc and km/s
GP = 4.30091e-6 * 1e10        # G in luminosity-corrected units
H0 = 50.0
R0_MPC = 10.0                 # clustering scale for xi
N_MC = 4_000_000
H_R = 5.0                     # half-width of the R_p slab for density estimation

# --------------------------------------------------------------------------
# Observed pairs
# --------------------------------------------------------------------------
obs = pd.read_csv('honma99_pairs.csv')

# "Pure spiral pairs" (his sample III, N = 30): the only reading of the type
# column that yields exactly 30 pairs is "both members spiral or irregular",
# i.e. everything except E*, S0/SB0 and Pec -- so his "later than Sa"
# includes Sa itself.
NONSPIRAL = {'E', 'E0', 'E1', 'E2', 'E3', 'E5', 'E6', 'S0', 'SB0', 'Pec'}
def is_spiral(t):
    return str(t).strip() not in NONSPIRAL

samples = {
    'I  (57 pairs)': obs,
    'III (pure spirals)': obs[[is_spiral(a) and is_spiral(b)
                               for a, b in zip(obs.type1, obs.type2)]],
}
for name, d in samples.items():
    print('sample %-20s N = %d' % (name, len(d)))
print()

# --------------------------------------------------------------------------
# Bound-pair ensemble at M/L = 1
# --------------------------------------------------------------------------
def simulate(beta, n=N_MC):
    """Return (Rp, Vp) at M/L = 1 in luminosity-corrected units."""
    a = 3.0 - GAMMA   # p(R) ~ R^{2-gamma}; exponent+1
    u = rng.random(n)
    R = (RMIN**a + u*(RMAX**a - RMIN**a))**(1/a)

    if beta == 0:
        sigma = np.sqrt(GP / ((GAMMA + 1.0) * R))
        Vlos = np.abs(rng.normal(0.0, sigma))
        Rp = R * np.sqrt(1.0 - rng.random(n)**2)      # cos(theta) ~ U(0,1)
    else:  # circular
        Vc = np.sqrt(GP / R)
        # one shared line of sight per system: r_hat isotropic, v_hat uniform
        # in the plane perpendicular to r_hat
        ct = rng.uniform(-1, 1, n); st = np.sqrt(1 - ct**2)
        ph = rng.uniform(0, 2*np.pi, n)
        rx, ry, rz = st*np.cos(ph), st*np.sin(ph), ct
        # basis of the perpendicular plane
        ax, ay, az = -np.sin(ph), np.cos(ph), np.zeros(n)
        bx = ry*az - rz*ay; by = rz*ax - rx*az; bz = rx*ay - ry*ax
        psi = rng.uniform(0, 2*np.pi, n)
        vz = np.cos(psi)*az + np.sin(psi)*bz
        Rp = R * np.sqrt(1.0 - rz**2)
        Vlos = Vc * np.abs(vz)
    return Rp, Vlos

# --------------------------------------------------------------------------
# Optical-pair density on the window (analytic, per pair via its L10)
# --------------------------------------------------------------------------
def p_opt_column(Rp_i, L10, q, v_grid):
    """Normalised p_opt along the V_p column at R_p = Rp_i."""
    if q == 0:
        # ~ Rp, uniform in Vp; integral over window = W^2/2 * W
        return np.full_like(v_grid, Rp_i / (WINDOW**2/2 * WINDOW))
    Lc = L10**(1/3.)
    gr = np.linspace(0.5, WINDOW, 800)
    gv = np.linspace(0.5, WINDOW, 800)
    RR, VV = np.meshgrid(gr, gv, indexing='ij')
    r_mpc = np.sqrt((RR*Lc/1000.)**2 + (VV*Lc/H0)**2)
    dens = RR * (1.0 + (R0_MPC / r_mpc)**q)
    norm = np.trapz(np.trapz(dens, gv, axis=1), gr)
    r_v = np.sqrt((Rp_i*Lc/1000.)**2 + (v_grid*Lc/H0)**2)
    return Rp_i * (1.0 + (R0_MPC / np.maximum(r_v, 1e-3))**q) / norm

# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------
V_STEP = 2.0
V_GRID = np.arange(V_STEP/2, WINDOW, V_STEP)   # 200 column points
H_V = 3.0    # km/s half-width Gaussian used to smooth the MC column density

def fit(sample, Rp_mc, Vp_mc, q, ml_grid, f_grid):
    Rp_i = sample.Rp.values; Vp_i = sample.Vp.values
    # floor the measurement error at 1 km/s: one pair has e_Vel = 0 in the
    # table, and a zero-width kernel poisons the whole likelihood
    sig_i = np.maximum(sample.sigma_Vp.values, 1.0); L_i = sample.L10.values
    N = len(Rp_i)

    # observed clouds g_i on the column grid, each normalised to sum 1
    G_obs = np.exp(-0.5*((V_GRID[None, :] - Vp_i[:, None])/sig_i[:, None])**2)
    G_obs /= G_obs.sum(axis=1, keepdims=True)

    popt_col = np.stack([p_opt_column(Rp_i[k], L_i[k], q, V_GRID)
                         for k in range(N)])                       # (N, NV)

    slabs = [Vp_mc[np.abs(Rp_mc - r) < H_R] for r in Rp_i]
    in_win_R = Rp_mc < WINDOW
    n_tot = len(Rp_mc)

    logL = np.full((len(ml_grid), len(f_grid)), -np.inf)
    for a, ml in enumerate(ml_grid):
        s = np.sqrt(ml)
        frac = max(np.mean(in_win_R & (Vp_mc*s < WINDOW)), 1e-12)
        # model density along each pair's column: smoothed histogram of the
        # slab velocities (reflected at 0), per unit (Rp, Vp) area, normalised
        # on the window
        pbin_col = np.empty((N, len(V_GRID)))
        edges = np.arange(0.0, WINDOW + V_STEP, V_STEP)
        for k in range(N):
            v = slabs[k] * s
            # histogram (cheap), then Gaussian smooth; reflection at 0 handled
            # by prepending the mirrored first bins
            h, _ = np.histogram(v, bins=edges)
            pad = h[:8][::-1]
            hs = gaussian_filter1d(np.concatenate([pad, h]).astype(float),
                                   sigma=H_V / V_STEP)[len(pad):]
            pbin_col[k] = hs / V_STEP / (n_tot*2*H_R) / frac
        for b, f in enumerate(f_grid):
            p = f*pbin_col + (1-f)*popt_col
            if np.all(p > 0):
                logL[a, b] = (G_obs * np.log(p)).sum()
    return logL

def profile(ml_grid, logL):
    prof = logL.max(axis=1)
    i = prof.argmax()
    ok = prof >= prof.max() - 0.5          # 68% for one parameter
    return ml_grid[i], ml_grid[ok].min(), ml_grid[ok].max()

# --------------------------------------------------------------------------
t0 = time.perf_counter()
ml_grid = np.geomspace(2, 150, 90)
f_grid = np.linspace(0.30, 1.00, 71)

ens = {}
for beta_name, beta in [('beta=0 (isotropic)', 0), ('beta=-inf (circular)', -1)]:
    ens[beta_name] = simulate(beta)

print('%-20s %-22s %-8s %8s %14s %8s   %s' %
      ('sample', 'orbits', 'q', 'M/L', '68% int', 'f_best', 'Honma'))
targets = {('I  (57 pairs)', 'beta=0 (isotropic)', 0): '35 +7-5, f=0.88',
           ('I  (57 pairs)', 'beta=0 (isotropic)', 1.8): '36 +8-4, f=0.71',
           ('I  (57 pairs)', 'beta=-inf (circular)', 0): '28 +5-3, f=0.88',
           ('I  (57 pairs)', 'beta=-inf (circular)', 1.8): '30 +6-3, f=0.73',
           ('III (pure spirals)', 'beta=0 (isotropic)', 0): '15 +5-3, f=0.95',
           ('III (pure spirals)', 'beta=0 (isotropic)', 1.8): '16 +4-4, f=0.84',
           ('III (pure spirals)', 'beta=-inf (circular)', 0): '12 +4-3',
           ('III (pure spirals)', 'beta=-inf (circular)', 1.8): '12 +3-3'}

results = []
for sname, sample in samples.items():
    for bname, (Rp_mc, Vp_mc) in ens.items():
        for q in (0, 1.8):
            logL = fit(sample, Rp_mc, Vp_mc, q, ml_grid, f_grid)
            ml, lo, hi = profile(ml_grid, logL)
            ia, ib = np.unravel_index(logL.argmax(), logL.shape)
            fbest = f_grid[ib]
            print('%-20s %-22s q=%-5s %8.1f %8.1f-%-5.1f %8.2f   %s' %
                  (sname, bname, q, ml, lo, hi, fbest,
                   targets.get((sname, bname, q), '')))
            results.append(dict(sample=sname, orbits=bname, q=q, ML=ml,
                                lo=lo, hi=hi, f=fbest))

pd.DataFrame(results).to_csv('honma99_ML_results.csv', index=False)
print('\nElapsed: %.1f s' % (time.perf_counter() - t0))
