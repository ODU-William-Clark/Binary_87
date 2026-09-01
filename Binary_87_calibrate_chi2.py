"""Calibrate the minimised chi^2, and bootstrap confidence intervals on y.

Two things the raw scan cannot tell you:

1. The value reported by Binary_87_Chi_Square_3 is the MINIMUM of chi^2 over the
   y scan, not a single draw of the statistic. A minimum is biased low, so it
   must NOT be compared against the degrees of freedom: a correct model does
   not produce reduced chi^2 = 1 here. This script measures what it does
   produce.

2. The delta-chi^2 = 1 sublevel set is DISCONNECTED (chi^2 steps whenever a u
   value crosses one of the nb cell boundaries), so reporting its min and max
   as an "interval" reports the convex hull of a set with holes. The bootstrap
   below gives a real interval instead.

It also serves as S87's own closure test (sec. VI[c]): she generated 10
synthetic samples of 43 bound pairs with known M/L and recovered
(M/L)_out/(M/L)_in = 1.03 +/- 0.07, from which her 22% formal error follows.
The ratio reported here is the same quantity.

Everything is read from the pipeline outputs -- nothing is hardcoded -- and the
conditional (r_p-binned) PDFs match Binary_87_ML_8 exactly.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --- must mirror Binary_87_ML_8 -------------------------------------------
NB = 5
NBINS_PSI = 50
USE_R_CONDITIONING = True
N_R_BINS = 8
MIN_SIM_PER_BIN = 15
PSI_RANGE_PCT = None
DEFAULT_SIGMA_V = 9.0
Y_GRID_N = 4000
Q_MODEL = 'q3'
N_BOOT = 300
SEED = 7

q_funcs = {'q1': lambda t: 6.47 - 0.39 * t,
           'q2': lambda t: np.where(t <= 0, 2.0, 1.0),
           'q3': lambda t: np.ones_like(t)}


def _sigma(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return DEFAULT_SIGMA_V


def load_observed():
    d = pd.read_csv("Binary_gal_87.txt")
    d = d[pd.to_numeric(d["P"], errors="coerce") > 0.5]
    q = q_funcs[Q_MODEL]
    S, P, sig, rp = [], [], [], []
    for _, r in d.iterrows():
        qA = q(np.array([float(r["type1"])]))[0]
        qB = q(np.array([float(r["type2"])]))[0]
        S.append((qA * float(r["L1"]) + qB * float(r["L2"])) * 1e10)
        P.append(float(r["P"]))
        sig.append(np.sqrt(_sigma(r["sigma1"])**2 + _sigma(r["sigma2"])**2))
        rp.append(float(r["r"]))
    return map(np.asarray, (S, P, sig, rp))


def build_pdfs(sim_r, sim_psi, r_obs):
    """Conditional psi_p pdf per observed pair, exactly as Binary_87_ML_8 does."""
    if USE_R_CONDITIONING:
        r_bins = np.quantile(sim_r, np.linspace(0, 1, N_R_BINS + 1))
        r_bins[0], r_bins[-1] = -np.inf, np.inf
        groups = [sim_psi[(sim_r >= r_bins[k]) & (sim_r < r_bins[k + 1])]
                  for k in range(N_R_BINS)]
    else:
        r_bins = np.array([-np.inf, np.inf])
        groups = [sim_psi]

    built = []
    for g in groups:
        if len(g) < MIN_SIM_PER_BIN:
            built.append(None)
            continue
        hi = np.max(g) * 1.2 if PSI_RANGE_PCT is None else np.percentile(g, PSI_RANGE_PCT)
        hist, edges = np.histogram(g, bins=NBINS_PSI, range=(0, hi))
        bc = 0.5 * (edges[:-1] + edges[1:])
        dx = bc[1] - bc[0]
        built.append((bc, dx, hist / np.sum(hist * dx), g))

    idx = np.clip(np.digitize(r_obs, r_bins) - 1, 0, len(built) - 1)
    return built, idx


def chi2_curve(dv, M, cdf_tab, bc0, dx, P):
    """chi^2 against uniformity of u, for every y on the grid. Returns the array.

    bc0 and dx are PER PAIR: with r_p conditioning each group gets its own
    histogram range (max of that group x 1.2), so the bin width differs from
    one r_p bin to the next. Using a single dx for all pairs mis-locates
    psi_obs on the CDF grid and biases the recovered y by a factor of ~3.
    """
    N, NY = M.shape
    psi_obs = dv[:, None] / np.sqrt(M)
    t = np.clip((psi_obs - bc0[:, None]) / dx[:, None], 0, NBINS_PSI - 1.0001)
    lo = t.astype(int)
    fr = t - lo
    I, J = np.indices((N, NY))
    u = cdf_tab[I, J, lo] * (1 - fr) + cdf_tab[I, J, lo + 1] * fr
    kb = np.clip((u * NB).astype(int), 0, NB - 1)
    W = np.zeros((NB, NY))
    np.add.at(W, (kb, J), P[:, None])
    E = W.sum(0) / NB
    return (((W - E) ** 2) / E).sum(0)


def main():
    S, P, sig, r_obs = load_observed()
    N = len(P)
    ygrid = np.linspace(0.1, 500, Y_GRID_N)
    M = ygrid[None, :] * S[:, None]

    sim = np.load("simulated_psi_data_test.npz", allow_pickle=True)
    fit = pd.read_csv("ML_fit_results.csv")
    rng = np.random.default_rng(SEED)

    print("N = %d pairs, q model = %s, %d-point y grid, %d bootstraps"
          % (N, Q_MODEL, Y_GRID_N, N_BOOT))
    print("r_p conditioning: %s\n" % ("ON (%d bins)" % N_R_BINS if USE_R_CONDITIONING else "OFF"))

    # --- null distribution of the UNMINIMISED statistic ---------------------
    chis = np.empty(200000)
    for k in range(len(chis)):
        W, _ = np.histogram(rng.random(N), bins=NB, range=(0, 1), weights=P)
        E = W.sum() / NB
        chis[k] = ((W - E) ** 2 / E).sum()
    print("Unminimised null (u ~ U(0,1), real P weights):")
    print("   mean = %.3f   median = %.3f   [analytic: (nb-1)*sum(P^2)/sum(P) = %.3f]"
          % (chis.mean(), np.median(chis), (NB - 1) * (P**2).sum() / P.sum()))
    print()

    hdr = "%-5s %8s %9s %9s %9s %9s %9s" % (
        "model", "y_fit", "chi2_min", "pctile", "boot med", "68% interval", "out/in")
    print(hdr)
    print("-" * len(hdr))

    for model in ['f1', 'f2', 'f3', 'f4', 'f5']:
        sim_r = np.asarray(sim['filtered_r_proj'].item()[model])
        sim_psi = np.asarray(sim['filtered_psi_proj'].item()[model])
        built, idx = build_pdfs(sim_r, sim_psi, r_obs)

        bc0 = np.array([built[idx[i]][0][0] for i in range(N)])
        dx = np.array([built[idx[i]][1] for i in range(N)])

        # convolved CDF for every (pair, y)
        cdf_tab = np.empty((N, Y_GRID_N, NBINS_PSI))
        for i in range(N):
            _, dx_i, pdf, _ = built[idx[i]]
            sb = (sig[i] / np.sqrt(M[i])) / dx_i
            for j in range(Y_GRID_N):
                pc = gaussian_filter1d(pdf, sigma=sb[j])
                pc /= np.sum(pc * dx_i)
                c = np.cumsum(pc) * dx_i
                cdf_tab[i, j] = c / c[-1]

        s = fit[(fit.f_model == model) & (fit.q_model == Q_MODEL)]
        y_fit = s.loc[s.chi_squared.idxmin(), 'y']
        chi2_obs = s.chi_squared.min()

        # --- parametric bootstrap at the fitted y --------------------------
        y_boot = np.empty(N_BOOT)
        c_boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            dv = np.empty(N)
            for i in range(N):
                psi_p = rng.choice(built[idx[i]][3])
                dv[i] = psi_p * np.sqrt(y_fit * S[i]) + rng.normal(0, sig[i])
            c = chi2_curve(np.abs(dv), M, cdf_tab, bc0, dx, P)
            y_boot[b] = ygrid[c.argmin()]
            c_boot[b] = c.min()

        lo, hi = np.percentile(y_boot, [16, 84])
        print("%-5s %8.2f %9.3f %8.1f%% %9.2f  %6.1f-%-6.1f %8.3f"
              % (model, y_fit, chi2_obs, 100 * (c_boot < chi2_obs).mean(),
                 np.median(y_boot), lo, hi, np.median(y_boot) / y_fit))

        if model == 'f4':
            print("       minimised chi2 at f4: mean %.3f, median %.3f"
                  " -> 'perfect fit' reduced chi2 = %.3f (mean), %.3f (median)"
                  % (c_boot.mean(), np.median(c_boot),
                     c_boot.mean() / (NB - 1), np.median(c_boot) / (NB - 1)))


if __name__ == '__main__':
    main()
