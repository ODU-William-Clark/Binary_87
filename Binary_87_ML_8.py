import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import time

'''
This code uses histograms instead of KDEs.

For each observed pair it forms u = CDF_sim(psi_obs).  If the assumed M/L
scaling y is correct the u values are uniform on [0,1]; chi^2 against
uniformity is scanned over y and the minimum taken as the best fit.
'''
# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
nb = 5              # number of u bins for the uniformity chi^2 (best 10)
NBINS_PSI = 50      # bins used to represent the simulated psi_p pdf (best 50)
G = 4.30091e-6      # kpc * (km/s)^2 / M_sun
DEFAULT_SIGMA_V = 9.0   # km/s, Schweizer 1987 Paper I formal accuracy

# Condition the simulated psi_p distribution on the pair's projected
# separation.  psi ~ r^(-1/2), so r_p is the strongest single predictor of
# psi_p and ignoring it both weakens the uniformity test and biases it if the
# observed and simulated r_p distributions differ.
# Left False to reproduce the previous behaviour; set True to switch it on.
USE_R_CONDITIONING = True

# Upper edge of the psi histogram, as a percentile of the simulated sample.
# Set to None for the legacy behaviour (max * 1.2), which is NOT recommended:
# psi has a heavy tail (psi ~ r^-1/2), so the sample maximum -- and therefore
# the bin width over the bulk -- drifts with n_samples.  A statistic that
# moves when you add Monte Carlo samples is not measuring the data.
_p = os.environ.get("PSI_RANGE_PCT", "none")
PSI_RANGE_PCT = None if _p.lower() in ("none", "legacy") else float(_p)
N_R_BINS = 8
MIN_SIM_PER_BIN = 15

# --------------------------------------------
# Load Simulated Data
# --------------------------------------------
sim_data = np.load("simulated_psi_data_test.npz", allow_pickle=True)
psi_proj_samples = sim_data['filtered_psi_proj'].item()
r_proj_samples = sim_data['filtered_r_proj'].item()

# --------------------------------------------
# Load Observed Data
# --------------------------------------------
df_obs = pd.read_csv("Binary_gal_87.txt")

required_cols = ["Pair", "type1", "type2", "theta", "r", "v", "L1", "L2", "M", "P", "sigma1", "sigma2"]
for col in required_cols:
    if col not in df_obs.columns:
        raise KeyError(f"Missing required column: {col}")
df_obs = df_obs[df_obs["P"] > 0.5]

for col in ["type1", "type2", "r", "v", "L1", "L2", "M", "P"]:
    df_obs[col] = pd.to_numeric(df_obs[col], errors='raise')

# Pull the per-row quantities out of the DataFrame once, instead of calling
# df.iterrows() inside the y scan (5 x 3 x 1000 times over).
obs = []
for _, row in df_obs.iterrows():
    try:
        sigma_v = np.sqrt(float(row["sigma1"])**2 + float(row["sigma2"])**2)
    except (TypeError, ValueError):
        sigma_v = DEFAULT_SIGMA_V
    obs.append(dict(dv=row["v"], r=row["r"], L1=row["L1"] * 1e10, L2=row["L2"] * 1e10,
                    t1=row["type1"], t2=row["type2"], P=row["P"], sigma_v=sigma_v))

# --------------------------------------------
# Morphology Functions
# --------------------------------------------
def q1(t): return 6.47 - 0.39 * t
def q2(t): return np.where(t <= 0, 2.0, 1.0)
def q3(t): return np.ones_like(t)
q_funcs = {'q1': q1, 'q2': q2, 'q3': q3}

# --------------------------------------------
# Main Loop
# --------------------------------------------
results = []
start_time = time.perf_counter()

for model in psi_proj_samples:
    print(model)
    sim_r = np.array(r_proj_samples[model])
    sim_psi = np.array(psi_proj_samples[model])

    # --- Build the simulated psi_p pdf(s) ONCE per model --------------------
    # These depend only on `model`, never on q, y or the observed row.  The
    # previous version rebuilt an identical 50-bin histogram of ~10,000 values
    # inside the innermost loop -- 5 x 3 x 1000 x n_rows times.
    if USE_R_CONDITIONING:
        r_bins = np.quantile(sim_r, np.linspace(0, 1, N_R_BINS + 1))
        r_bins[0], r_bins[-1] = -np.inf, np.inf
        groups = [sim_psi[(sim_r >= r_bins[k]) & (sim_r < r_bins[k + 1])]
                  for k in range(N_R_BINS)]
    else:
        r_bins = np.array([-np.inf, np.inf])
        groups = [sim_psi]

    pdfs = []
    for g in groups:
        if len(g) < MIN_SIM_PER_BIN:
            pdfs.append(None)
            continue
        hi = np.max(g) * 1.2 if PSI_RANGE_PCT is None else np.percentile(g, PSI_RANGE_PCT)
        hist, edges = np.histogram(g, bins=NBINS_PSI, range=(0, hi))
        bc = 0.5 * (edges[:-1] + edges[1:])
        d = bc[1] - bc[0]
        pdfs.append((bc, d, hist / np.sum(hist * d)))

    for q_label, q_func in q_funcs.items():
        qA_all = np.array([q_func(np.array([o["t1"]]))[0] for o in obs])
        qB_all = np.array([q_func(np.array([o["t2"]]))[0] for o in obs])

        for y in np.linspace(0.1, 500, 1000):
            n_over = 0
            u_values = []
            p_values = []

            for i, o in enumerate(obs):
                M = y * (qA_all[i] * o["L1"] + qB_all[i] * o["L2"])
                if M <= 0:
                    continue

                k = int(np.clip(np.digitize(o["r"], r_bins) - 1, 0, len(pdfs) - 1))
                entry = pdfs[k]
                if entry is None:
                    continue
                bin_centers, dx, pdf = entry

                psi_obs = o["dv"] / np.sqrt(M)
                sigma_bins = (o["sigma_v"] / np.sqrt(M)) / dx

                pdf_convolved = gaussian_filter1d(pdf, sigma=sigma_bins)
                pdf_convolved /= np.sum(pdf_convolved * dx)
                cdf = np.cumsum(pdf_convolved) * dx
                cdf /= cdf[-1]

                cdf_interp = interp1d(bin_centers, cdf, bounds_error=False, fill_value=(0, 1))
                if psi_obs > bin_centers[-1]:
                    n_over += 1
                u_values.append(cdf_interp(psi_obs))
                p_values.append(o["P"])

            if len(u_values) >= nb:
                hist_weighted, _ = np.histogram(u_values, bins=nb, range=(0, 1), weights=p_values)
                total_weight = np.sum(hist_weighted)
                expected_weight = total_weight / nb
                chi_sq = np.sum((hist_weighted - expected_weight) ** 2 / expected_weight)
                results.append((model, q_label, y, chi_sq, n_over))

# --------------------------------------------
# Save Results
# --------------------------------------------
results_df = pd.DataFrame(results, columns=["f_model", "q_model", "y", "chi_squared", "n_over"])
results_df["reduced_chi_squared"] = results_df["chi_squared"] / (nb - 1)
results_df["nb"] = nb
results_df.to_csv("ML_fit_results.csv", index=False)
print("Analysis complete. Results with reduced chi-square saved to ML_fit_results.csv")
print(f"Elapsed time: {time.perf_counter() - start_time:.4f} seconds")
