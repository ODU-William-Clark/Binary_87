"""Calibrate the minimised chi^2 statistic by parametric bootstrap.

The value reported by Binary_87_Chi_Square_3 is the MINIMUM of chi^2 over the
y scan, not a single random draw of the statistic.  A minimum is biased low,
so it must NOT be compared against dof: a perfect model does not produce
reduced chi^2 = 1 here.

This script generates synthetic samples from the model at a chosen y, runs the
identical y scan on each, and reports the empirical distribution of chi2_min.
Judge the real chi2_min against THAT, not against nb-1.

Only q3 (q == 1 for all morphological types) is implemented; extend `S` below
for q1/q2.
"""
F_MODEL = 'f4'      # eccentricity model to calibrate
Y_TRUE  = 21.72     # best-fit y for that model, from best_fit_y_results.csv
N_BOOT  = 400
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

nb, NBINS = 5, 50
sim = np.load("simulated_psi_data_test.npz", allow_pickle=True)
sim_psi = np.array(sim['filtered_psi_proj'].item()[F_MODEL])

d = pd.read_csv("Binary_gal_87.txt")
d = d[pd.to_numeric(d['P'], errors='coerce') > 0.5]
P = pd.to_numeric(d['P']).values
S = (pd.to_numeric(d['L1']).values + pd.to_numeric(d['L2']).values) * 1e10   # q3 == 1
sig = []
for _, r in d.iterrows():
    try: sig.append(np.sqrt(float(r['sigma1'])**2 + float(r['sigma2'])**2))
    except (TypeError, ValueError): sig.append(9.0)
sig = np.array(sig); N = len(P)

hist, edges = np.histogram(sim_psi, bins=NBINS, range=(0, sim_psi.max()*1.2))
bc = 0.5*(edges[:-1]+edges[1:]); dx = bc[1]-bc[0]
pdf = hist/np.sum(hist*dx)

ygrid = np.linspace(0.1, 500, 1000); NY = len(ygrid)
M = ygrid[None, :]*S[:, None]                       # (N, NY)
sb = (sig[:, None]/np.sqrt(M))/dx

cdf_tab = np.empty((N, NY, NBINS))
for i in range(N):
    for j in range(NY):
        pc = gaussian_filter1d(pdf, sigma=sb[i, j]); pc /= np.sum(pc*dx)
        c = np.cumsum(pc)*dx; cdf_tab[i, j] = c/c[-1]

def chi2_min(dv):
    psi_obs = dv[:, None]/np.sqrt(M)
    t = np.clip((psi_obs - bc[0])/dx, 0, NBINS-1.0001)
    lo = t.astype(int); fr = t - lo
    I, J = np.indices((N, NY))
    u = cdf_tab[I, J, lo]*(1-fr) + cdf_tab[I, J, lo+1]*fr
    kb = np.clip((u*nb).astype(int), 0, nb-1)
    W = np.zeros((nb, NY)); np.add.at(W, (kb, J), P[:, None])
    E = W.sum(0)/nb
    return (((W-E)**2)/E).sum(0).min()

y_true = Y_TRUE
rng = np.random.default_rng(7)
out = []
for b in range(N_BOOT):
    psi_p = rng.choice(sim_psi, size=N)
    dv = np.abs(psi_p*np.sqrt(y_true*S) + rng.normal(0, sig))
    out.append(chi2_min(dv))
out = np.array(out)
print("Parametric bootstrap, f4/q3, 400 synthetic samples drawn from the model at y=%.2f" % y_true)
print("  distribution of chi2_MIN over the y scan:")
print("    mean=%.3f  median=%.3f  5th=%.3f  95th=%.3f" % (out.mean(), np.median(out), *np.percentile(out, [5, 95])))
print("  reduced (dividing by nb-1=4): mean=%.3f  median=%.3f" % (out.mean()/4, np.median(out)/4))
print()
print("  Unminimised null (single random y) had mean 3.88.")
print("  Your observed chi2_min for f4,q3 = 0.399  ->  percentile %.1f%% of this distribution"
      % (100*(out < 0.399).mean()))
