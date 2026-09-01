import numpy as np
from scipy.interpolate import interp1d

# --- Compute r distribution using Lucy's deconvolution ---
def get_r_distribution(n_bins=10000, r_min=0, r_max=1500, n_iter=5):
    # n_iter = 5 per S87 sec. Ve: "A close approximation to f(r_p) was achieved
    # after cycling through five iterations."
    #
    # The r_p grid must start at 0, not at r_min.  Lucy (1974) permits dropping
    # the normalising denominator only when integral K(r_p|r) dr_p = 1 for every
    # r, and that integral runs from 0 to r.  Truncating it at r_min made the
    # normalisation fail at small r (0.00 at r = r_min, 0.44 at 11.5 kpc), so
    # f(r) was multiplied by a factor much less than 1 on every iteration and
    # never converged -- f(15 kpc) fell monotonically with n_iter.
    r_p_bins = np.linspace(0.0, r_max, n_bins)
    r_p_centers = 0.5 * (r_p_bins[1:] + r_p_bins[:-1])

    f_rp_obs = 416 / r_p_centers * np.exp(-0.63 * (np.log(r_p_centers) - 4.32) ** 2)
    f_rp_obs /= np.trapz(f_rp_obs, r_p_centers)

    r_vals = np.linspace(r_min, r_max, n_bins)
    dr = r_vals[1] - r_vals[0]

    def abel_kernel(rp, r):
        # Strict inequality: with r_p and r now on different spacings, an exact
        # coincidence r == rp gives sqrt(0) -> inf, which poisons the matmul
        # with NaN.  The r == rp point is the integrable singularity and
        # contributes zero measure, so dropping it is safe.
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.where(r > rp, rp / r / np.sqrt(r**2 - rp**2), 0.0)
        return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

    K = np.array([abel_kernel(rp, r_vals) for rp in r_p_centers])
    f_r = np.ones_like(r_vals)
    f_r /= np.trapz(f_r, r_vals)

    for _ in range(n_iter):
        f_proj = K @ f_r * dr
        ratio = np.where(f_proj > 0, f_rp_obs / f_proj, 0)
        correction = K.T @ ratio * (r_p_centers[1] - r_p_centers[0])
        f_r *= correction
        f_r = np.clip(f_r, 0, np.inf)
        f_r /= np.trapz(f_r, r_vals)

    return r_vals, f_r

# Optional utility to return a sampler
def build_r_sampler(r_vals, f_r):
    cdf = np.cumsum(f_r)
    cdf /= cdf[-1]
    return interp1d(cdf, r_vals, bounds_error=False, fill_value=(r_vals[0], r_vals[-1]))






