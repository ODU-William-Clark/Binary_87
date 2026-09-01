import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from functools import lru_cache

# --------------------
# Constants and Settings
# --------------------
H0 = 50  # Hubble constant in km/s/Mpc  (1987 distance scale -- M/L scales as H0)
c = 299792.458  # speed of light in km/s
ARCSEC = np.pi / (180.0 * 3600.0)  # radians per arcsec

# Set False to restore the previous effective behaviour, in which the minimum
# angular separation cut never fired (see min_separation_arcsec below).
APPLY_MIN_SEP = True

OBS_FILE = "Binary_gal_87 - Sheet1.CSV"
M0 = -20.70
beta = -0.03
deg_to_rad = np.pi / 180
fields = 1000
theta_p_max_deg = 1.0
theta_p_max_rad = theta_p_max_deg * deg_to_rad
# (debug print of theta_p_max_rad removed)
mag_limit = 15.0


" WILL NEED TO RE VISIT THE GAMMAS USED TO PREFORM FLYSPANKING OR SOMETHING SIMILAR TO GET BETTER n_galaxies_range"
n_galaxies_range = (50, 150)

# --------------------
# Luminosity Function
# --------------------
def min_separation_arcsec(m_bright, m_faint):
    """Minimum angular separation for a pair to make it into the sample.

    !!! UNVERIFIED -- CHECK AGAINST SCHWEIZER 1987b (ApJS 64, 427) !!!

    The original line read

        if dtheta < 10**(0.72 - 0.13 * m_n - m_br):

    which Python parses as  (0.72 - 0.13*m_n) - m_br.  For realistic
    magnitudes (m_br ~ 12) that exponent is about -12.8, so the threshold was
    ~1e-13 and the cut NEVER FIRED.  The grouping below assumes the intended
    expression was 0.13*(m_n - m_br), i.e. a function of the magnitude
    DIFFERENCE.

    Units were also wrong: dtheta is in radians, while this expression yields
    ~5 arcsec at equal magnitudes.  The caller now converts explicitly.

    Two things still need confirming from Paper III: the grouping, and the
    sign of the magnitude-difference term.  A blending criterion arguably
    wants a LARGER minimum separation for a more unequal pair, which would
    make the term +0.13*(m_n - m_br).
    """
    return 10.0 ** (0.72 - 0.13 * (m_faint - m_bright))


def phi_M(M):
    x = 10 ** (0.4 * (M0 - M))
    return x ** beta * np.exp(-x)

def p_v_given_m(v, m):
    with np.errstate(divide='ignore', invalid='ignore'):
        M = m - 5 * np.log10(v / H0) - 25
    return phi_M(M) * (v**2) *4*np.pi/3/H0

@lru_cache(maxsize=None)
def _velocity_inv_cdf(m_key, v_min=50, v_max=7500, n_points=1000):
    """Inverse CDF of p(v|m), cached on magnitude rounded to 0.01 mag."""
    v_values = np.linspace(v_min, v_max, n_points)
    p_values = p_v_given_m(v_values, m_key)
    cdf = np.cumsum(p_values)
    cdf /= cdf[-1]
    return interp1d(cdf, v_values, bounds_error=False, fill_value=(v_values[0], v_values[-1]))


def sample_velocity(m):
    """Draw one radial velocity for a galaxy of apparent magnitude m."""
    return float(_velocity_inv_cdf(round(float(m), 2))(np.random.uniform(0, 1)))


def precompute_velocity_sampler(m, v_min=50, v_max=7500, n_points=1000):
    v_values = np.linspace(v_min, v_max, n_points)
    p_values = p_v_given_m(v_values, m)
    cdf = np.cumsum(p_values)
    cdf /= cdf[-1]  # normalize
    inv_cdf = interp1d(cdf, v_values, bounds_error=False, fill_value=(v_values[0], v_values[-1]))
    return lambda: float(inv_cdf(np.random.uniform(0, 1)))

# --------------------
# Simulate Galaxy Fields
# --------------------
def generate_fields_with_velocity_sampler(n_fields):
    all_pairs = []
    fail_max_theta = 0
    fail_min_theta = 0
    fail_mag = 0
    fail_edge = 0
    kept = 0

    ra_center, dec_center = 180.0, 0.0  # arbitrary field center

    for _ in range(n_fields):
        N = np.random.randint(*n_galaxies_range)
        galaxies = []
        for _ in range(N):
            # Simulate in ±5° field (10°x10°)
            ra = ra_center + np.random.uniform(-5, 5)
            dec = dec_center + np.random.uniform(-5, 5)
            m = np.random.uniform(9.0, 18.0)
            # One velocity per GALAXY, drawn once at creation.  Previously this
            # was drawn inside the pair loop, so a galaxy got a fresh,
            # independent velocity for every partner it was paired with -- the
            # field was never a self-consistent realisation.
            galaxies.append((ra, dec, m, sample_velocity(m)))

        for i in range(len(galaxies)):
            for j in range(i + 1, len(galaxies)):
                ra1, dec1, m1, v1 = galaxies[i]
                ra2, dec2, m2, v2 = galaxies[j]

                # Check if both galaxies lie in inner 6°x6° field
                if (abs(ra1 - ra_center) > 3 or abs(dec1 - dec_center) > 3 or
                    abs(ra2 - ra_center) > 3 or abs(dec2 - dec_center) > 3):
                    fail_edge += 1
                    continue

                # great-circle angular separation
                ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
                ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)

                cos_dtheta = (
                    np.sin(dec1_rad) * np.sin(dec2_rad) +
                    np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad)
                )
                dtheta = np.arccos(np.clip(cos_dtheta, -1, 1))  # radians

                if dtheta > theta_p_max_rad:
                    fail_max_theta += 1
                    continue

                m_br = min(m1, m2)
                m_f = max(m1, m2)
                m_n = m_f

                if APPLY_MIN_SEP and dtheta < min_separation_arcsec(m_br, m_n) * ARCSEC:
                    fail_min_theta += 1
                    continue

                if m_f > mag_limit:
                    fail_mag += 1
                    continue

                # apparent luminosities
                L1_app = 10**(-0.4 * m1)
                L2_app = 10**(-0.4 * m2)

                # luminosity-weighted mean velocity
                v_avg = (v1 * L1_app + v2 * L2_app) / (L1_app + L2_app)

                # projected separation
                rp = 40 * v_avg * np.tan(dtheta / 2)

                # velocity difference with relativistic correction
                dv = abs(v1 - v2) / (1 + v_avg / c)

                all_pairs.append((rp, dv))
                kept += 1

    print(f"Rejected by max theta: {fail_max_theta}")
    print(f"Rejected by min theta: {fail_min_theta}")
    print(f"Rejected by mag: {fail_mag}")
    print(f"Rejected by edge effect: {fail_edge}")
    print(f"Kept pairs: {kept}")
    return all_pairs
if __name__ == '__main__':
    # Generate simulated galaxy pair distribution
    simulated_pairs = generate_fields_with_velocity_sampler(fields)
    df_simulated = pd.DataFrame(simulated_pairs, columns=["r", "v"])

    # Load and Process Observed Data
    df_observed = pd.read_csv(OBS_FILE)
    df_observed = df_observed.dropna(subset=["r", "v"])
    df_observed = df_observed[df_observed["v"] <= 7000]

    # Define velocity bins according to paper
    bins = [(0, 200), (200, 1000), (1000, 7000)]
    P_N = 5.3 / len(df_observed)  # as per paper

    # Bin counts for simulated and observed data
    sim_counts = []
    obs_counts = []
    for vmin, vmax in bins:
        sim_count = ((df_simulated["v"] >= vmin) & (df_simulated["v"] < vmax)).sum()
        obs_count = ((df_observed["v"] >= vmin) & (df_observed["v"] < vmax)).sum()
        sim_counts.append(sim_count)
        obs_counts.append(obs_count)

    # Total simulated optical pairs
    total_sim = len(df_simulated)

    # Compute P(B|C) for each bin (no scaling of probabilities)
    P_B_given_C_bins = []
    for i in range(len(bins)):
        P_C = obs_counts[i] / len(df_observed)
        print(total_sim)
        P_CN = sim_counts[i] / total_sim
        P_BC = 1 - (P_CN * P_N) / P_C if P_C > 0 else 0
        P_B_given_C_bins.append(P_BC)

    # Assign probabilities to observed data
    P_B_given_C = []
    for v in df_observed["v"]:
        if v < 200:
            P_B_given_C.append(P_B_given_C_bins[0])
        elif v < 1000:
            P_B_given_C.append(P_B_given_C_bins[1])
        else:
            P_B_given_C.append(P_B_given_C_bins[2])

    df_observed["P(B|C)"] = P_B_given_C

    # Save results to a text file
    output_filename = "Binary_gal_97_with_probabilities.txt"
    df_observed.to_csv(output_filename, index=False, sep='\t')

    # Plot Δv vs r_p for simulated data (linear scales)
    plt.figure(figsize=(8,6))
    plt.scatter(df_simulated["r"], df_simulated["v"], alpha=0.5, s=10, label="Simulated Pairs")
    plt.xlabel(r"Projected Separation $r_p$ [kpc]", fontsize=12)
    plt.ylabel(r"Velocity Difference $\Delta v$ [km/s]", fontsize=12)
    plt.title(r"Simulated Galaxy Pairs: $\Delta v$ vs $r_p$", fontsize=14)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
