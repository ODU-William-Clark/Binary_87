import time

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from Binary_87_r_distro import get_r_distribution, build_r_sampler
from Binary_87_psi import sample_psi

G = 4.30091e-6  # kpc * (km/s)^2 / M_sun
ecc_models = ['f1', 'f2', 'f3', 'f4', 'f5']
n_samples = 500000

# --- M8 selection function -------------------------------------------------
# S87 eq. (25) gives the RECIPROCAL correction factor for physical pairs that
# were accidentally excluded, nu^-1(r_p) proportional to dex(-0.0017 r_p);
# eq. (26) writes it as nu(r_p) = c exp(r_p / 255 kpc).  Since nu is the factor
# by which observed counts are scaled UP, the acceptance probability is its
# reciprocal:
#
#     P_accept(r_p) = exp(-r_p / 255 kpc)
#
# The previous form, P_accept = 1 - exp((r_p - 800)/255), was nearly flat out
# to an invented hard wall at 800 kpc (no such cutoff appears in the paper) and
# removed only ~8 per cent of pairs where eq. (26) removes ~31 per cent.
M8_SCALE_KPC = 255.0


def simulate_model(model, r_vals, f_r):
    r_sampler = build_r_sampler(r_vals, f_r)
    r = r_sampler(np.random.uniform(0, 1, size=n_samples))
    psi, phi = sample_psi(model, r, n_samples=n_samples)

    # --- Projection onto ONE shared line of sight --------------------------
    # Work in the orbital frame with the separation vector along x.  The
    # velocity then lies at angle (90 deg - phi) from it, still in the plane:
    #     r_hat   = (1, 0, 0)
    #     psi_hat = (sin phi, cos phi, 0)
    # Drawing the observer direction n_hat isotropically is equivalent to
    # drawing a random orbit orientation.
    #
    # Both r and psi MUST be projected with the same n_hat -- there is one
    # line of sight per system.  The previous version drew two independent
    # angles, which destroyed the r_p-psi_p correlation the chi^2 fit relies
    # on, and drew them uniform in angle rather than uniform on the sphere.
    cos_t = np.random.uniform(-1, 1, size=n_samples)
    sin_t = np.sqrt(1.0 - cos_t**2)
    alpha = np.random.uniform(0, 2 * np.pi, size=n_samples)
    n_x = sin_t * np.cos(alpha)
    n_y = sin_t * np.sin(alpha)

    # projected separation = component of r perpendicular to the line of sight
    r_proj = r * np.sqrt(1.0 - n_x**2)
    # observed velocity difference = component of psi along the line of sight
    psi_proj = psi * np.abs(np.sin(phi) * n_x + np.cos(phi) * n_y)

    # --- Apply M8: acceptance falls exponentially with projected separation --
    accept_probs = np.exp(-r_proj / M8_SCALE_KPC)
    keep_mask = np.random.rand(n_samples) < accept_probs
    r_proj_kept = r_proj[keep_mask]
    psi_proj_kept = psi_proj[keep_mask]

    return model, psi, r, r_proj, psi_proj, r_proj_kept, psi_proj_kept


if __name__ == '__main__':
    start_time = time.perf_counter()

    # Step 1: Get the reconstructed r distribution.
    # Computed ONCE here and passed to the workers.  Previously this sat at
    # module scope, so every child process rebuilt the ~800 MB Abel kernel.
    r_vals, f_r = get_r_distribution(r_min=10, r_max=1500)

    # Run simulations in parallel
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_model, model, r_vals, f_r) for model in ecc_models]
        for future in futures:
            model, psi, r, r_proj, psi_proj, r_proj_kept, psi_proj_kept = future.result()
            results[model] = {
                "psi": psi,
                "r": r,
                "r_proj": r_proj,
                "psi_proj": psi_proj,
                "r_proj_kept": r_proj_kept,
                "psi_proj_kept": psi_proj_kept
            }

    # Save data
    np.savez_compressed(
        "simulated_psi_data_test.npz",
        psi_samples={k: results[k]["psi"] for k in results},
        r_samples={k: results[k]["r"] for k in results},
        r_proj_samples={k: results[k]["r_proj"] for k in results},
        psi_proj_samples={k: results[k]["psi_proj"] for k in results},
        filtered_r_proj={k: results[k]["r_proj_kept"] for k in results},
        filtered_psi_proj={k: results[k]["psi_proj_kept"] for k in results}
    )

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    for m in ecc_models:
        print(f"  {m}: kept {len(results[m]['r_proj_kept'])}/{n_samples} after M8")

    # Plotting f(r)
    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, f_r * 100, 'k--', label='Estimated f(r) x 10^2')
    plt.xlabel("Separation r (kpc)")
    plt.ylabel("Probability Density x 10^2 (kpc^-1)")
    plt.title("Reconstructed f(r) Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot psi distributions
    plt.figure(figsize=(10, 6))
    for model in ecc_models:
        plt.hist(results[model]["psi"], bins=100, alpha=0.6, label=f'{model}')
    plt.xlabel(r"psi [km/s / $\sqrt{M_\odot}$]")
    plt.ylabel("Count")
    plt.title("Distribution of psi for Different Eccentricity Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Scatter plots before and after rejection
    for tag, kx, ky in [("Before", "r_proj", "psi_proj"), ("After", "r_proj_kept", "psi_proj_kept")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, model in enumerate(ecc_models):
            ax = axes[i]
            ax.scatter(results[model][kx], results[model][ky], s=5, alpha=0.5)
            ax.set_title(f"{tag} Rejection - f{i+1}: {model}")
            ax.set_xlabel("$r_p$ (kpc)")
            ax.set_ylabel(r"psi_p [km/s / $\sqrt{M_\odot}$]")
            ax.grid(True)
        if len(ecc_models) < len(axes):
            axes[-1].axis('off')
        plt.suptitle(f"{tag} M8: $r_p$ vs. psi_p by Eccentricity Model")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
