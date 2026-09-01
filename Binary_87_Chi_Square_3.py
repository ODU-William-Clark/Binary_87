import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load the data
df = pd.read_csv("ML_fit_results.csv")
nb = min(df['nb'])
f_models = sorted(df['f_model'].unique())
q_models = sorted(df['q_model'].unique())

best_y_results = []

for f_model in f_models:
    for q_model in q_models:
        subset = df[(df['f_model'] == f_model) & (df['q_model'] == q_model)]

        subset = subset.sort_values("y")
        chi_min = subset["chi_squared"].min()

        # Schweizer 1987b, sec. VI(b)iii: "we fitted second-order polynomials
        # to all points within 3 times the value of the minimum.  The minimum
        # of the fitted polynomial was then designated as the 'best' value of
        # y."  That window is kept here for fidelity.
        #
        # She reports the chi2(y) curves as "nearly parabolic".  Ours are not:
        # chi2 is a step function of y (it jumps whenever a u value crosses one
        # of the 5 cell boundaries) and shows many local minima within
        # delta-chi2 < 2.  The parabola fit therefore fails on some models --
        # a<=0, or a vertex off the grid.  Those cases fall back to the raw
        # argmin rather than emitting a nonsense value (the unguarded version
        # returned a NEGATIVE M/L for f1).  This is the main open discrepancy
        # with the paper; see README.
        y_raw = subset.loc[subset["chi_squared"].idxmin(), "y"]
        filtered = subset[subset["chi_squared"] <= 3 * chi_min]

        if len(filtered) < 3:
            continue

        # delta-chi2 = 1 interval on y, reported as the uncertainty
        near = subset[subset["chi_squared"] <= chi_min + 1.0]
        y_lo, y_hi = near["y"].min(), near["y"].max()

        coeffs = np.polyfit(filtered["y"], filtered["chi_squared"], 2)
        a, b, c = coeffs
        y_best = -b / (2 * a) if a > 0 else y_raw
        if not (filtered["y"].min() <= y_best <= filtered["y"].max()):
            y_best = y_raw
        chi_best = max(a * y_best**2 + b * y_best + c, chi_min)
        reduced_chi_best = chi_best / (nb - 1)

        best_y_results.append({
            "f_model": f_model,
            "q_model": q_model,
            "y_best": y_best,
            "y_raw_argmin": y_raw,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "chi_squared_best": chi_best,
            "reduced_chi_squared_best": reduced_chi_best
        })

        # Main plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(subset["y"], subset["chi_squared"], s=1, alpha=0.5, label="All points")
        ax.scatter(filtered["y"], filtered["chi_squared"], s=10, color='black', label="Used for fit")

        y_fit = np.linspace(filtered["y"].min(), filtered["y"].max(), 500)
        chi_fit = a * y_fit**2 + b * y_fit + c
        ax.plot(y_fit, chi_fit, label="Quadratic fit", color='orange')
        ax.axvline(y_best, color='purple', linestyle='--', label=f"Best y = {y_best:.2f}")

        ax.set_title(f"χ² vs y for {f_model}, {q_model}")
        ax.set_xlabel("y (M/L scaling factor)")
        ax.set_ylabel("χ²")
        ax.legend()

        # Inset (zoomed in on fitting region)
        ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper right")
        ax_inset.scatter(filtered["y"], filtered["chi_squared"], s=10, color='black')
        ax_inset.plot(y_fit, chi_fit, color='orange')
        ax_inset.axvline(y_best, color='purple', linestyle='--')
        ax_inset.set_xlim(filtered["y"].min(), filtered["y"].max())
        ax_inset.set_ylim(filtered["chi_squared"].min(), filtered["chi_squared"].max())
        ax_inset.set_xlabel("y", fontsize=8)
        ax_inset.set_ylabel("χ²", fontsize=8)
        ax_inset.tick_params(axis='both', labelsize=6)
        ax_inset.set_title("Zoomed Fit", fontsize=8)
        plt.show()

# Save best-fit results
best_df = pd.DataFrame(best_y_results)
best_df.to_csv("best_fit_y_results.csv", index=False)
print("Saved best-fit y values to 'best_fit_y_results.csv'")
