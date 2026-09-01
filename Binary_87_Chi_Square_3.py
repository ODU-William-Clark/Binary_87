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

        chi_min = subset["chi_squared"].min()
        filtered = subset[subset["chi_squared"] <= 3 * chi_min]

        if len(filtered) < 3:
            continue  # Avoid fitting if too few points

        coeffs = np.polyfit(filtered["y"], filtered["chi_squared"], 2)
        a, b, c = coeffs
        y_best = -b / (2 * a)
        chi_best = a * y_best**2 + b * y_best + c
        reduced_chi_best = chi_best / (nb - 1)

        best_y_results.append({
            "f_model": f_model,
            "q_model": q_model,
            "y_best": y_best,
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
