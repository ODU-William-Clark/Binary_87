# Binary_87

Mass-to-light ratios for binary galaxies, reproducing the method of
**Schweizer (1987), ApJS 64** — Paper I (411, radial velocities), Paper II
(417, BV photometry), Paper III (427, analysis and sample selection).
Works in K-magnitudes and B−V colour indices.

## Method

For a Kepler orbit `v² = GM(1 + e·cos E)/r`, so

    psi = |v| / sqrt(M) = sqrt( G(1 + e·cos E) / r )

is independent of the total mass. The pipeline builds the projected
distribution of `psi_p` by Monte Carlo, then for each observed pair forms

    u = CDF_sim( psi_obs ),   psi_obs = dv / sqrt(M),  M = y·(q(t1)·L1 + q(t2)·L2)

If `y` is correct the `u` values are uniform on [0,1]. A chi-square test of
uniformity is scanned over `y`, and the minimum taken as the best fit.

## Pipeline

| Step | Script |
|---|---|
| 1. r_p, v_p, luminosities from raw coordinates and magnitudes | `Binary_87_luminosity_calculator.py` |
| 2. Probability that an identified pair is a true binary | `Binary_87_probabilites.py` |
| 3. Deprojected separation distribution f(r) (Lucy deconvolution) | `Binary_87_r_distro.py` |
| 4. psi sampling from Kepler orbits | `Binary_87_psi.py` |
| 5. Monte Carlo projected psi_p distributions | `Binary_87_Para_Main_2.py` |
| 6. M/L scan over eccentricity and morphology models | `Binary_87_ML_8.py` |
| 7. Best-fit y and chi-square per model | `Binary_87_Chi_Square_3.py` |
| 8. Calibrate the minimised chi-square by bootstrap | `Binary_87_calibrate_chi2.py` |

Run `Binary_87_Para_Main_2.py` before `Binary_87_ML_8.py`.

## Reading the chi-square

`Binary_87_Chi_Square_3.py` reports the **minimum** of chi-square over the `y`
scan. A minimum is biased low, so it must not be compared against the degrees
of freedom — **a correct model does not give reduced chi-square ≈ 1 here.**

A parametric bootstrap (`Binary_87_calibrate_chi2.py`, f4/q3, 400 synthetic
samples drawn from the model at its own best-fit y) gives:

| statistic | value |
|---|---|
| unminimised null, mean | 3.88 (≈ nb−1 = 4, so the weighting is fine) |
| minimised chi², mean | 2.03 |
| minimised chi², median | 1.55 |
| implied "perfect fit" reduced chi² | **≈ 0.39 (median), 0.51 (mean)** |

Judge the observed chi²_min against that distribution, not against 1.

## Caveats

- **Distance scale.** `H0 = 50 km/s/Mpc` throughout (the 1987 value). M ∝ 1/H0
  and L ∝ 1/H0², so **M/L ∝ H0** — multiply by ~1.4 for H0 = 70.
- **Colour band.** The distributions and M/L values are specific to the
  magnitude/colour band used here; other bands need different inputs.
- **Unverified selection cut.** `min_separation_arcsec` in
  `Binary_87_probabilites.py` needs checking against Paper III — both the
  grouping of the exponent and the sign of the magnitude-difference term. It
  is currently inert (simulated pairs never land within ~5 arcsec).
- **r and orbital phase** are sampled independently in `Binary_87_psi.py`; for
  a true orbit `r = a(1 − e·cos E)` links them. Inherited from the original
  implementation, still to be checked against Paper III.
- Statistical choices left as-is and worth revisiting: `nb = 5`, dof of `nb−1`
  (fitting `y` costs one), the parabola fitted over `chi² ≤ 3·chi²_min`, and
  the psi histogram range set by the sample maximum.

## Data

`Binary_gal_87.txt` — 48 southern pairs compiled from Schweizer Papers I & II.
