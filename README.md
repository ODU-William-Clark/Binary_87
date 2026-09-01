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

## Comparison with Schweizer's published values

Paper III Table 4, `q3` column (M/L = y directly, H0 = 50), against this code
with `USE_R_CONDITIONING = True` and `n_samples = 500000`:

| ecc model | Schweizer M/L | this code | 1-sigma range | Schweizer chi2_nu |
|---|---|---|---|---|
| f1 (circular) | 13 | 8.8 | 7.1 – 15.1 | 0.95 |
| f2 = 2(1−e) | 15 | 9.2 | 8.1 – 19.6 | 0.85 |
| f3 = 1 | 21 | 20.6 | 12.6 – 20.6 | 0.84 |
| f4 = 2e | 24 | 16.6 | 16.6 – 24.1 | 0.62 |
| f5 = δ(0.9) | 32 | 28.5 | 24.6 – 31.6 | 0.81 |

Her published result: **M/L_V = 21 ± 5 for Sc, 39 ± 9 for E** (H0 = 50), with
f4(e) = 2e the preferred eccentricity distribution. Her formal error is **22%**
per solution, so e.g. the f4/q3 entry is 24 ± 5 — most of the differences above
are comparable to that.

Verified as matching her recipe: steps M1–M8, the five eccentricity
distributions (eqs. 43a–43e), the psi relation (eq. 44), the inverse
transformation method, the lognormal f(r_p) (eq. 39), r drawn independently of
orbital phase over 0.01–1.5 Mpc (M5), nb = 5, and dof = nb − 1.

## Reading the chi-square

`Binary_87_Chi_Square_3.py` reports the **minimum** of chi-square over the `y`
scan. A minimum is biased low, so it must not be compared against the degrees
of freedom — **a correct model does not give reduced chi-square ≈ 1 here.**

Schweizer's own Table 4 values run **0.53 to 1.02**, and she selects the model
with the *lowest* chi2_nu (f4, at 0.53), not the one nearest unity. Her
remark that "in chi2 tests with a good fit, chi2_nu should ideally be
approximately equal to unity" is the textbook ideal, not what her analysis
achieves. Do not tune binning to reach 1.

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
- **Minimum-separation cut.** Paper III (bias B5) applies *no* correction for
  pair overlap, so `min_separation_arcsec` is not part of her method. It is
  inert here regardless; set `APPLY_MIN_SEP = False` to drop it.
- **Open discrepancy: the chi2(y) curve.** Schweizer reports her curves as
  "nearly parabolic" and fits a second-order polynomial over `chi2 <= 3*chi2_min`
  (sec. VI[b]iii). Ours are *jagged* — chi2 steps whenever a `u` value crosses
  one of the 5 cell boundaries, giving many local minima within delta-chi2 < 2,
  so the global argmin is unstable and the parabola fit sometimes fails. Things
  tried that did **not** fix it: percentile-based psi histogram ranges
  (`PSI_RANGE_PCT`, 98–99.9), Savitzky-Golay smoothing, and coarsening the y
  grid. The likely real fix is a statistic that varies *continuously* with y —
  Cramer-von Mises or Anderson-Darling on the `u` values, or a direct
  likelihood `sum log f(psi_i | r_p,i, y)` — which removes the jaggedness at
  its source and gives proper confidence intervals.
- **psi histogram range** (`PSI_RANGE_PCT`) is set by the sample maximum by
  default, matching the original code; the paper does not specify it. Note it
  drifts with `n_samples`, so a high percentile is the statistically safer
  choice even though it did not improve agreement with Table 4.
- **Not yet run: her closure test.** Sec. VI(c) generates 10 synthetic samples
  of 43 bound pairs on circular orbits with known M/L and recovers
  `(M/L)_out/(M/L)_in = 1.03 +/- 0.07`, from which her quoted **22% formal
  error** derives. Running the same test here is the right way to decide
  whether the Table 4 differences are bias or noise.
- **M8 rejection constants** (255, 800 kpc) are still unverified against her
  eq. (26).
- **y grid** step is 0.5, which is coarse for `q1` where y ~ 5.
- Statistical choices left as-is and worth revisiting: `nb = 5`, dof of `nb−1`
  (fitting `y` costs one), the parabola fitted over `chi² ≤ 3·chi²_min`, and
  the psi histogram range set by the sample maximum.

## Data

`Binary_gal_87.txt` — 48 southern pairs compiled from Schweizer Papers I & II.
