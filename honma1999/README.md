# Honma (1999) reproduction

Reproduces Honma 1999, ApJ 516, 693 ("The Mass-to-Light Ratio of Binary
Galaxies"): maximum-likelihood M/L for 57 isolated pairs with optical-pair
contamination modelled explicitly.

## Data

`honma_table1_full.dat` is his Table 1 from VizieR (J/ApJ/516/693): 114 rows =
57 pairs with positions, velocities and errors, types, B magnitudes, and his
luminosity-corrected R_p, V_p, (M/L)_p. `Honma99_table1_to_pairs.py` parses it
to `honma99_pairs.csv` and validates by recomputing R_p and V_p from the raw
columns (median ratio 1.001 and 0.993 vs his values).

Unit convention (verified against his (M/L)_p column): all separations and
velocities are normalised by L10^(1/3) with L10 = (L1+L2)/1e10 Lsun, so
(M/L)_p = R_p V_p^2 / (G x 1e10).

## Method notes settled during reproduction

- His nu(R) ~ R^-2.6 is a VOLUME density (Jeans convention): projecting
  p(scalar R) ~ R^-0.6 matches the observed R_p distribution at KS p = 0.96;
  the scalar reading is excluded at p ~ 0. Jeans solution for beta = 0:
  sigma_r^2 = G'(M/L)/((gamma+1) R).
- "Pure spiral pairs" (sample III, N = 30) = both members anything but
  E/S0/Pec — i.e. his "later than Sa" includes Sa. This is the only reading
  of the type column that yields exactly 30.
- His eqs. (17)-(18) are an EXPECTED log-likelihood: the observed pair is
  spread as a Gaussian cloud in V_p and integrated against log p. Implementing
  the (more standard) convolved form log INT g p dV instead lands
  systematically at ~0.6x his M/L; a moment-matching cross-check on the
  median (M/L)_p — estimator-independent — agrees with HIS values, confirming
  the expected-log reading.
- Results are insensitive to the unstated R_max of the separation
  distribution (tested 400-2000 kpc).

## Results (`Honma99_ML.py`, ~150 s)

| sample | orbits | q | ours | 68% | Honma |
|---|---|---|---|---|---|
| I (57)  | isotropic | 0   | 28.8 | 22.6-35.0 | 35 +7-5 |
| I (57)  | isotropic | 1.8 | 31.8 | 24.9-42.5 | 36 +8-4 |
| I (57)  | circular  | 0   | 27.5 | 24.9-30.3 | 28 +5-3 |
| I (57)  | circular  | 1.8 | 28.8 | 26.2-31.8 | 30 +6-3 |
| III (30 spirals) | isotropic | 0   | 12.6 | 9.9-17.7 | 15 +5-3 |
| III (30 spirals) | isotropic | 1.8 | 13.3 | 9.9-17.7 | 16 +4-4 |
| III (30 spirals) | circular  | 0   | 10.9 | 9.4-12.6 | 12 +4-3 |
| III (30 spirals) | circular  | 1.8 | 10.4 | 9.4-12.6 | 12 +3-3 |

Every published value falls inside our 68% interval; the circular-orbit cases
agree to a few per cent. True-pair fractions f reproduce closely (e.g. 0.89
vs his 0.88; 0.73 vs his 0.73). The isotropic cases run ~15% low, possibly a
distribution-function detail (we draw V_los Gaussian at the Jeans dispersion;
he does not state his sampling), still within his quoted errors.

All at H0 = 50 as in the paper; M/L scales as H0.
