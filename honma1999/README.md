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

## Sample selection: data access and the catalogue-epoch problem

- **NED bulk access**: `ned_tap_fetch.py` queries NED's TAP service
  (`ned.ipac.caltech.edu/tap`, table `NEDTAP.objdir`) with asynchronous ADQL
  jobs. NED aborts any job at ~60 s, so large selections are chunked into thin
  redshift slices. The full cz = 840-4676 km/s shell is 43,572 galaxies in 8
  slices (~9 requests). TAP carries NO photometry, and NED's legacy per-object
  CGI currently fails ("EGRET error"), so magnitudes come from **HyperLEDA**
  (`leda_photometry_fill.py`; `btc` is corrected total B on the RC3 B_T^0
  system). `rc3_photometry_supplement.csv` holds the 13 Honma-pair members
  RC3 lacks photometry for (NGC 2979 has no B in HyperLEDA either and uses
  Honma's tabulated value).

- **Catalogue drift since 1999 is large, not cosmetic.** Of the 43,572
  current NED redshifts in the shell, only 4,426 (10%) cite a source published
  by 1999; 90% of preferred redshifts postdate the paper. The finder has a
  `Z_BIBCODE_MAX_YEAR` switch to restrict to <=1999 sources, but this is a
  LOWER bound on the 1999 catalogue: NED replaces its preferred redshift when
  a better measurement appears, so galaxies Honma demonstrably used (NGC 5899/
  5900, IC 4888/4889, NGC 1134/IC 267 are in his Table 1) now carry post-1999
  bibcodes and are wrongly dropped in 1999 mode. The two modes therefore
  BRACKET his selection epoch:

  | mode | parent | pairs | of Honma's 57 | extras |
  |---|---|---|---|---|
  | 2026 (all z) | 4,127 | 56 | 15 | 41 |
  | 1999 (bibcode <= 1999) | 3,336 | 41 | 11 | 30 |

  Of the 56 pairs in 2026 mode, 18 depend on at least one post-1999 preferred
  redshift; 15 of those are extras -- pairs that could not have been selected
  in 1999. Catalogue growth therefore accounts for roughly a third of the
  extras; the remainder trace to the isolation-criterion reading (still open)
  and companion-pool depth.
