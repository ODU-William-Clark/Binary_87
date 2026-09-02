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

## Isolation criteria 5-6: the reading is RAW physical units

All 57 of Honma's pairs survived his own isolation cut, so the correct reading
must keep them. Scoring four candidate readings against his sample
(`Honma99_isolation_test.py`; companion pool = RC3 magnitudes + epoch-limited
NED velocities, in-shell known-z companions plus redshift-unknown ones):

| reading | kept of 57 (1999 pool) | (2026 pool) |
|---|---|---|
| literal: r/L^(1/3) and dv/L^(1/3) vs a,b x 400 | 23 | 26 |
| raw velocity only | 28 | 33 |
| **both raw: r vs 1000 kpc, dv vs 600 km/s** | **43** | **45** |
| 2.0 finder's combined-luminosity norm | 15 | 17 |

Despite the L^(1/3) scaling typeset in his eqs. 5-6, the implemented criterion
used fixed thresholds. (Every worked example in his text sets L10 = 1, where
the readings coincide; luminous pairs like NGC 7537/7541, L10 = 13, expose the
difference.) Residual kills are borderline magnitude-threshold cases and
anonymous RC3 objects whose 1999 redshift status is unknowable from current
NED. With this reading the finder recovers **42 of his 57** (2026 mode; 33 in
1999 mode).

## The remaining structural gap: companion-pool depth

The raw reading is permissive, and with only RC3 as the blocker pool the
finder now returns 335 pairs against his 57. His 1999 NED blocked far more.
Upper-bound test (NED shell galaxies as blockers, no magnitude threshold
available since TAP has no photometry): 1999-epoch NED blockers would kill
183/293 extras but also 21/42 true matches; 2026 blockers would kill 289/293
extras AND 39/42 of his own pairs -- **by the modern catalogue almost nothing
from 1999 is still "isolated" at these thresholds**, which quantifies the
catalogue-epoch caveat in the section above. Making this faithful needs
magnitudes for the blocker pool (the threshold is m3 <= m_pair + 2): the next
step is a bulk HyperLEDA pull (v in shell + bt) to build a complete companion
catalogue with photometry in one query.

## 1999 sample reconstruction on HyperLEDA (v2.2): status

`Honma99_finder_leda.py` rebuilds the selection on a complete
magnitude-bearing catalogue: every HyperLEDA galaxy with btc <= 15.8
(102,800; `btc` is the RC3 B_T^0 system Honma used), pulled in 12 longitude
slices through HyperLEDA's fG.cgi interface. The 1999 epoch is defined by
which galaxies had a redshift by then: RC3 native velocities, the CfA ZCAT
June 1995 (VII/193, `zcat95_parse.py`), and NED <=1999 bibcodes. Galaxies not
in that set become redshift-unknown: not eligible as members, blocking only
by projection. Parent: **5,918 members vs his 6,475** -- the closest match
yet.

| mode | parent | pairs | of his 57 | extras |
|---|---|---|---|---|
| 2026 (HyperLEDA velocities) | 7,480 | 481 | 40 | 441 |
| 1999 (RC3 + ZCAT95 + NED<=99) | 5,918 | 294 | 37 | 257 |

**What matches his paper.** The blind (redshift-unknown) rejection step
removes 27% of the candidates that reach it; he reports ~30%. The pair cuts
are confirmed L-corrected (13 of his pairs exceed 400 kpc raw). The raw
isolation reading keeps 43-45/57 of his pairs; a = 1.5 keeps 52.

**What does not, and was tested.** After known-z isolation we hold 615
candidates where his numbers imply ~95. Ruled out: out-of-shell galaxies as
blockers (keeps 38/57, worse); velocity-blind blocking (36/57); raw vs
corrected magnitudes (242 vs 294 pairs, same recovery). Partial
discriminators: his pairs are mutual nearest neighbours 89% of the time vs
54% of our extras; and his sample spans v_bar = 1513-3653 km/s only, with
zero pairs beyond 3653 although his stated cut is 4500 and volume alone would
put ~half the sample there -- a 1999 redshift-completeness signature that
no current catalogue reproduces. Our extras are systematically wider and
faster (median R_p 172 vs 94 kpc; V_p 47 vs 26 km/s). Applying both empirical
filters (his velocity range + mutual nearest neighbour) gives 148 pairs with
31 of his -- still ~2.5x his count.

**Conclusion.** The documented procedure reproduces a core of ~37-40 of his
57 pairs and his parent size, but his final selectivity is not derivable
from the paper; the undocumented remainder is most plausibly the NED-1999
redshift catalogue's depth profile, which no present-day source preserves.
The modern-data study should therefore be framed as a NEW selection on a
version-stamped catalogue rather than as an extension of his sample.

## Re-read of the paper: two misreadings found and fixed

A line-by-line re-read against the 19-20 missed pairs (`Honma99_miss_audit.py`
traces each one through the pipeline) found two things we had wrong:

1. **Velocity frame.** Honma says "heliocentric velocities" and never applies
   a Local Group correction. His Table 1 velocities match RC3 heliocentric
   values to MAD 13 km/s (vs 133 km/s for LG-corrected). We had been applying
   the Yahil et al. correction (up to +/-308 km/s) to the 1000-4500 window,
   distances and companion velocities. Fixed (`FRAME='helio'` default).
2. **Epoch-match tolerance.** ZCAT-95 positions are coarse B1950 values;
   NGC 1266 sits 0.036 deg from its ZCAT entry and was being counted as
   redshift-unknown. Tolerance raised to 0.05 deg. Parent is now **6,137
   members vs his 6,475**.

And one thing we can bracket but not settle -- **the magnitude system of the
companion pool**. The isolation killers of his pairs cluster suspiciously
close to the m_pair+2 threshold (0.01, 0.02, 0.03, 0.06, 0.08, 0.09 mag
above it), which says his companion magnitudes ran fainter than HyperLEDA's
corrected btc, i.e. a 1999 blind NED search returned raw magnitudes. But
using raw bt for all companions collapses the blind-rejection rate to 7%
against his ~30% and inflates extras; a hybrid (btc for RC3 galaxies, raw
otherwise -- his "NED, supplied with RC3") gives 9%. His companion photometry
was evidently heterogeneous in a way no single rule reproduces:

| companions | blind-step rejection (his ~30%) | of his 57 | pairs |
|---|---|---|---|
| btc (corrected) | 23% | 38 | 328 |
| hybrid | 9% | 43 | 447 |
| raw bt | 7% | 40 | 465 |

**Ceiling on recovery.** Of the remaining misses, five are pairs with a
genuine bright in-shell companion inside 1000 kpc and 600 km/s that he kept
anyway (NGC 7185/7188 with NGC 7180 at 170 kpc; NGC 5198/5173 with NGC 5169;
NGC 7444/7443 with NGC 7450; NGC 5916/5915 with NGC 5916A at 67 kpc, dv 8;
NGC 6484/UGC 11029 with UGC 11027) -- violations of his own stated criterion
under any reading, presumably companions whose 1999 magnitude or redshift
status differed. Two are photometric-source differences on the members
(ESO 122-IG002 btc 15.10 vs his 14.5; NGC 2979 has no magnitude in
HyperLEDA). One blocker (an SDSSJ object) did not exist in 1999. The
reproducible core is therefore ~40-43 of 57; the rest is not recoverable
from the paper plus present-day catalogues.

## Revision after three independent audits (M/L method, selection, whole paper)

Corrections adopted:
- **Isolation units are OPEN, not settled.** "Keeps his 57" cannot
  discriminate raw from L^(1/3)-scaled thresholds: all his pairs have
  L10 >= 1.3, so scaled thresholds are uniformly stricter and any looser rule
  scores higher. Non-monotone diagnostics split (scaled: 78 pairs, blind 43%,
  38 group drops, 18/57; raw: 328, 23%, 171, 38/57; Sample II ratio his 1.91,
  scaled 3.27, raw 1.56). His text ("the volume depends only on the total
  luminosity of a pair") supports scaled. The finder runs both (ISO arg).
- **Parent comparison made like-for-like**: 6,821 at B<=15.5 vs his 6,475
  (the earlier 6,137 was members at 15.0).
- **The "1999 catalogue depth" explanation for the extras is withdrawn**: 28%
  of our own 1999-epoch members lie beyond 3653 km/s, and H99 states the
  cause himself (bright pairs become scarce at large redshift -- a flux-limit
  effect).
- **Bound-truncated isotropic DF** adopted (scale-free rejection above
  escape): isotropic results move 10-16% toward his and all nine published
  values now sit inside the 2-D 68% intervals. The radial (beta=1) case is
  added: 49.2 vs his 42 +34-7.
- **His separation-dependence result (Fig. 7 / sec 4.1) is NOT reproduced**:
  our inner bin (R_p<100) gives 24.9 vs his 36, with a rising trend where he
  finds flat. Survives every variant; flagged as unresolved.
- Expected-log likelihood justified by the circular-orbit case (convolved
  form 12.0 vs his 28), not by median matching (which fails for circular).
- Eq. 19's (2 pi sigma)^(-1/2) read as a typo: sqrt(sigma) weighting
  overshoots (38.6, 46.8) with the bound DF.
- Companion-magnitude system: btc is the only textually supported choice;
  three of the five "irrecoverable" pairs are recoverable on raw bt, two are
  violations under either system.
- Fidelity nits: (1+z) factor removed (V_p validation now 1.001), M_B,sun =
  5.48 in both scripts, e_v/e_bt member cut shown to be inert (his 7% cuts
  are pair-level and unreproducible), "|b|" flagged as our reading of his
  "b", the strict either/or reading of criteria 5-6 scored (0/57).
- Cross-checks now used: 7 pairs above the (M/L)_p = 20 envelope (his ~7),
  sample III mean raw separation 205.8 kpc (his ~206), spiral fraction 70.2%
  (his 70%), subgroup counts 27/12/18 exact.
