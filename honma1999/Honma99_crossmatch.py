"""Cross-match the RC3 pair finder output against Honma (1999) Table 1.

Matching is positional: a finder pair matches a Honma pair when both members
lie within TOL degrees of the Honma members' (GLON, GLAT), in either order.
Name matching is unreliable across catalogues (RC3 pads names, Honma
zero-pads UGC numbers, ESO formats differ), so names are only reported.
"""
import sys

import numpy as np
import pandas as pd

TOL = 0.03  # deg ~ 2 arcmin

finder_csv = sys.argv[1] if len(sys.argv) > 1 else 'binary_candidates_isolated_a2.5.csv'
mine = pd.read_csv(finder_csv)
his = pd.read_csv('honma99_pairs.csv')

# Honma per-member coordinates come from the raw table
raw = pd.read_fwf('honma_table1_full.dat',
                  colspecs=[(0, 16), (18, 34), (36, 45), (46, 55)],
                  names=['Name', 'Pair', 'GLON', 'GLAT'])
raw['Name'] = raw.Name.str.strip()
coord = {r.Name: (r.GLON, r.GLAT) for r in raw.itertuples()}

def close(l1, b1, l2, b2):
    dl = (l1 - l2 + 180) % 360 - 180
    return np.hypot(dl * np.cos(np.radians(b1)), b1 - b2) < TOL

matched, missed = [], []
used = set()
for _, h in his.iterrows():
    (lA, bA), (lB, bB) = coord[h.name1], coord[h.name2]
    hit = None
    for j, m in mine.iterrows():
        if ((close(m.l1, m.b1, lA, bA) and close(m.l2, m.b2, lB, bB)) or
                (close(m.l1, m.b1, lB, bB) and close(m.l2, m.b2, lA, bA))):
            hit = j
            break
    if hit is not None:
        used.add(hit)
        m = mine.loc[hit]
        rp_col = 'r_p' if 'r_p' in mine.columns else 'rp_kpc'
        matched.append((h.name1, h.name2, str(m.Name_1).strip(), str(m.Name_2).strip(),
                        h.Rp, m.R, h.Vp, m.V))
    else:
        missed.append((h.name1, h.name2, h.Rp, h.Vp))

print('Finder pairs: %d   Honma pairs: 57' % len(mine))
print('MATCHED: %d   MISSED (in Honma, not found by finder): %d   '
      'EXTRA (found, not in Honma): %d' % (len(matched), len(missed),
                                           len(mine) - len(used)))
print()
if matched:
    print('%-34s %-30s %7s %7s %7s %7s' %
          ('Honma pair', 'finder pair', 'Rp_H', 'R_mine', 'Vp_H', 'V_mine'))
    for a, b, c, d, rh, rm, vh, vm in matched:
        print('%-34s %-30s %7.1f %7.1f %7.1f %7.1f' %
              (a + ' + ' + b, c + ' + ' + d, rh, rm, vh, vm))
print()
print('Honma pairs the finder did NOT recover:')
for a, b, rp, vp in missed:
    print('   %-18s %-18s  Rp=%6.1f  Vp=%6.1f' % (a, b, rp, vp))
print()
print('Finder pairs NOT in Honma:')
for j, m in mine.iterrows():
    if j not in used:
        print('   %-18s %-18s  R=%6.1f  V=%6.1f' %
              (str(m.Name_1).strip(), str(m.Name_2).strip(), m.R, m.V))
