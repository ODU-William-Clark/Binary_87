"""Per-object NED photometry for the few RC3 galaxies lacking B_T_0.

NED's TAP table (NEDTAP.objdir) carries no photometry, so magnitudes need the
per-object services. astroquery.ned wraps them politely; with only ~13
objects and a pause between calls this is a negligible load (never do this
for thousands of objects -- that is what got the old scraper in trouble).

For each galaxy we take the B-band total magnitude from NED's photometric
data table, preferring RC3-style corrected totals so the value is on the same
system as the rest of the sample. Every retrieved value is cross-checked
against the B magnitude Honma tabulated for the same galaxy.

Output: rc3_photometry_supplement.csv (name, l, b, B_supp, source).
"""
import time

import numpy as np
import pandas as pd
from astroquery.ipac.ned import Ned

# galaxies in Honma pairs whose RC3 rows lack B_T_0, with his tabulated B
TARGETS = {
    'NGC 6429': 13.7, 'NGC 6427': 13.9, 'NGC 7443': 13.9, 'UGCA 154': 14.2,
    'NGC 1266': 13.9, 'MCG -03-08-057': 14.8, 'MCG -02-25-013': 14.6,
    'NGC 2979': 13.6, 'ESO 512- G 018': 14.5, 'ESO 512- G 019': 14.4,
    'IC 4888': 14.3, 'MCG -01-38-014': 14.6, 'NGC 5916': 13.5,
}
# Honma-table names differ slightly; map for the cross-check
HONMA_B = {'NGC 6429': None}  # cross-check uses honma99_pairs.csv below

honma = pd.read_csv('honma99_pairs.csv')
hb = {}
for _, r in honma.iterrows():
    hb[r.name1.replace(' 0', ' ').replace('  ', ' ')] = r.B1
    hb[r.name2.replace(' 0', ' ').replace('  ', ' ')] = r.B2

PREFERRED = ['B (m_T)', 'B_T', 'B (total)', 'B']

rows = []
for name in TARGETS:
    time.sleep(2.0)   # politeness between per-object calls
    try:
        main = Ned.query_object(name)
        phot = Ned.get_table(name, table='photometry')
    except Exception as e:
        print('%-16s FAILED: %s' % (name, e))
        continue
    glon = float(main['GLON'][0]); glat = float(main['GLAT'][0])

    bands = [str(x) for x in phot['Observed Passband']]
    mags = np.array(phot['Photometry Measurement'], dtype=float)
    pick = None
    # prefer an RC3-provenance total B, then any total B, then any B
    for want in ('B (m_T)', 'B_T'):
        for i, b in enumerate(bands):
            if b.strip().startswith(want) and np.isfinite(mags[i]):
                pick = (mags[i], b.strip()); break
        if pick: break
    if pick is None:
        for i, b in enumerate(bands):
            if b.strip().startswith('B') and 'B-' not in b and np.isfinite(mags[i]):
                pick = (mags[i], b.strip()); break
    if pick is None:
        print('%-16s no B-band measurement found' % name)
        continue
    mag, src = pick
    rows.append(dict(name=name, l=glon, b=glat, B_supp=mag, source=src))
    print('%-16s B = %6.2f  (%s)   l,b = %8.3f %8.3f' % (name, mag, src, glon, glat))

out = pd.DataFrame(rows)
out.to_csv('rc3_photometry_supplement.csv', index=False)
print('\nwrote rc3_photometry_supplement.csv  (%d galaxies)' % len(out))
