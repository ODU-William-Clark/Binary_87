"""Is Honma's pairing rule mutual-nearest-neighbour?

For every member of the 1999-epoch parent, find its nearest other member in
projected separation among members within |dv| < 600 km/s. A candidate pair
is 'mutual' if each member is the other's nearest neighbour. Score: fraction
of Honma-matched pairs that are mutual vs fraction of extras that are.
"""
import numpy as np, pandas as pd
H0 = 50.0
P = pd.read_csv('leda_parent_epoch1999.csv')
pairs = pd.read_csv('leda_pairs_epoch1999_pregroup.csv')
raw = pd.read_fwf('honma_table1_full.dat', colspecs=[(0,16),(36,45),(46,55)], names=['Name','GLON','GLAT'])
hc = np.array([(r.GLON, r.GLAT) for r in raw.itertuples()])
def inh(l0,b0):
    dl=(hc[:,0]-l0+180)%360-180
    return (np.hypot(dl*np.cos(np.radians(b0)), hc[:,1]-b0) < 0.03).any()

l, b, v, pgc = P.l2.values, P.b2.values, P.V_lg.values, P.pgc.values
lr, br = np.radians(l), np.radians(b)
nn = {}
for i in range(len(P)):
    cos = np.sin(br[i])*np.sin(br) + np.cos(br[i])*np.cos(br)*np.cos(lr[i]-lr)
    th = np.arccos(np.clip(cos, -1, 1))
    rp = 2*(v[i]/H0)*1000*np.tan(th/2)
    ok = (np.abs(v - v[i]) < 600.0)
    ok[i] = False
    rp[~ok] = np.inf
    j = rp.argmin()
    nn[pgc[i]] = pgc[j] if np.isfinite(rp[j]) else None

pairs['match'] = [inh(r.l1,r.b1) and inh(r.l2,r.b2) for r in pairs.itertuples()]
pairs['mutual'] = [(nn.get(r.pgc1) == r.pgc2) and (nn.get(r.pgc2) == r.pgc1) for r in pairs.itertuples()]
m, e = pairs[pairs.match], pairs[~pairs.match]
print('pre-group candidates: %d   (Honma-matched %d, extra %d)' % (len(pairs), len(m), len(e)))
print('mutual nearest neighbours:  matched %d/%d = %.0f%%     extras %d/%d = %.0f%%'
      % (m.mutual.sum(), len(m), 100*m.mutual.mean(), e.mutual.sum(), len(e), 100*e.mutual.mean()))
keep = pairs[pairs.mutual]
print('if MNN were required: %d pairs kept, of which %d are his' % (len(keep), keep.match.sum()))
# and with the group rule on top
counts = pd.concat([keep.pgc1, keep.pgc2]).value_counts()
g = set(counts[counts>=2].index)
final = keep[~(keep.pgc1.isin(g) | keep.pgc2.isin(g))]
print('  + group rejection: %d pairs, %d his   (Honma: 57)' % (len(final), final.match.sum()))
