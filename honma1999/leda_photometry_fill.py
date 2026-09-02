"""B_T^0-equivalent magnitudes from HyperLEDA for RC3 rows lacking B_T_0.

NED's TAP has no photometry and its legacy per-object CGI is broken ("EGRET
error"), so HyperLEDA is the source: its `btc` is the corrected total B on the
same system as RC3's B_T^0. Queried positionally (one small box per galaxy,
2 s apart) via the fG.cgi interface.
"""
import io, time, urllib.request, urllib.parse
import numpy as np, pandas as pd

BASE = 'http://atlas.obs-hp.fr/hyperleda/fG.cgi'

TARGETS = ['NGC 6429','NGC 6427','NGC 7443','UGCA 154','NGC 1266',
           'MCG-03-08-057','MCG-02-25-013','NGC 2979','ESO 512-G018',
           'ESO 512-G019','IC 4888','MCG-01-38-014','NGC 5916']

raw = pd.read_fwf('honma_table1_full.dat', colspecs=[(0,16),(36,45),(46,55),(70,74)],
                  names=['Name','GLON','GLAT','Bmag'])
raw['Name'] = raw.Name.str.strip()
coord = {r.Name: (r.GLON, r.GLAT, r.Bmag) for r in raw.itertuples()}

def leda_box(l0, b0, half=0.04):
    sql = "l2>%.4f and l2<%.4f and b2>%.4f and b2<%.4f" % (l0-half/np.cos(np.radians(b0)),
          l0+half/np.cos(np.radians(b0)), b0-half, b0+half)
    q = urllib.parse.urlencode({'n':'meandata','c':'o','of':'1,leda','nra':'l',
        'nakd':'1','d':'objname,bt,e_bt,btc,vopt,l2,b2','sql':sql,'ob':'','a':'csv'})
    with urllib.request.urlopen(BASE+'?'+q, timeout=90) as r:
        txt = r.read().decode('utf-8','replace')
    lines = [ln for ln in txt.splitlines() if ln and not ln.startswith('#')]
    if len(lines) < 2: return None
    # header row is space-separated, data rows are tab-separated
    cols = lines[0].split()
    return pd.read_csv(io.StringIO('\n'.join(lines[1:])), sep='\t',
                       names=cols, engine='python')

rows = []
for name in TARGETS:
    l0, b0, hB = coord[name]
    time.sleep(2)
    t = leda_box(l0, b0)
    if t is None or not len(t):
        print('%-16s NOTHING in box' % name); continue
    for c in ['bt','btc','b2','l2']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t['d'] = np.hypot((t.l2-l0)*np.cos(np.radians(b0)), t.b2-b0)
    t = t.sort_values('d')
    g = t.iloc[0]
    B = g.btc if np.isfinite(g.btc) else g.bt
    src = 'btc' if np.isfinite(g.btc) else 'bt'
    rows.append(dict(name=name, leda_name=str(g.objname).strip(), l=l0, b=b0,
                     B_supp=B, source='leda_'+src, honma_B=hB))
    print('%-16s -> %-14s B=%6.2f (%s)   Honma B=%s   sep=%.4f deg'
          % (name, str(g.objname).strip(), B, src, hB, g.d))

out = pd.DataFrame(rows)
out.to_csv('rc3_photometry_supplement.csv', index=False)
dif = out.B_supp - pd.to_numeric(out.honma_B)
print('\nvs Honma: median offset %+.2f mag, scatter %.2f' % (dif.median(), dif.std()))
print('wrote rc3_photometry_supplement.csv (%d)' % len(out))
