"""Parse Honma (1999, ApJ 516, 693) Table 1 (VizieR J/ApJ/516/693) into a
per-pair CSV, and validate the parse by recomputing his luminosity-corrected
separation from raw coordinates, magnitudes and redshifts.

Table 1 has one row per GALAXY (114 rows = 57 pairs); the pair-level columns
LCSep (R_p), LCVel (V_p) and M/L are repeated on both rows. Unit convention
verified against the M/L column: (M/L)_p = R_p V_p^2 / (G * 1e10), i.e.
luminosities are normalised by L10 = (L1+L2)/1e10 Lsun and
R_p = r_p / L10^{1/3}, V_p = |dv| / L10^{1/3} (his eqs. 6-7).
"""
import numpy as np
import pandas as pd

H0 = 50.0        # km/s/Mpc, as in the paper
MB_SUN = 5.48    # B-band solar absolute magnitude (Bessell 1979 scale)
C_KMS = 299792.458

colspecs = [(0, 16), (18, 34), (36, 45), (46, 55), (56, 60), (61, 64),
            (65, 69), (70, 74), (75, 80), (82, 87), (88, 95)]
names = ['Name', 'Pair', 'GLON', 'GLAT', 'Vel', 'e_Vel', 'MType', 'Bmag',
         'LCSep', 'LCVel', 'MLp']

df = pd.read_fwf('honma_table1_full.dat', colspecs=colspecs, names=names)
for c in ['GLON', 'GLAT', 'Vel', 'e_Vel', 'Bmag', 'LCSep', 'LCVel', 'MLp']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['Name'] = df['Name'].str.strip()
df['Pair'] = df['Pair'].str.strip()

# --- pair up rows: each pair appears twice (A,B) and (B,A) -----------------
seen = set()
pairs = []
byname = {r.Name: r for r in df.itertuples()}
for r in df.itertuples():
    key = tuple(sorted([r.Name, r.Pair]))
    if key in seen:
        continue
    seen.add(key)
    p = byname.get(r.Pair)
    if p is None:
        raise ValueError('partner not found: %r' % r.Pair)
    pairs.append((r, p))
assert len(pairs) == 57, len(pairs)

def ang_sep(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.arccos(np.clip(np.sin(b1)*np.sin(b2) +
                             np.cos(b1)*np.cos(b2)*np.cos(l1-l2), -1, 1))

rows = []
for a, b in pairs:
    # apparent-flux weights for the luminosity-centre velocity
    fa, fb = 10**(-0.4*a.Bmag), 10**(-0.4*b.Bmag)
    vbar = (a.Vel*fa + b.Vel*fb) / (fa + fb)
    D_mpc = vbar / H0

    # B luminosities at the pair distance
    mu = 5*np.log10(D_mpc) + 25.0
    L1 = 10**(-0.4*(a.Bmag - mu - MB_SUN))
    L2 = 10**(-0.4*(b.Bmag - mu - MB_SUN))
    L10 = (L1 + L2) / 1e10

    theta = ang_sep(a.GLON, a.GLAT, b.GLON, b.GLAT)
    rp_kpc = 2.0 * D_mpc * 1000.0 * np.tan(theta/2)
    dv = abs(a.Vel - b.Vel)   # Honma applies no (1+z) factor; with it V_p validates at 0.993, without at 1.001

    Rp_check = rp_kpc / L10**(1/3.)
    Vp_check = dv / L10**(1/3.)

    sigma_dv = np.hypot(a.e_Vel, b.e_Vel)          # km/s, raw
    sigma_Vp = sigma_dv / L10**(1/3.)              # luminosity-corrected

    rows.append(dict(
        name1=a.Name, name2=b.Name, type1=a.MType, type2=b.MType,
        v1=a.Vel, v2=b.Vel, e_v1=a.e_Vel, e_v2=b.e_Vel,
        B1=a.Bmag, B2=b.Bmag, L10=L10, vbar=vbar,
        Rp=a.LCSep, Vp=a.LCVel, MLp=a.MLp,
        Rp_recomputed=Rp_check, Vp_recomputed=Vp_check,
        sigma_Vp=sigma_Vp))

out = pd.DataFrame(rows)
out.to_csv('honma99_pairs.csv', index=False)

r = out.Rp_recomputed / out.Rp
v = out.Vp_recomputed / out.Vp
print('57 pairs written to honma99_pairs.csv')
print('validation vs his table values:')
print('  Rp ratio: median %.3f  (16-84%%: %.3f-%.3f)'
      % (r.median(), r.quantile(.16), r.quantile(.84)))
print('  Vp ratio: median %.3f  (16-84%%: %.3f-%.3f)'
      % (v.median(), v.quantile(.16), v.quantile(.84)))
print('  worst Rp mismatches:')
for _, x in out.assign(rr=abs(np.log(r))).nlargest(3, 'rr').iterrows():
    print('    %-16s %-16s  table %6.1f  ours %6.1f'
          % (x.name1, x.name2, x.Rp, x.Rp_recomputed))
