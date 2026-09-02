"""Pinpoint the reading of Honma's isolation criteria 5-6.

Honma's text (sec 2.1): a pair is kept if every companion brighter than
m_{1+2} + 2.0 satisfies BOTH
    5:  r_i / L10^{1/3}  >= a * 400 kpc      (a = 2.5 for sample I)
    6:  |v_i| / L10^{1/3} >= b * 400 km/s    (b = 1.5)
i.e. a companion is harmful only if it is close in projection AND close in
velocity (redshift-unknown companions: close in projection alone).

A faithful implementation of that literal reading rejects pairs that are in
his own Table 1 (e.g. NGC 7537/7541, killed by NGC 7562 at R3 = 860 < 1000
after the L^{1/3} = 2.35 correction). So some part of the reading is wrong.

Decisive test: HIS 57 pairs all survived HIS criterion. Each candidate
reading is scored by how many of the 57 it keeps, using a companion pool as
close to his as we can get (RC3 magnitudes, complete to ~15.5, plus NED
velocities restricted to <=1999 bibcodes; also scored with the full 2026
velocities to show the epoch effect the companion pool has).

Variants:
    lit    : literal      -- r/L^{1/3} vs a*400,  |dv|/L^{1/3} vs b*400
    rawv   : raw velocity -- r/L^{1/3} vs a*400,  |dv|        vs b*400
    rawrv  : all raw      -- r          vs a*400,  |dv|        vs b*400
    comb   : the 2.0 finder's combined-luminosity normalisation (baseline)
"""
import numpy as np
import pandas as pd

H0, C_KMS = 50.0, 3e5
A_ISO, B_ISO = 2.5, 1.5

# ---------------------------------------------------------------- pool ----
colspecs = [(21, 36), (123, 130), (131, 138), (224, 230), (394, 400)]
rc3 = pd.read_fwf('RC3_data_7_5_1.txt', colspecs=colspecs,
                  names=['Name', 'l', 'b', 'V_opt', 'B'])
for c in ['l', 'b', 'V_opt', 'B']:
    rc3[c] = pd.to_numeric(rc3[c], errors='coerce')
supp = pd.read_csv('rc3_photometry_supplement.csv').dropna(subset=['B_supp'])
for _, srow in supp.iterrows():
    dl = (rc3.l - srow.l + 180) % 360 - 180
    d = np.hypot(dl * np.cos(np.radians(srow.b)), rc3.b - srow.b)
    i = d.idxmin()
    if d[i] < 0.03 and pd.isna(rc3.at[i, 'B']):
        rc3.at[i, 'B'] = srow.B_supp

ned = pd.read_csv('ned_shell_galaxies.csv').dropna(subset=['z', 'gallon', 'gallat'])
yr = pd.to_numeric(ned.z_bibcode.astype(str).str[:4], errors='coerce')

def make_pool(max_year):
    """RC3 galaxies with magnitudes; velocities from RC3 + epoch-limited NED."""
    p = rc3[rc3.B.notna()].copy().reset_index(drop=True)
    sel = ned if max_year is None else ned[yr <= max_year]
    nv = sel.z.values * 299792.458
    nl, nb = sel.gallon.values, sel.gallat.values
    order = np.argsort(nb); nbs = nb[order]
    filled = 0
    vv = p.V_opt.values.copy()
    for i in np.flatnonzero(p.V_opt.isna().values):
        l0, b0 = p.l.values[i], p.b.values[i]
        if np.isnan(l0) or np.isnan(b0):
            continue
        j0, j1 = np.searchsorted(nbs, b0 - .02), np.searchsorted(nbs, b0 + .02)
        c = order[j0:j1]
        if not len(c):
            continue
        dl = (nl[c] - l0 + 180) % 360 - 180
        d = np.hypot(dl * np.cos(np.radians(b0)), nb[c] - b0)
        k = d.argmin()
        if d[k] < .02:
            vv[i] = nv[c[k]]; filled += 1
    p['V_use'] = vv
    # Honma's velocities are heliocentric; the pair velocities scored here come
    # from his Table 1, so the companion pool must NOT be LG-corrected either.
    V_SUN, L_A, B_A = 0.0, np.radians(105.), np.radians(-7.)
    br, lr = np.radians(p.b), np.radians(p.l)
    p['V_lg'] = p.V_use + V_SUN * (np.sin(br) * np.sin(B_A)
                                   + np.cos(br) * np.cos(B_A) * np.cos(lr - L_A))
    return p, filled

# ---------------------------------------------------------------- pairs ---
his = pd.read_csv('honma99_pairs.csv')
raw = pd.read_fwf('honma_table1_full.dat', colspecs=[(0, 16), (36, 45), (46, 55)],
                  names=['Name', 'GLON', 'GLAT'])
raw['Name'] = raw.Name.str.strip()
coord = {r.Name: (r.GLON, r.GLAT) for r in raw.itertuples()}

def ang(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.arccos(np.clip(np.sin(b1) * np.sin(b2)
                             + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1))

def survives(h, pool, variant):
    """Does Honma pair h survive isolation under `variant` with this pool?"""
    (lA, bA), (lB, bB) = coord[h.name1], coord[h.name2]
    fA, fB = 10**(-.4 * h.B1), 10**(-.4 * h.B2)
    m_tot = -2.5 * np.log10(fA + fB)
    # luminosity-weighted centre (pairs are close; wrap via member A frame)
    dl = (lB - lA + 180) % 360 - 180
    lc = lA + dl * fB / (fA + fB); bc = (bA * fA + bB * fB) / (fA + fB)
    vbar = h.vbar
    Lc = h.L10 ** (1 / 3.)

    pl, pb = pool.l.values, pool.b.values
    pm, pv = pool.B.values, pool.V_lg.values
    keep = pm <= m_tot + 2.0
    # exclude the members themselves positionally
    for (l0, b0) in ((lA, bA), (lB, bB)):
        dl0 = (pl - l0 + 180) % 360 - 180
        keep &= np.hypot(dl0 * np.cos(np.radians(b0)), pb - b0) > 0.02
    th = ang(lc, bc, pl[keep], pb[keep])
    r3 = 2 * (vbar / H0) * 1000 * np.tan(th / 2)          # kpc, raw
    v3 = np.abs(pv[keep] - vbar) / (1 + vbar / C_KMS)     # km/s, raw
    # match the 2.0 finder: known-z companions OUTSIDE the survey shell are
    # skipped entirely (they are not sample members); only in-shell known-z
    # and redshift-unknown companions can harm
    hasv = np.isfinite(pv[keep])
    inshell = hasv & (pv[keep] >= 1000.0) & (pv[keep] <= 4500.0)

    if variant == 'lit':
        Rbad = r3 / Lc < A_ISO * 400
        Vbad = v3 / Lc < B_ISO * 400
    elif variant == 'rawv':
        Rbad = r3 / Lc < A_ISO * 400
        Vbad = v3 < B_ISO * 400
    elif variant == 'rawrv':
        Rbad = r3 < A_ISO * 400
        Vbad = v3 < B_ISO * 400
    elif variant == 'rawrv_oos':  # out-of-shell known-z treated as blind (R-only)
        Rbad = r3 < A_ISO * 400
        Vbad = np.where(inshell, v3 < B_ISO * 400, True)
    elif variant == 'either':    # printed text taken strictly: harmful if close in R OR in V (keeps 0 of 57)
        Rbad = r3 < A_ISO * 400
        Vbad = v3 < B_ISO * 400
    elif variant == 'rawR':      # projection alone, velocity ignored
        Rbad = r3 < A_ISO * 400
        Vbad = np.ones_like(r3, bool)
    elif variant == 'rawrv1.5':  # a = 1.5 (his sample II volume)
        Rbad = r3 < 1.5 * 400
        Vbad = v3 < B_ISO * 400
    elif variant == 'comb':
        f3 = 10**(-.4 * pm[keep])
        fs = fA + fB
        vbar3 = (vbar * fs + pv[keep] * f3) / (fs + f3)
        mu3 = 5 * np.log10(np.clip(vbar3 / H0, 1e-6, None)) + 25
        L3 = (10**(-.4 * (h.B1 - mu3 - 5.44)) + 10**(-.4 * (h.B2 - mu3 - 5.44))
              + 10**(-.4 * (pm[keep] - mu3 - 5.44))) / 1e10
        Lc3 = L3 ** (1 / 3.)
        r3c = 2 * (vbar3 / H0) * 1000 * np.tan(th / 2)
        v3c = np.abs(vbar3 - pv[keep]) / (1 + vbar3 / C_KMS)
        Rbad = np.where(hasv, r3c / Lc3 < A_ISO * 400, r3 / Lc < A_ISO * 400)
        Vbad = v3c / Lc3 < B_ISO * 400

    if variant == 'either':
        harmful = np.where(hasv, inshell & (Rbad | Vbad), Rbad)
    elif variant == 'rawrv_oos':
        harmful = np.where(hasv, Rbad & Vbad, Rbad)   # out-of-shell: Vbad forced True
    else:
        harmful = np.where(hasv, inshell & Rbad & Vbad, Rbad)
    if not harmful.any():
        return True, None
    idx = np.flatnonzero(harmful)[0]
    return False, str(pool.Name.values[keep][idx]).strip()

for epoch, label in [(1999, 'companion pool: NED z <= 1999'),
                     (None, 'companion pool: NED z 2026 (full)')]:
    pool, nf = make_pool(epoch)
    print('=== %s  (NED velocity fills into pool: %d) ===' % (label, nf))
    print('%-8s %10s' % ('variant', 'kept of 57'))
    for variant in ['lit', 'rawv', 'rawrv', 'rawR', 'rawrv1.5', 'comb', 'either']:
        kills = []
        for _, h in his.iterrows():
            ok, killer = survives(h, pool, variant)
            if not ok:
                kills.append((h.name1, h.name2, killer))
        print('%-8s %6d       killed: %s' % (variant, 57 - len(kills),
              '; '.join('%s+%s by %s' % k for k in kills[:4])
              + (' ...' if len(kills) > 4 else '')))
    print()
