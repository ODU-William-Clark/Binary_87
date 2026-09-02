"""Honma (1999) sample reconstruction on a HyperLEDA-based catalogue.  v2.2

Catalogue: every HyperLEDA galaxy with btc <= 15.8 (102,800 objects,
`leda_btc15.8_allsky.csv`, pulled in 12 longitude slices). btc is the
corrected total B on the RC3 B_T^0 system -- the magnitude Honma states he
used ("corrections for intrinsic absorption and galactic extinction were made
according to RC3"). Because the companion pool now carries magnitudes, the
m3 <= m_pair + 2 blocking threshold can be applied faithfully, which the
RC3-only pool could not do at NED depth.

Epoch modes (EPOCH):
  2026 : velocity = HyperLEDA v for every galaxy that has one.
  1999 : a galaxy counts as redshift-KNOWN only if it positionally matches an
         RC3 row with a native velocity, or a NED galaxy whose preferred
         redshift cites a <= 1999 bibcode. Others become redshift-unknown:
         they cannot be pair members and block only via criterion 5. This is a
         LOWER bound on the 1999 catalogue (NED supersedes bibcodes).

Selection (Honma sec. 2.1), isolation in raw units (see README):
  members  btc <= 15.0, |b| > 20, 1000 < V_LG < 4500
  pair     m_{1+2} <= 13.5,  R_p <= 400 kpc,  V_p <= 400 km/s  (L10-corrected)
  isolation: no companion with btc <= m_{1+2}+2 that is
             known-z, in-shell, r3 <= a*400 kpc AND |dv| <= b*400 km/s, or
             redshift-unknown with r3 <= a*400 kpc
  data quality (his two ~7% cuts): both members need a reported velocity
             error and a reported magnitude error (toggle)
  groups   any galaxy in >= 2 surviving pairs marks a group; drop its pairs
"""
import sys

import numpy as np
import pandas as pd

EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
MAGCOL = sys.argv[2] if len(sys.argv) > 2 else 'btc'   # member magnitudes: 'btc' (corrected) or 'bt' (raw)
FRAME = sys.argv[3] if len(sys.argv) > 3 else 'helio'    # 'helio' (Honma: heliocentric, no LG correction) or 'lg'
COMPMAG = sys.argv[4] if len(sys.argv) > 4 else 'bt'    # companion-pool magnitudes for the m_pair+2 threshold
A_ISO, B_ISO = 2.5, 1.5
H0, C_KMS = 50.0, 299792.458
M_SUN_B = 5.44
REQUIRE_ERRORS = True
OUT = 'leda_pairs_epoch%d_%s_%s_comp%s.csv' % (EPOCH, MAGCOL, FRAME, COMPMAG)

cat = pd.read_csv('leda_btc15.8_allsky.csv')
cat = cat.dropna(subset=['l2', 'b2', 'btc']).reset_index(drop=True)
if MAGCOL == 'bt':
    cat['btc'] = cat['bt']   # use raw total B everywhere (sensitivity test)

# ---------------------------------------------------------------- epoch ----
def positional_flag(src_l, src_b, tol=0.05):   # 0.05: ZCAT's B1950 positions are coarse (NGC 1266 sits 0.036 deg off)
    """Boolean per catalogue row: has a counterpart within tol deg in (src_l, src_b)."""
    order = np.argsort(src_b); sb = src_b[order]
    flag = np.zeros(len(cat), bool)
    cl, cb = cat.l2.values, cat.b2.values
    for i in range(len(cat)):
        j0, j1 = np.searchsorted(sb, cb[i] - tol), np.searchsorted(sb, cb[i] + tol)
        if j1 <= j0:
            continue
        c = order[j0:j1]
        dl = (src_l[c] - cl[i] + 180) % 360 - 180
        d = np.hypot(dl * np.cos(np.radians(cb[i])), src_b[c] - cb[i])
        flag[i] = d.min() < tol
    return flag

vel = cat.v.values.copy()
if EPOCH <= 1999:
    rc3 = pd.read_fwf('RC3_data_7_5_1.txt', colspecs=[(123, 130), (131, 138), (224, 230)],
                      names=['l', 'b', 'V'])
    for c in rc3.columns:
        rc3[c] = pd.to_numeric(rc3[c], errors='coerce')
    rc3 = rc3.dropna()
    ned = pd.read_csv('ned_shell_galaxies.csv').dropna(subset=['z', 'gallon', 'gallat'])
    yr = pd.to_numeric(ned.z_bibcode.astype(str).str[:4], errors='coerce')
    ned = ned[yr <= EPOCH]
    # CfA ZCAT June 1995 (VizieR VII/193): a genuine pre-1999 redshift
    # compilation, 57,536 galaxies. This is the main "known by 1999" source;
    # the NED <=1999-bibcode subset badly undercounts because NED supersedes
    # preferred redshifts with later remeasurements.
    zc = pd.read_csv('zcat95_galactic.csv').dropna(subset=['l', 'b', 'Vh'])
    known = (positional_flag(rc3.l.values, rc3.b.values)
             | positional_flag(ned.gallon.values, ned.gallat.values)
             | positional_flag(zc.l.values, zc.b.values))
    vel[~known] = np.nan
    print('epoch %d: %d of %d catalogue galaxies keep a velocity'
          % (EPOCH, np.isfinite(vel).sum(), len(cat)))
else:
    print('epoch 2026: %d of %d catalogue galaxies have a velocity'
          % (np.isfinite(vel).sum(), len(cat)))

# Honma's Table 1 velocities match RC3 HELIOCENTRIC values to MAD 13 km/s
# (vs 133 km/s for LG-corrected): he applied no Local Group correction.
V_SUN = 0.0 if FRAME == 'helio' else 308.0
L_A, B_A = np.radians(105.0), np.radians(-7.0)
lr, br = np.radians(cat.l2.values), np.radians(cat.b2.values)
vlg = vel + V_SUN * (np.sin(br) * np.sin(B_A) + np.cos(br) * np.cos(B_A) * np.cos(lr - L_A))
cat['V_lg'] = vlg

# ---------------------------------------------------------------- parent ---
member = ((cat.btc <= 15.0) & (cat.b2.abs() > 20.0)
          & (cat.V_lg > 1000.0) & (cat.V_lg < 4500.0))
if REQUIRE_ERRORS:
    member &= cat.e_v.notna() & cat.e_bt.notna()
P = cat[member].reset_index(drop=True)
P.to_csv('leda_parent_epoch%d.csv' % EPOCH, index=False)
print('parent (members): %d   (Honma: 6475 at m<=15.5 incl. non-members)' % len(P))

# companion pool: whole catalogue (btc <= 15.8 covers every m_pair+2 <= 15.5)
# Companion magnitudes: a 1999 redshift-blind NED search returned raw
# (uncorrected) magnitudes, and the isolation killers of his own pairs sit
# 0.01-0.2 mag above the btc threshold -- so raw bt is the faithful choice.
c_l, c_b = cat.l2.values, cat.b2.values
if COMPMAG == 'bt':
    c_m = np.where(np.isfinite(cat.bt.values), cat.bt.values, cat.btc.values)
elif COMPMAG == 'hybrid':
    # "NED, supplied with RC3": RC3 galaxies carried corrected B_T^0 in NED,
    # everything else a raw magnitude. Use btc where the galaxy is an RC3
    # object with B_T_0, raw bt otherwise.
    _rc3 = pd.read_fwf('RC3_data_7_5_1.txt', colspecs=[(123, 130), (131, 138), (394, 400)],
                       names=['l', 'b', 'B'])
    for _c in _rc3.columns:
        _rc3[_c] = pd.to_numeric(_rc3[_c], errors='coerce')
    _rc3 = _rc3.dropna()
    _is_rc3 = positional_flag(_rc3.l.values, _rc3.b.values, tol=0.03)
    c_m = np.where(_is_rc3, cat.btc.values,
                   np.where(np.isfinite(cat.bt.values), cat.bt.values, cat.btc.values))
    print('hybrid companion magnitudes: %d RC3 galaxies use btc, %d others use raw bt'
          % (_is_rc3.sum(), (~_is_rc3).sum()))
else:
    c_m = cat.btc.values
c_v = cat.V_lg.values
c_has = np.isfinite(c_v)
c_in = c_has & (c_v >= 1000.0) & (c_v <= 4500.0)
c_pgc = cat.pgc.values

p_l, p_b, p_m, p_v, p_pgc = P.l2.values, P.b2.values, P.btc.values, P.V_lg.values, P.pgc.values
p_f = 10 ** (-0.4 * p_m)
N = len(P)

def ang(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.arccos(np.clip(np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1))

cand = []
n_primary = n_iso = n_iso_known = n_iso_blind = 0
for i in range(N - 1):
    j = np.arange(i + 1, N)
    fs = p_f[i] + p_f[j]
    m_tot = -2.5 * np.log10(fs)
    ok = m_tot <= 13.5
    j, fs, m_tot = j[ok], fs[ok], m_tot[ok]
    if not len(j):
        continue
    vbar = (p_v[i] * p_f[i] + p_v[j] * p_f[j]) / fs
    mu = 5 * np.log10(vbar / H0) + 25.0
    L10 = (10 ** (-0.4 * (p_m[i] - mu - M_SUN_B)) + 10 ** (-0.4 * (p_m[j] - mu - M_SUN_B))) / 1e10
    Lc = L10 ** (1 / 3.)
    vp = np.abs(p_v[i] - p_v[j]) / (1 + vbar / C_KMS)
    V = vp / Lc
    ok = V <= 400.0
    j, fs, m_tot, vbar, L10, Lc, vp, V = (x[ok] for x in (j, fs, m_tot, vbar, L10, Lc, vp, V))
    if not len(j):
        continue
    th = ang(p_l[i], p_b[i], p_l[j], p_b[j])
    rp = 2 * (vbar / H0) * 1000 * np.tan(th / 2)
    R = rp / Lc
    ok = R <= 400.0
    j, fs, m_tot, vbar, L10, Lc, vp, V, rp, R, th = (x[ok] for x in
        (j, fs, m_tot, vbar, L10, Lc, vp, V, rp, R, th))

    for jj, f_s, mt, vb, l10, vpp, VV, rpp, RR, tt in zip(j, fs, m_tot, vbar, L10, vp, V, rp, R, th):
        n_primary += 1
        dl = (p_l[jj] - p_l[i] + 180) % 360 - 180
        lc = p_l[i] + dl * p_f[jj] / f_s
        bc = (p_b[i] * p_f[i] + p_b[jj] * p_f[jj]) / f_s
        keep = (c_m <= mt + 2.0) & (c_pgc != p_pgc[i]) & (c_pgc != p_pgc[jj])
        th3 = ang(lc, bc, c_l[keep], c_b[keep])
        r3 = 2 * (vb / H0) * 1000 * np.tan(th3 / 2)
        v3 = np.abs(c_v[keep] - vb) / (1 + vb / C_KMS)
        bad_known = c_in[keep] & (r3 <= A_ISO * 400) & (v3 <= B_ISO * 400)
        bad_blind = (~c_has[keep]) & (r3 <= A_ISO * 400)
        if bad_known.any():
            n_iso += 1; n_iso_known += 1
            continue
        if bad_blind.any():
            n_iso += 1; n_iso_blind += 1
            continue
        cand.append(dict(Name_1=P.objname[i], Name_2=P.objname[jj], pgc1=p_pgc[i], pgc2=p_pgc[jj],
                         l1=p_l[i], b1=p_b[i], l2=p_l[jj], b2=p_b[jj],
                         B_1=p_m[i], B_2=p_m[jj], V_lg_1=p_v[i], V_lg_2=p_v[jj],
                         L10=l10, v_bar=vb, v_p=vpp, r_p=rpp, R=RR, V=VV, m_pair=mt))

pairs = pd.DataFrame(cand)
pairs.to_csv(OUT.replace('.csv', '_pregroup.csv'), index=False)
print('primary candidates (pass mag/V/R): %d   rejected by isolation: %d   kept: %d'
      % (n_primary, n_iso, len(pairs)))
print('   killed by KNOWN-z in-shell companion: %d   survivors of that step: %d'
      % (n_iso_known, n_primary - n_iso_known))
print('   then killed by BLIND (z-unknown) companion: %d  (= %.0f%% of the %d that reached it;'
      ' Honma: ~30%%)' % (n_iso_blind, 100.0 * n_iso_blind / max(n_primary - n_iso_known, 1),
                          n_primary - n_iso_known))
if len(pairs):
    counts = pd.concat([pairs.pgc1, pairs.pgc2]).value_counts()
    grouped = set(counts[counts >= 2].index)
    ing = pairs.pgc1.isin(grouped) | pairs.pgc2.isin(grouped)
    print('group rejection: %d pairs dropped (%d galaxies in >=2 pairs)' % (ing.sum(), len(grouped)))
    pairs = pairs[~ing].reset_index(drop=True)
pairs.to_csv(OUT, index=False)
print('FINAL: %d pairs -> %s   (Honma sample I: 57)   [epoch %d, members %s, frame %s, companions %s]' % (len(pairs), OUT, EPOCH, MAGCOL, FRAME, COMPMAG))
