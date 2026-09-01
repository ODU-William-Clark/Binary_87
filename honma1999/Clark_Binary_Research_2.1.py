"""RC3 + NED binary-candidate finder, v2.1.

Changes vs Clark_Binary_Research_2.0:

1. B_T_0_err is no longer REQUIRED. RC3 tabulates it for only 4,014 of 23,010
   galaxies, and requiring it cut the parent sample to 1,357 against Honma's
   6,475 -- the single largest cause of missing his pairs. Missing errors are
   assigned 0.2 mag (Honma: "The uncertainty in the magnitude is typically
   0.2 mag").

2. Velocities missing from RC3 are filled from NED via its TAP service
   (`ned_tap_fetch.py` pulls every NED galaxy in the velocity shell in ONE
   asynchronous ADQL query; positional match on galactic coordinates). This
   mirrors Honma's parent construction: "data ... mainly taken from NED, and
   supplied with RC3".

4. Group rejection (Honma sec. 2.1): after candidate pairs are assembled, any
   galaxy appearing in two or more pairs marks a probable group; all pairs
   containing such a galaxy are dropped.

The isolation criterion itself is UNCHANGED from 2.0 (including its
combined-luminosity normalisation) -- revisiting that is a separate, deliberate
decision. The inner loops are vectorised; results are identical to 2.0's
logic, just ~100x faster.
"""
import os

import numpy as np
import pandas as pd

CONFIG = {
    "RC3_FWF_PATH": "RC3_data_7_5_1.txt",
    "NED_SHELL_CSV": "ned_shell_galaxies.csv",   # from ned_tap_fetch.py; optional
    "OUTPUT_CSV": "binary_candidates_v2.1.csv",
    "H0": 50.0,
    "BAND": "B",
    "MAG_COLUMN": "B_T_0",
    "M_SUN_MAP": {"B": 5.44, "V": 4.83, "K": 3.28},
    "DEFAULT_MAG_ERR": 0.2,
    "a": 2.5,
    "b": 1.5,
    "GAL_LAT_MIN_ABS": 20.0,
    "MAG_LIMIT": 15.0,
    "V_LG_MIN": 1000.0,
    "V_LG_MAX": 4500.0,
    "PAIR_MAG_SUM_LIMIT": 13.5,
    "NED_MATCH_TOL_DEG": 0.02,
}

H0 = CONFIG["H0"]
MAG = CONFIG["MAG_COLUMN"]
M_SUN = CONFIG["M_SUN_MAP"][CONFIG["BAND"]]
a_iso, b_iso = CONFIG["a"], CONFIG["b"]
C_KMS = 3e5

# ----------------------------------------------------------------------------
# RC3
# ----------------------------------------------------------------------------
colspecs = [(21, 36), (37, 47), (123, 130), (131, 138), (224, 230),
            (394, 400), (74, 77), (59, 64), (162, 167)]
names = ["Name", "Hubble_type", "l", "b", "V_opt", "B_T_0", "B_T_0_err",
         "PA", "logR25"]
df = pd.read_fwf(CONFIG["RC3_FWF_PATH"], colspecs=colspecs, names=names)
for col in ["l", "b", "V_opt", "B_T_0", "B_T_0_err", "PA", "logR25"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["ID"] = np.arange(len(df))

# --- (1) magnitude errors: default, not required ---------------------------
df["B_T_0_err"] = df["B_T_0_err"].fillna(CONFIG["DEFAULT_MAG_ERR"])

# --- (2) fill missing velocities from the NED shell ------------------------
n_filled = 0
if os.path.exists(CONFIG["NED_SHELL_CSV"]):
    ned = pd.read_csv(CONFIG["NED_SHELL_CSV"])
    ned = ned.dropna(subset=["z", "gallon", "gallat"])
    ned_v = ned.z.values * 299792.458
    ned_l = ned.gallon.values
    ned_b = ned.gallat.values
    # match RC3 rows lacking V_opt but having a magnitude
    need = df.index[df.V_opt.isna() & df[MAG].notna()]
    tol = CONFIG["NED_MATCH_TOL_DEG"]
    # sort NED by gallat for a cheap window search
    order = np.argsort(ned_b)
    nb_sorted = ned_b[order]
    for i in need:
        l0, b0 = df.at[i, "l"], df.at[i, "b"]
        if pd.isna(l0) or pd.isna(b0):
            continue
        j0 = np.searchsorted(nb_sorted, b0 - tol)
        j1 = np.searchsorted(nb_sorted, b0 + tol)
        cand = order[j0:j1]
        if len(cand) == 0:
            continue
        dl = (ned_l[cand] - l0 + 180) % 360 - 180
        d = np.hypot(dl * np.cos(np.radians(b0)), ned_b[cand] - b0)
        k = d.argmin()
        if d[k] < tol:
            df.at[i, "V_opt"] = ned_v[cand[k]]
            n_filled += 1
    print("NED velocity fill: %d RC3 galaxies gained a velocity" % n_filled)
else:
    print("NED shell file not found -- running RC3-only")

# ----------------------------------------------------------------------------
# Local Group correction and parent cuts
# ----------------------------------------------------------------------------
V_SUN, L_APEX, B_APEX = 308.0, np.radians(105.0), np.radians(-7.0)
lr, br = np.radians(df.l), np.radians(df.b)
df["V_lg"] = df.V_opt + V_SUN * (np.sin(br) * np.sin(B_APEX)
                                 + np.cos(br) * np.cos(B_APEX)
                                 * np.cos(lr - L_APEX))

cut = ((df.b.abs() > CONFIG["GAL_LAT_MIN_ABS"]) & (df[MAG] < CONFIG["MAG_LIMIT"])
       & (df.V_lg > CONFIG["V_LG_MIN"]) & (df.V_lg < CONFIG["V_LG_MAX"])
       & df.V_opt.notna() & df[MAG].notna())
P = df[cut].reset_index(drop=True)
print("parent sample: %d galaxies  (Honma: 6475 at m<=15.5)" % len(P))

# companion pool: everything in RC3 with a magnitude (as in 2.0)
comp = df[df[MAG].notna()].reset_index(drop=True)
c_l = comp.l.values; c_b = comp.b.values
c_m = comp[MAG].values; c_vlg = comp.V_lg.values; c_id = comp.ID.values
c_has_v = comp.V_opt.notna().values

# ----------------------------------------------------------------------------
# Pair search (vectorised over j for each i)
# ----------------------------------------------------------------------------
p_l = P.l.values; p_b = P.b.values; p_m = P[MAG].values
p_vlg = P.V_lg.values; p_id = P.ID.values
p_f = 10 ** (-0.4 * p_m)
N = len(P)

def ang_sep(l1, b1, l2, b2):
    """Great-circle separation, radians; vectorised."""
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.arccos(np.clip(np.sin(b1) * np.sin(b2)
                             + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2),
                             -1.0, 1.0))

cand = []
stats = dict(mag=0, V=0, R=0, iso=0)
for i in range(N - 1):
    j = np.arange(i + 1, N)
    fsum = p_f[i] + p_f[j]
    m_tot = -2.5 * np.log10(fsum)
    ok = m_tot <= CONFIG["PAIR_MAG_SUM_LIMIT"]
    stats["mag"] += (~ok).sum()
    j = j[ok]; fsum = fsum[ok]; m_tot = m_tot[ok]
    if len(j) == 0:
        continue

    vbar = (p_vlg[i] * p_f[i] + p_vlg[j] * p_f[j]) / fsum
    mu = 5 * np.log10(np.clip(vbar / H0, 1e-6, None)) + 25.0
    L1 = 10 ** (-0.4 * (p_m[i] - mu - M_SUN))
    L2 = 10 ** (-0.4 * (p_m[j] - mu - M_SUN))
    Ln = (L1 + L2) / 1e10

    vp = np.abs(p_vlg[i] - p_vlg[j]) / (1 + vbar / C_KMS)
    V = vp / Ln ** (1 / 3)
    ok = V <= 400.0
    stats["V"] += (~ok).sum()
    j, fsum, m_tot, vbar, Ln, vp, V = (x[ok] for x in
                                       (j, fsum, m_tot, vbar, Ln, vp, V))
    if len(j) == 0:
        continue

    th = ang_sep(p_l[i], p_b[i], p_l[j], p_b[j])
    rp = 2 * (vbar / H0) * 1000 * np.tan(th / 2)
    R = rp / Ln ** (1 / 3)
    ok = R <= 400.0
    stats["R"] += (~ok).sum()
    j, fsum, m_tot, vbar, Ln, vp, V, rp, R, th = (x[ok] for x in
        (j, fsum, m_tot, vbar, Ln, vp, V, rp, R, th))

    for jj, fs, mt, vb, ln, vpp, VV, rpp, RR, tt in zip(
            j, fsum, m_tot, vbar, Ln, vp, V, rp, R, th):
        # ---- isolation (same semantics as 2.0, vectorised) -----------------
        lc = (p_l[i] * p_f[i] + p_l[jj] * p_f[jj]) / fs
        bc = (p_b[i] * p_f[i] + p_b[jj] * p_f[jj]) / fs

        keep = (c_m <= mt + 2.0) & (c_id != p_id[i]) & (c_id != p_id[jj])
        th3 = ang_sep(lc, bc, c_l[keep], c_b[keep])
        m3 = c_m[keep]; vlg3 = c_vlg[keep]; hasv3 = c_has_v[keep]

        # redshift-unknown companions: reject on projected distance alone
        r3u = 2 * (vb / H0) * 1000 * np.tan(th3 / 2) / ln ** (1 / 3)
        bad_u = (~hasv3) & (r3u <= a_iso * 400.0)

        # known-z companions (2.0 semantics: combined-luminosity norm)
        kz = hasv3 & (vlg3 >= CONFIG["V_LG_MIN"])
        f3 = 10 ** (-0.4 * m3)
        vbar3 = (vb * fs + vlg3 * f3) / (fs + f3)
        mu3 = 5 * np.log10(np.clip(vbar3 / H0, 1e-6, None)) + 25.0
        Lp3 = (10 ** (-0.4 * (p_m[i] - mu3 - M_SUN))
               + 10 ** (-0.4 * (p_m[jj] - mu3 - M_SUN))
               + 10 ** (-0.4 * (m3 - mu3 - M_SUN))) / 1e10
        vp3 = np.abs(vbar3 - vlg3) / (1 + vbar3 / C_KMS)
        rp3 = 2 * (vbar3 / H0) * 1000 * np.tan(th3 / 2)
        bad_k = kz & (rp3 / Lp3 ** (1 / 3) <= a_iso * 400.0) \
                   & (vp3 / Lp3 ** (1 / 3) <= b_iso * 400.0)

        if np.any(bad_u | bad_k):
            stats["iso"] += 1
            continue

        cand.append(dict(
            Name_1=P.Name[i], Name_2=P.Name[jj],
            Type_1=P.Hubble_type[i], Type_2=P.Hubble_type[jj],
            ID_1=p_id[i], ID_2=p_id[jj],
            l1=p_l[i], b1=p_b[i], l2=p_l[jj], b2=p_b[jj],
            B_1=p_m[i], B_2=p_m[jj],
            B_err_1=P.B_T_0_err[i], B_err_2=P.B_T_0_err[jj],
            V_lg_1=p_vlg[i], V_lg_2=p_vlg[jj],
            theta_rad=tt, L_total_norm=ln, v_bar=vb,
            v_p=vpp, r_p=rpp, R=RR, V=VV,
            H0=H0, BAND=CONFIG["BAND"], M_sun_band=M_SUN))

pairs = pd.DataFrame(cand)
print("pairs passing cuts+isolation: %d" % len(pairs))

# --- (4) group rejection: a galaxy in >= 2 pairs marks a group --------------
if len(pairs):
    counts = pd.concat([pairs.ID_1, pairs.ID_2]).value_counts()
    grouped = set(counts[counts >= 2].index)
    in_group = pairs.ID_1.isin(grouped) | pairs.ID_2.isin(grouped)
    print("group rejection: %d pairs dropped (%d galaxies in >=2 pairs)"
          % (in_group.sum(), len(grouped)))
    pairs = pairs[~in_group].reset_index(drop=True)

pairs.to_csv(CONFIG["OUTPUT_CSV"], index=False)
print("saved %d pairs to %s" % (len(pairs), CONFIG["OUTPUT_CSV"]))
print("cut summary:", stats)
print("a = %s, b = %s, H0 = %s, NED fill = %d" % (a_iso, b_iso, H0, n_filled))
