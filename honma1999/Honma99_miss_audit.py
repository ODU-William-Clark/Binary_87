"""Trace every Honma pair through the v2.2 (1999-mode) pipeline and report
exactly where it is lost: catalogue match, member cuts, epoch-known velocity,
error availability, pair cuts, isolation killer, group rejection."""
import numpy as np, pandas as pd
H0, C = 50.0, 299792.458
A_ISO, B_ISO = 2.5, 1.5
cat = pd.read_csv('leda_btc15.8_allsky.csv').dropna(subset=['l2','b2','btc']).reset_index(drop=True)
# epoch-1999 known set (same as finder)
def flag(src_l, src_b, tol=0.02):
    order=np.argsort(src_b); sb=src_b[order]; out=np.zeros(len(cat),bool)
    cl,cb=cat.l2.values,cat.b2.values
    for i in range(len(cat)):
        j0,j1=np.searchsorted(sb,cb[i]-tol),np.searchsorted(sb,cb[i]+tol)
        if j1<=j0: continue
        c=order[j0:j1]; dl=(src_l[c]-cl[i]+180)%360-180
        out[i]=np.hypot(dl*np.cos(np.radians(cb[i])),src_b[c]-cb[i]).min()<tol
    return out
rc3=pd.read_fwf('RC3_data_7_5_1.txt',colspecs=[(123,130),(131,138),(224,230)],names=['l','b','V'])
for c in rc3.columns: rc3[c]=pd.to_numeric(rc3[c],errors='coerce')
rc3=rc3.dropna()
ned=pd.read_csv('ned_shell_galaxies.csv').dropna(subset=['z','gallon','gallat'])
yr=pd.to_numeric(ned.z_bibcode.astype(str).str[:4],errors='coerce'); ned=ned[yr<=1999]
zc=pd.read_csv('zcat95_galactic.csv').dropna(subset=['l','b','Vh'])
known = flag(rc3.l.values,rc3.b.values)|flag(ned.gallon.values,ned.gallat.values)|flag(zc.l.values,zc.b.values)
vel = cat.v.values.copy(); vel[~known]=np.nan
V_SUN,L_A,B_A=308.,np.radians(105.),np.radians(-7.)
lr,br=np.radians(cat.l2.values),np.radians(cat.b2.values)
vlg = vel+V_SUN*(np.sin(br)*np.sin(B_A)+np.cos(br)*np.cos(B_A)*np.cos(lr-L_A))
cat['V_lg']=vlg; cat['known']=known

his=pd.read_csv('honma99_pairs.csv')
raw=pd.read_fwf('honma_table1_full.dat',colspecs=[(0,16),(36,45),(46,55)],names=['Name','GLON','GLAT'])
raw['Name']=raw.Name.str.strip(); coord={r.Name:(r.GLON,r.GLAT) for r in raw.itertuples()}
found=pd.read_csv('leda_pairs_epoch1999.csv'); pre=pd.read_csv('leda_pairs_epoch1999_pregroup.csv')
def inset(df,l1,b1,l2,b2):
    for r in df.itertuples():
        a=np.hypot(((r.l1-l1)+180)%360-180,r.b1-b1)<0.03 and np.hypot(((r.l2-l2)+180)%360-180,r.b2-b2)<0.03
        b=np.hypot(((r.l1-l2)+180)%360-180,r.b1-b2)<0.03 and np.hypot(((r.l2-l1)+180)%360-180,r.b2-b1)<0.03
        if a or b: return True
    return False
def ang(l1,b1,l2,b2):
    l1,b1,l2,b2=map(np.radians,(l1,b1,l2,b2))
    return np.arccos(np.clip(np.sin(b1)*np.sin(b2)+np.cos(b1)*np.cos(b2)*np.cos(l1-l2),-1,1))
def member(l0,b0):
    dl=(cat.l2.values-l0+180)%360-180; d=np.hypot(dl*np.cos(np.radians(b0)),cat.b2.values-b0)
    i=d.argmin(); return cat.iloc[i] if d[i]<0.03 else None

print('%-30s %s' % ('pair', 'fate'))
for _,h in his.iterrows():
    (lA,bA),(lB,bB)=coord[h.name1],coord[h.name2]
    if inset(found,lA,bA,lB,bB): continue
    gA,gB=member(lA,bA),member(lB,bB)
    probs=[]
    for tag,g,hisB in (('A',gA,h.B1),('B',gB,h.B2)):
        if g is None: probs.append('%s: not in HyperLEDA'%tag); continue
        if g.btc>15.0: probs.append('%s: btc=%.2f>15 (his B=%.1f)'%(tag,g.btc,hisB))
        if not g.known: probs.append('%s: no 1999 velocity'%tag)
        elif not (1000<g.V_lg<4500): probs.append('%s: V_lg=%.0f out of window'%(tag,g.V_lg))
        if pd.isna(g.e_v): probs.append('%s: no e_v'%tag)
        if pd.isna(g.e_bt): probs.append('%s: no e_bt'%tag)
        if abs(g.b2)<=20: probs.append('%s: |b|<=20'%tag)
    if probs:
        print('%-30s MEMBER: %s'%(h.name1+'+'+h.name2,'; '.join(probs))); continue
    # pair cuts with our values
    fA,fB=10**(-.4*gA.btc),10**(-.4*gB.btc); mt=-2.5*np.log10(fA+fB)
    vb=(gA.V_lg*fA+gB.V_lg*fB)/(fA+fB); mu=5*np.log10(vb/H0)+25
    L10=(10**(-.4*(gA.btc-mu-5.44))+10**(-.4*(gB.btc-mu-5.44)))/1e10; Lc=L10**(1/3.)
    V=abs(gA.V_lg-gB.V_lg)/(1+vb/C)/Lc; R=2*(vb/H0)*1000*np.tan(ang(lA,bA,lB,bB)/2)/Lc
    if mt>13.5 or V>400 or R>400:
        print('%-30s PAIR CUT: m1+2=%.2f V=%.0f R=%.0f  (his Rp=%.0f Vp=%.0f)'%(h.name1+'+'+h.name2,mt,V,R,h.Rp,h.Vp)); continue
    if inset(pre,lA,bA,lB,bB):
        print('%-30s GROUP REJECTION'%(h.name1+'+'+h.name2)); continue
    # isolation killer
    dl=(lB-lA+180)%360-180; lc=lA+dl*fB/(fA+fB); bc=(bA*fA+bB*fB)/(fA+fB)
    keep=(cat.btc.values<=mt+2)&(cat.pgc.values!=gA.pgc)&(cat.pgc.values!=gB.pgc)
    th=ang(lc,bc,cat.l2.values[keep],cat.b2.values[keep]); r3=2*(vb/H0)*1000*np.tan(th/2)
    v3=np.abs(cat.V_lg.values[keep]-vb); hasv=np.isfinite(v3); insh=hasv&(cat.V_lg.values[keep]>=1000)&(cat.V_lg.values[keep]<=4500)
    bad=np.where(hasv, insh&(r3<=1000)&(v3<=600), r3<=1000)
    k=np.flatnonzero(bad)
    if len(k):
        i=k[np.argmin(r3[k])]; nm=str(cat.objname.values[keep][i])
        print('%-30s ISOLATION: %s btc=%.2f (thr %.2f) r3=%.0f %s'%(h.name1+'+'+h.name2,nm,cat.btc.values[keep][i],mt+2,r3[i],
              ('dv=%.0f'%v3[i]) if hasv[i] else 'z-unknown'))
    else:
        print('%-30s ??? passes everything here'%(h.name1+'+'+h.name2))
