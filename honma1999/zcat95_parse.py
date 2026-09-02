"""Parse the CfA Redshift Catalogue, June 1995 (VizieR VII/193, zcat.dat) to
galactic coordinates. A genuine pre-1999 redshift compilation (41,600
galaxies with velocities), used to define which galaxies were redshift-known
at Honma's epoch. Download: https://cdsarc.cds.unistra.fr/ftp/VII/193/zcat.dat
"""
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord, FK4
import astropy.units as u
colspecs = [(0,11),(11,13),(13,15),(15,19),(19,20),(20,22),(22,24),(24,26),(26,31),(31,36),(36,39),(61,67)]
names = ['Name','RAh','RAm','RAs','DEsign','DEd','DEm','DEs','Bmag','Vh','e_Vh','BT']
z = pd.read_fwf('zcat.dat', colspecs=colspecs, names=names)
for c in names[1:]:
    if c != 'DEsign': z[c] = pd.to_numeric(z[c], errors='coerce')
z = z.dropna(subset=['RAh','RAm','RAs','DEd','DEm','Vh'])
ra = 15.0*(z.RAh + z.RAm/60 + z.RAs/3600)
dec = (z.DEd + z.DEm/60 + z.DEs.fillna(0)/3600) * np.where(z.DEsign.astype(str).str.strip()=='-', -1, 1)
g = SkyCoord(ra=ra.values*u.deg, dec=dec.values*u.deg, frame=FK4(equinox='B1950')).galactic
pd.DataFrame({'Name': z.Name.str.strip(), 'l': g.l.deg, 'b': g.b.deg, 'Vh': z.Vh,
              'e_Vh': z.e_Vh, 'Bmag': z.Bmag, 'BT': z.BT}).to_csv('zcat95_galactic.csv', index=False)
print('ZCAT 1995 parsed:', len(z))
