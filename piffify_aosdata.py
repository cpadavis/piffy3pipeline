"""
Take AOS fits and translate their terms into Piff terms. This comes about because the AOS units for e.g. tilting a zernike aberration are defined in terms of millimeters on the focal plane, whereas Piff defines them in arcseconds on the sky. They are _basically_ proportional.
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np

"""
so vals[1] / u = 0.00044444, and vals[2] / v same
vals[1] / fy = -0.0077037 and vals[2] / fx same

inverses: 2250 for uv / vals, -129.80769 for fxy / vals

ie fx = -v, fy = -u

thus we get the right value if b_2 = -fy / vals[2] thetax; b_3 = -fx / vals[3] theta_y
"""
conversion_factor = -129.80769230769229

# load AOS
path_out = '/nfs/slac/g/ki/ki18/cpd/Projects/piff_des/aosdata-pifftestimages-piffified.csv'
fits = pd.read_csv('/nfs/slac/g/ki/ki18/cpd/Projects/piff_des/aosdata-pifftestimages-doffr.csv')
fits['expid'] = fits['expid'].astype(int)

new_fits = {'expid': fits['expid'].values}
# translate delta terms
positive_key_pairs = [['zPupil004_zFocal001', 'z4d'],
                      ['zPupil011_zFocal001', 'z11d'],
                      ['zPupil005_zFocal001', 'z5d'],
                      ['zPupil007_zFocal001', 'z8d'],
                      ['zPupil008_zFocal001', 'z7d']]
for new_key, old_key in positive_key_pairs:
    new_fits[new_key] = fits[old_key].values
    # translate thetax, thetay terms
    try:
        new_key_thetax = new_key[:-1] + '3'
        old_key_thetax = old_key[:-1] + 'x'
        new_fits[new_key_thetax] = fits[old_key_thetax].values * conversion_factor
        new_key_thetay = new_key[:-1] + '2'
        old_key_thetay = old_key[:-1] + 'y'
        new_fits[new_key_thetay] = fits[old_key_thetay].values * conversion_factor
    except KeyError:
        pass

negative_key_pairs = [
                      ['zPupil006_zFocal001', 'z6d'],
                      ['zPupil009_zFocal001', 'z10d'],
                      ['zPupil010_zFocal001', 'z9d']]
for new_key, old_key in negative_key_pairs:
    new_fits[new_key] = -fits[old_key].values
    # translate thetax, thetay terms
    try:
        new_key_thetax = new_key[:-1] + '3'
        old_key_thetax = old_key[:-1] + 'x'
        new_fits[new_key_thetax] = fits[old_key_thetax].values * conversion_factor
        new_key_thetay = new_key[:-1] + '2'
        old_key_thetay = old_key[:-1] + 'y'
        new_fits[new_key_thetay] = fits[old_key_thetay].values * conversion_factor
    except KeyError:
        pass

df = pd.DataFrame(new_fits)
df.to_csv(path_out, index=False)
# check that we wrote it OK
df2 = pd.read_csv(path_out)

