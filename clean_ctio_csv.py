"""
Take the CSV aaron gave and do the following:
    - fix the rows which have too many commas
    - filter to just rows we want
    - drop rows with no donut fits
    - add columns corresponding to Piff donut units
"""
from __future__ import print_function, division
import pandas as pd
import os

path = '/nfs/slac/g/ki/ki06/roodman/CtioDB/db-part1.csv'
out_path = '/nfs/slac/g/ki/ki19/des/cpd/piff_test/CtioDB_db-part1.csv'
Nbad = 0
Ntot = 0
with open(path) as f:
    with open(out_path, 'w') as fo:
        for i, line in enumerate(f):
            entries = line.split(',')
            length = len(entries)
            if length != 161:
                print(i, length, entries[0])
                Nbad += 1
            else:
                fo.write(line)
            Ntot += 1
print(Nbad, Ntot)

# OK now filter to just ones with donut measurements, and convert to piff units
fits = pd.read_csv(out_path)
print('loaded csv')

# filter rows
rows_to_keep = ['expid', 'filter',
                'dodx', 'dody', 'dodz', 'doxt', 'doyt',
                'dodxerr', 'dodyerr', 'dodzerr', 'doxterr', 'doyterr',
                'seeing', 'fwhm', 'r50', 'ellipticity', 'whisker',
                'e1', 'e2', 'w1', 'w2',
                ]
for col in fits.columns:
    if ('delta' in col) or ('theta' in col):
        rows_to_keep.append(col)

print('have {0} columns'.format(len(fits.columns)))
fits = fits[rows_to_keep]
print('dropped to {0} columns'.format(len(fits.columns)))

# remove NaN rows. Let's assume if zdelta is NaN, then we don't want it
print('Before NaN filter, {0} rows'.format(len(fits)))
fits = fits.dropna(axis=0, subset=['zdelta'])
print('After NaN filter, {0} rows'.format(len(fits)))

# make new columns
conversion_factor = -129.80769230769229
defocus_parameter = 1. / 172.
key_pairs = [['zPupil004_zFocal001', 'zdelta', defocus_parameter],
             ['zPupil005_zFocal001', 'z5delta', 1.],
             ['zPupil006_zFocal001', 'z6delta', -1.],
             ['zPupil007_zFocal001', 'z8delta', 1.],
             ['zPupil008_zFocal001', 'z7delta', 1.],
             ['zPupil009_zFocal001', 'z10delta', -1.],
             ['zPupil010_zFocal001', 'z9delta', -1.],
             ['zPupil011_zFocal001', 'z11delta', 1.],
             ]
# throw in thetax, thetay
new_key_pairs = []
for new, old, convert in key_pairs:
    new_key_pairs.append([new, old, convert])
    for n, o in [['zFocal002', 'thetay'], ['zFocal003', 'thetax']]:
        new_s = new.replace('zFocal001', n)
        old_s = old.replace('delta', o)
        new_key_pair = [new_s, old_s, convert * conversion_factor]
        new_key_pairs.append(new_key_pair)
key_pairs = new_key_pairs

for new, old, convert in key_pairs:
    try:
        print(new, old, convert)
        fits[new] = fits[old] * convert
    except KeyError:
        print('Did not find {0} in fits. Skipping'.format(old))

print('After AOS, now have {0} columns'.format(len(fits.columns)))

# save
converted_out_path = '/nfs/slac/g/ki/ki19/des/cpd/piff_test/CtioDB_db-part1_filtered.csv'
if os.path.exists(converted_out_path):
    os.remove(converted_out_path)
fits.to_csv(converted_out_path, index=False)
print('writing to {0}'.format(converted_out_path))
fits2 = pd.read_csv(converted_out_path)
import ipdb; ipdb.set_trace()
