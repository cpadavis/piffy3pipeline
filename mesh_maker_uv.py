"""
Take Aaron's meshes and save them in fits format piff can read

Now we fix the uv coordinate transformation
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import fitsio
import glob

import piff

decaminfo = piff.des.decaminfo.DECamInfo()

out_dir = '/nfs/slac/g/ki/ki19/des/cpd/des_meshes/mesh_maker_uv'

meshes = []
mesh_dirs = [
             '/u/ec/roodman/Astrophysics/Donuts/ComboMeshes',
             '/u/ec/roodman/Astrophysics/Donuts/ComboMeshesv20',
             '/u/ec/roodman/Astrophysics/Donuts/ComboMeshesv21',
             ]
mesh_values = {}
for mesh_dir in mesh_dirs:
    print('looking in', mesh_dir)
    mesh_files = glob.glob(mesh_dir + '/*Mesh_*_All.dat')
    print('found {0} files'.format(len(mesh_files)))

    # OK now we want to go through all of these, get their column values. Then we need to recombine
    for mesh_name in mesh_files:
        zindex = int(mesh_name.split('/')[-1].split('Mesh_')[0][1:])
        expid = mesh_name.split('/')[-1].split('Mesh_')[-1].split('_All.dat')[0]
        if 'FandA' in expid:
            continue
        if zindex == 4:
            print('    ', expid)
        dfi = pd.read_csv(mesh_name, delim_whitespace=True, names=['chip', 'focal_x', 'focal_y',
                                                                   'z{0}'.format(zindex), 'w{0}'.format(zindex)])
        dfi['expid'] = expid

        if expid not in mesh_values:
            mesh_values[expid] = {}
        mesh_values[expid][zindex] = dfi

mkeys = mesh_values.keys()
print('concatenating {0} meshes'.format(len(mkeys)))
for expid in mkeys:
    print('    ', expid)
    keys = mesh_values[expid].keys()
    for zindex in keys:
        dfi = mesh_values[expid][zindex]
        if zindex == keys[0]:
            df = dfi
        else:
            len_dfi = len(dfi)
            len_df = len(df)
            zkey = 'z{0}'.format(zindex)
            wkey = 'w{0}'.format(zindex)
            df[wkey] = dfi[wkey]
            df[zkey] = dfi[zkey]
            len_dfdfi = len(df)
            if len_dfdfi != len_df:
                print(expid, zindex, len_dfi, len_df, len_dfdfi)
    meshes.append(df)

df = pd.concat(meshes, ignore_index=True)

print('correcting z4')
# gotta correct z4 to waves == / by 172.
df['z4'] = df['z4'] * 1. / 172

print('correcting UV issues')
z6 = df['z6'].values.copy()
df.drop('z6', axis=1, inplace=True)
df['z6'] = -z6
z7 = df['z7'].values.copy()
df.drop('z7', axis=1, inplace=True)
z8 = df['z8'].values.copy()
df.drop('z8', axis=1, inplace=True)
df['z8'] = z7
df['z7'] = z8
z9 = df['z9'].values.copy()
df.drop('z9', axis=1, inplace=True)
z10 = df['z10'].values.copy()
df.drop('z10', axis=1, inplace=True)
df['z9'] = -z10
df['z10'] = -z9

# repeat for weights
w6 = df['w6'].values.copy()
df.drop('w6', axis=1, inplace=True)
df['w6'] = w6
w7 = df['w7'].values.copy()
df.drop('w7', axis=1, inplace=True)
w8 = df['w8'].values.copy()
df.drop('w8', axis=1, inplace=True)
df['w8'] = w7
df['w7'] = w8
w9 = df['w9'].values.copy()
df.drop('w9', axis=1, inplace=True)
w10 = df['w10'].values.copy()
df.drop('w10', axis=1, inplace=True)
df['w9'] = w10
df['w10'] = w9

print('setting nans to 0')
if 'z11' not in df:
    df['z11'] = 0
if 'w11' not in df:
    df['w11'] = 0
for col in df:
    df[col][df[col] != df[col]] = 0

# make sure ccdnum is key ccdnum
print('adding ccdnum information')
df['ccdnum'] = np.array([decaminfo.infoDict[x]['chipnum'] for x in df['chip']])
df['chipnum'] = df['ccdnum']

# also add pixel values
print('adding pixel coordinates')
ix, iy = decaminfo.getPixel_chipnum(df['chipnum'], df['focal_x'], df['focal_y'])
df['x'] = ix
df['y'] = iy

# save each kind to fitsio
print('saving each to separate fits files')
for name in df['expid'].unique():
    print(name)
    dfi = df[df['expid'] == name]
    recarray = dfi.to_records()
    filename = '{0}/{1}.fits'.format(out_dir, name)
    fitsio.write(filename, recarray, clobber=True)

# save to hdf
print('saving to hdf')
df.to_hdf('{0}/meshes.h5'.format(out_dir), 'data')
