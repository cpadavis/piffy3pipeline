"""
Do the analytic shapes. Use argparse to do either each shape individually, or run it all together. A bit of a mess.
"""
# coding: utf-8

# # Analytic Coefficients for Piff Optical
#
# This notebook creates the Piff analytic coefficients and makes plots illustrating their accuracy.
#
# Our goal is to map input atmosphere and zernike coefficients to HSM shapes. This is a bit tricky since we're mapping something like 10-20 variables to 3 (7 if we include third moments). We take the approach here of fitting pairs of variables in succession.
from __future__ import print_function, division
# import matplotlib.pyplot as plt
import numpy as np
import copy
# import os
from itertools import product
from sklearn.linear_model import LinearRegression
import galsim
import piff

from piff.optatmo_psf import poly, poly_full

test_mode = False
out_dir = '/nfs/slac/g/ki/ki18/cpd/Projects/piff_des/analytics'

# In[310]:


config = piff.read_config('/u/ki/cpd/ki19/piff_test/y3/mar_mesh_configs_fix_sph/00233466/Science-20121120s1-v20i2_limited/2018.03.29/config.yaml')


# In[311]:


# create an OptAtmoPSF for drawing
psf_fit = piff.read('/u/ki/cpd/ki19/piff_test/y3/mar_mesh_configs_fix_sph/00233466/Science-20121120s1-v20i2_limited/2018.03.29/psf.piff')


# In[312]:


# load up some stars
# do import modules so we can import pixmappy wcs
galsim.config.ImportModules(config)
# only load one ccd
config['input']['image_file_name'] = config['input']['image_file_name'].replace('*', '10')
# only load 10 stars in the ccd
config['input']['nstars'] = 10

if test_mode:
    config['input']['stamp_size'] = 15

stars, wcs, pointing = piff.Input.process(config['input'])


# In[313]:


config['psf']['jmax_pupil'] = 21  # get the higher zernikes while we're at it

if test_mode:
    config['psf']['jmax_pupil'] = 7  # test


psf = piff.PSF.process(copy.deepcopy(config['psf']))


# In[314]:


# create the function for drawing and measuring stars
from piff.util import hsm_higher_order
def draw_and_measure(param, star=stars[0]):
    params = psf.getParams(star)
    prof = psf.getProfile(param)
    star_drawn = psf.drawProfile(star, prof, params)
    # remove weighting and masking
    star_drawn.data.weight.array[:] = 1.
#     shape = psf.measure_shape(star_drawn, return_error=False)
    shape = hsm_higher_order(star_drawn)  # also returns 3rd order moments
    return shape, star_drawn


# In[315]:


nparams = psf.jmax_pupil + 6


# In[318]:


Nsamples = 201

if test_mode:
    Nsamples = 5

sizes = np.linspace(0.45, 1.5, Nsamples)
gs = np.linspace(-0.1, 0.1, Nsamples)
zs = np.linspace(-1.5, 1.5, Nsamples)


# In[319]:


def determine_unique_indices(list_of_lists):
    values = []
    for l in list_of_lists:
        for li in l:
            values.append(li)
    values = sorted(list(set(values)))
    return values


# In[320]:


# create sets of batches
all_batch_indices_flat = []
all_batch_indices = []

# atmosphere size
batch_indices = [[0, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 1],
                 [0, 1, 1, 1],
                 [1, 1, 1, 1]]
for indx in batch_indices:
    all_batch_indices_flat.append(indx)
all_batch_indices.append(batch_indices)

# single ellipticity or zernike with size
for j in range(2, psf.jmax_pupil + 1):
    batch_indices = []
    indices = [[0, 0, 0, j],
               [0, 0, j, j],
               [0, j, j, j],
               [j, j, j, j]]
    for indx in indices:
        indx = sorted(indx)
        if indx not in all_batch_indices_flat:
            batch_indices.append(indx)
            all_batch_indices_flat.append(indx)
    all_batch_indices.append(batch_indices)

for j in range(2, psf.jmax_pupil + 1):
    batch_indices = []
    indices = [[0, 0, 1, j],
               [0, 1, 1, j],
               [1, 1, 1, j],
               [0, 1, j, j],
               [1, 1, j, j],
               [1, j, j, j]]
    for indx in indices:
        indx = sorted(indx)
        if indx not in all_batch_indices_flat:
            batch_indices.append(indx)
            all_batch_indices_flat.append(indx)
    all_batch_indices.append(batch_indices)

# double zernike
for i in range(4, psf.jmax_pupil + 1):
    for j in range(i + 1, psf.jmax_pupil + 1):
        batch_indices = []
        indices = [[0, 0, i, j],
                   [0, i, i, j],
                   [0, i, j, j],
                   [i, i, j, j],
                   [i, i, i, j],
                   [i, j, j, j]]
        for indx in indices:
            indx = sorted(indx)
            if indx not in all_batch_indices_flat:
                batch_indices.append(indx)
                all_batch_indices_flat.append(indx)
        all_batch_indices.append(batch_indices)

# double zernike with size
for i in range(4, psf.jmax_pupil + 1):
    for j in range(i + 1, psf.jmax_pupil + 1):
        batch_indices = []
        indices = [[0, 1, i, j],
                   [1, i, i, j],
                   [1, 1, i, j],
                   [1, i, j, j]]
        for indx in indices:
            indx = sorted(indx)
            if indx not in all_batch_indices_flat:
                batch_indices.append(indx)
                all_batch_indices_flat.append(indx)
        all_batch_indices.append(batch_indices)

# triple zernike
for i in range(4, psf.jmax_pupil + 1):
    for j in range(i + 1, psf.jmax_pupil + 1):
        for k in range(j + 1, psf.jmax_pupil + 1):
            batch_indices = []
            indices = [[0, i, j, k],
                       [i, i, j, k],
                       [i, j, j, k],
                       [i, j, k, k]]
        for indx in indices:
            indx = sorted(indx)
            if indx not in all_batch_indices_flat:
                batch_indices.append(indx)
                all_batch_indices_flat.append(indx)
        all_batch_indices.append(batch_indices)

print(len(all_batch_indices))

for i in range(4, psf.jmax_pupil + 1):
    for j in range(i + 1, psf.jmax_pupil + 1):
        for k in range(j + 1, psf.jmax_pupil + 1):
            batch_indices = []
            indices = [[1, i, j, k]]
        for indx in indices:
            indx = sorted(indx)
            if indx not in all_batch_indices_flat:
                batch_indices.append(indx)
                all_batch_indices_flat.append(indx)
        all_batch_indices.append(batch_indices)


print(len(all_batch_indices_flat))
print(len(all_batch_indices))


# first index is shape, second index is particular set of indices or coefs
master_indices = [[], [], [], [], [], [], []]
master_coefs = [[], [], [], [], [], [], []]
master_batch_indices = [[], [], [], [], [], [], []]
master_batch_coefs = [[], [], [], [], [], [], []]
fewer_master_indices = [[], [], [], [], [], [], []]
fewer_master_coefs = [[], [], [], [], [], [], []]

batch_shapes = []
batch_params_analytic = []
batch_params_draw = []
batch_model_shapes = []

chi2s = [[], [], [], [], [], [], []]  # tied to batch

model = LinearRegression(fit_intercept=False, copy_X=True)
fewer_model = LinearRegression(fit_intercept=False, copy_X=True)

def run_batch(batch_i):
    batch = all_batch_indices[batch_i]
    # figure out which params
    unique_indices = determine_unique_indices(batch)
    print('batch {0}'.format(batch_i))
    print(unique_indices)

    # turn into values
    unique_values = []
    for val in unique_indices:
        if val == 0:
            continue
        elif val == 1:
            if len(unique_indices) >= 4:
                unique_values.append(sizes[::4])
            else:
                unique_values.append(sizes)
        elif val in [2, 3]:
            unique_values.append(gs)
        else:
            if len(unique_indices) >= 4:
                unique_values.append(zs[::4])
            else:
                unique_values.append(zs)
    iterator = product(*unique_values)

    # generate indices and shapes
    params_analytic = []
    params_draw = []
    shapes = []
    for vals in iterator:
        param_draw = np.zeros(nparams)
        param_analytic = np.zeros(nparams - 3 + 1)  # 1 for onehot, -3 for the extra 3 in middle
        param_analytic[0] = 1
        # also set the size to 1 unless otherwise noted
        param_analytic[1] = 1.
        param_draw[0] = 1.
        for i, val in enumerate(vals):
            if 0 in unique_indices:
                ind = unique_indices[i + 1] - 1
            else:
                ind = unique_indices[i] - 1

            param_analytic[ind + 1] = val
            # put into param_draw
            if ind >= 3:
                # account for the extra 3 params\
                ind += 3
            param_draw[ind] = val
        try:
            shape, star_drawn = draw_and_measure(param_draw)
        except:
            print('Failed!')
            print(unique_indices)
            print(vals)
            print(param_draw)
            print(param_analytic)
            print('skipping')
            continue
        shape = np.array(shape)

        if np.any(shape != shape):
            print('Failure cause of NaNs')
            print(vals)
            continue
        else:
            shapes.append(shape[3:])
            params_analytic.append(param_analytic)
            params_draw.append(param_draw)
    print('shapes drawn and measured')
    shapes = np.array(shapes)
    params_analytic = np.array(params_analytic)
    params_draw = np.array(params_draw)

    np.save('{0}/shapes_{1}.npy'.format(out_dir, batch_i), shapes)
    np.save('{0}/params_analytic_{1}.npy'.format(out_dir, batch_i), params_analytic)
    np.save('{0}/params_draw_{1}.npy'.format(out_dir, batch_i), params_draw)

def solve_and_merge():
    fails = []
    batch_flat = []
    shape_flat = []
    for batch_i, batch in enumerate(all_batch_indices):

        try:
            shapes = np.load('{0}/shapes_{1}.npy'.format(out_dir, batch_i))
            params_analytic = np.load('{0}/params_analytic_{1}.npy'.format(out_dir, batch_i))
            params_draw = np.load('{0}/params_draw_{1}.npy'.format(out_dir, batch_i))
        except:
            fails.append(batch_i)
            print('Failed {0}'.format(batch_i))
            continue
        print(batch_i, determine_unique_indices(batch), [len(c) for c in master_coefs], [len(c) for c in fewer_master_coefs])

        batch_shapes.append(shapes)
        batch_params_analytic.append(params_analytic)
        batch_params_draw.append(params_draw)

        all_fin_shapes = []
        for i, yshape in enumerate(shapes.T):
            # subtract off expected shapes from current master list
            if len(master_coefs[i]) == 0:
                y = yshape
            else:
                pshape = poly(params_analytic, np.array(master_coefs[i]), np.array(master_indices[i]))
                y = yshape - pshape

            if len(fewer_master_coefs[i]) == 0:
                fewer_y = yshape
            else:
                fewer_pshape = poly(params_analytic, np.array(fewer_master_coefs[i]), np.array(fewer_master_indices[i]))
                fewer_y = yshape - fewer_pshape

            # fit polyomial
            Xpoly = poly_full(params_analytic, np.array(batch))
            model.fit(Xpoly, y)

            # append
            coef = model.coef_.tolist()
            # purge the small ones
            batch_out = []
            coef_out = []
            for coefi, batchi in zip(coef, batch):
                if np.abs(coefi) > 1e-8:
                    coef_out.append(coefi)
                    batch_out.append(batchi)
            master_indices[i] += batch_out
            master_coefs[i] += coef_out

            # repeat for fewer
            fewer_model.fit(Xpoly, fewer_y)
            fewer_batch_out = []
            fewer_coef_out = []
            coef = fewer_model.coef_.tolist()
            for coefi, batchi in zip(coef, batch):
                if np.abs(coefi) > 1e-2:
                    fewer_coef_out.append(coefi)
                    fewer_batch_out.append(batchi)

            fewer_master_indices[i] += fewer_batch_out
            fewer_master_coefs[i] +=   fewer_coef_out

            # master_batch_indices[i].append(batch)
            # master_batch_coefs[i].append(model.coef_.tolist())

            # stats
            # fin_shapes = poly(params_analytic, np.array(master_coefs[i]), np.array(master_indices[i]))
            # all_fin_shapes.append(fin_shapes)
            # chi2i = np.sum(np.square(yshape - fin_shapes))
            # chi2s[i].append(chi2i)
        # all_fin_shapes = np.array(all_fin_shapes).T
        # batch_model_shapes.append(all_fin_shapes)

    # write the analytic coefs
    final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
    np.save(out_dir + '/notebook_analytic_coefs.npy', final_analytic_coefs)
    final_analytic_coefs = {'coefs': fewer_master_coefs, 'indices': fewer_master_indices, 'after_burner': [[0, 1] * len(fewer_master_coefs)]}
    np.save(out_dir + '/notebook_fewer_analytic_coefs.npy', final_analytic_coefs)

def solve_and_merge_flat():
    fails = []
    batch_flat = []
    shapes_flat = []
    params_analytic = []

    for batch_i, batch in enumerate(all_batch_indices):

        try:
            shapes = np.load('{0}/shapes_{1}.npy'.format(out_dir, batch_i))
            param_analytic = np.load('{0}/params_analytic_{1}.npy'.format(out_dir, batch_i))
        except:
            fails.append(batch_i)
            print('Failed {0}'.format(batch_i))
            continue
        print(batch_i)

        batch_flat += batch
        for shape in shapes:
            shapes_flat.append(shape)
        for p in param_analytic:
            params_analytic.append(p)

    params_analytic = np.array(params_analytic)
    shapes_flat = np.array(shapes_flat)

    # try a single mega fit
    print('mega fit')
    Xpoly = poly_full(params_analytic, np.array(batch_flat))
    for i, yshape in enumerate(shapes_flat.T):
        model.fit(Xpoly, yshape)
        coef = fewer_model.coef_.tolist()
        batch_out = []
        coef_out = []
        for coefi, batchi in zip(coef, batch_flat):
            if np.abs(coefi) > 1e-8:
                coef_out.append(coefi)
                batch_out.append(batchi)
        master_indices[i] += batch_out
        master_coefs[i] += coef_out


    final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
    np.save(out_dir + '/notebook_mega_analytic_coefs.npy', final_analytic_coefs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, dest='index', help='which zone to call')
    options = parser.parse_args()
    index = options.index - 1

    if index == -1:
        # merge
        solve_and_merge()
    elif index == -2:
        # merge
        solve_and_merge_flat()
    else:
        run_batch(index)
