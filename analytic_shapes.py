# Analytic Coefficients for Piff Optical

# TODO: when saving pdfs and variables also indicate that indices are used

from __future__ import print_function, division
# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import copy
# import os
import itertools
from sklearn.linear_model import LinearRegression
import galsim
import piff

from piff.optatmo_psf import poly, poly_full

#####
# thing to modify
#####
config_path = '/u/ki/cpd/ki19/piff_test/y3pipeline/00514872/psf_optatmo.yaml'
verbose = 2
jmax_pupil = 22  # inclusive
samples = 1000
samples_test = 100
order = 6  # max polynomial degree
variables_at_once = 3  # how many different terms allowed at once. I think it has to be less than or equal to order

# verbose = 3
# jmax_pupil = 5  # inclusive
# samples = 10
# samples_test = 2
# order = 4  # max polynomial degree
# variables_at_once = 2  # how many different terms allowed at once. I think it has to be less than or equal to order

# verbose = 3
# jmax_pupil = 10  # inclusive
# samples = 500
# samples_test = 50
# order = 6  # max polynomial degree
# variables_at_once = 2  # how many different terms allowed at once. I think it has to be less than or equal to order

out_file = './analytic/full_analytic_shapes__jmax_{0}__order_{1}__upto_{2}__samples__{3}.npy'.format(jmax_pupil, order, variables_at_once, samples)

np.random.seed(123456)
size_max = 1.5
g_max = 0.15
z_max = 1.5
size_min = 0.4
g_min = -g_max
z_min = -z_max

# we also will filter if the parameter std is less than this value
do_filter = True
min_shapes_ranges = [0.01, 0.01, 0.01,
                     0.002, 0.002, 0.002,
                     0.002, 0.002, 0.002, 0.002
                     ]

nshapes = 10
natmoparams = 3
nparams = jmax_pupil - 3 + natmoparams

base_params_draw = np.zeros(nparams + natmoparams)
base_params_draw[natmoparams] = 0.7  # sets size to 0.7 by default -- good seeing
base_params_analytic = np.zeros(nparams + 1)  # for the onehot
base_params_analytic[0] = 1  # one hot needs to have, well, ones there

# load up an example image to get wcs and such correct
config = piff.read_config(config_path)

# modify config to be just one ccd
config['input']['image_file_name'] = config['input']['image_file_name'].replace('*', '*_20')  # use ccd 20, selected at random

# we really only need one star total
config['input']['nstars'] = 1

# and also up the number of jmax_pupil to desired
config['psf']['jmax_pupil'] = jmax_pupil

if 'modules' in config:
    galsim.config.ImportModules(config)
logger = piff.setup_logger(verbose=verbose)
logger.info('Loading up stars from an image')
stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)

# remove stars weights
for star in stars:
    star.data.weight.array[:] = 1.

base_star = stars[0]

# initialize psf
base_psf = piff.PSF.process(config['psf'], logger=logger)

# initialize sklearn model
model = LinearRegression(fit_intercept=False, copy_X=True)

def make_empty_master_indices_coefs():
    master_coefs = []
    master_indices = []
    for j in range(nshapes):
        master_coefs.append([])
        master_indices.append([])
    return master_indices, master_coefs

# p_indices is in terms of the analytic relations
def measure_shapes(p, p_indices, star=base_star, psf=base_psf, logger=None):
    param_draw = base_params_draw.copy()
    param_analytic = base_params_analytic.copy()
    for i, pi in zip(p_indices, p):
        param_draw[i + natmoparams - 1] = pi
        param_analytic[i] = pi

    prof = psf.getProfile(param_draw)
    drawn_star = psf.drawProfile(star, prof, param_draw)
    shape = psf.measure_shape(drawn_star, return_error=False, logger=logger)
    return shape, param_analytic, drawn_star

def predict_shapes(p, coefs, indices):
    ypred = []
    for coef, index in zip(coefs, indices):
        if len(coef) == 0:
            ypred.append(np.zeros(len(p)))
        else:
            ypred.append(poly(p, np.array(coef), np.array(index)))
    ypred = np.array(ypred).T
    return ypred

def consolidate_indices_median(shapes_indices, shapes_coefs, all_indices, all_coefs, logger):
    # this one _adds_ the new coefs to the old ones
    # shapes_of_indices.shape = (nshapes, ncoefs, norder)
    # shapes_of_coefs.shape = (nshapes, ncoefs)
    # all_coefs = (nfits, nshapes, ncoefs)
    # all_indices = (nfits, nshapes, ncoefs, norder)
    for new_indices, new_coefs in zip(all_indices, all_coefs):
        for shapes_indices_i, shapes_coefs_i, indices, coefs in zip(shapes_indices, shapes_coefs, new_indices, new_coefs):
            for index, coef in zip(indices, coefs):
                index = sorted(index)
                if index in shapes_indices_i:
                    loc = shapes_indices_i.index(index)
                    shapes_coefs_i[loc].append(coef)
                else:
                    shapes_indices_i.append(index)
                    shapes_coefs_i.append([coef])
    final_shapes_coefs = []
    for i, shapes_coefs_i in enumerate(shapes_coefs):
        final_shapes_coefs.append([])
        for j, coefs in enumerate(shapes_coefs_i):
            coef = np.median(coefs)
            print(shapes_indices[i][j], coef)
            print(coefs)
            final_shapes_coefs[i].append(coef)

    return shapes_indices, shapes_coefs

def consolidate_indices(shapes_indices, shapes_coefs, new_indices, new_coefs, logger):
    # this one _adds_ the new coefs to the old ones
    # shapes_of_indices.shape = (nshapes, ncoefs, norder)
    # shapes_of_coefs.shape = (nshapes, ncoefs)
    # same with new_
    for shapes_indices_i, shapes_coefs_i, indices, coefs in zip(shapes_indices, shapes_coefs, new_indices, new_coefs):
        for index, coef in zip(indices, coefs):
            index = sorted(index)
            if index in shapes_indices_i:
                loc = shapes_indices_i.index(index)
                shapes_coefs_i[loc] += coef
            else:
                shapes_indices_i.append(index)
                shapes_coefs_i.append(coef)

    return shapes_indices, shapes_coefs

def generate_sample(p_indices, nsamples, logger):
    logger.debug('generating shapes')
    p_min = []
    p_max = []
    for pi in p_indices:
        if pi == 1:
            p_min.append(size_min)
            p_max.append(size_max)
        elif pi == 2 or pi == 3:
            p_min.append(g_min)
            p_max.append(g_max)
        else:
            p_min.append(z_min)
            p_max.append(z_max)
    p_min = np.array(p_min)
    p_max = np.array(p_max)

    p_params_full = np.random.random((nsamples, len(p_indices))) * (p_max - p_min) + p_min

    p_analytic = []
    p_shapes = []
    p_params = []
    # generate values
    for p in p_params_full:
        # measure shapes
        try:
            shape, param_analytic, drawn_star = measure_shapes(p, p_indices)
            p_shapes.append(shape)
            p_analytic.append(param_analytic)
            p_params.append(p)
        except piff.ModelFitError:
            logger.warning('Failed model fit for params {0} indices {1}'.format(str(p), str(p_indices)))
    p_analytic = np.array(p_analytic)
    p_shapes = np.array(p_shapes)
    p_params = np.array(p_params)

    return p_params, p_shapes, p_analytic

def fit_model(p_indices, samples, samples_test, master_indices, master_coefs, logger):

    # make fit indices
    p_fit_indices = np.array([_ for _ in itertools.combinations_with_replacement([0] + p_indices, order)])
    logger.debug('p fit indices:')
    logger.debug('{0}'.format(str(p_fit_indices)))

    p_params, p_shapes, p_analytic = generate_sample(p_indices, samples + samples_test, logger)

    # make prediction from current coefs
    Xpoly = poly_full(p_analytic, p_fit_indices)
    if len(master_coefs[0]) != 0:
        ypred = predict_shapes(p_analytic, master_coefs, master_indices)

    new_indices = []
    new_coefs = []
    for j in range(p_shapes.shape[1]):
        logger.debug('Fitting param {0}'.format(len(new_coefs)))
        yi = p_shapes[:, j]
        if len(master_coefs[j]) == 0:
            logger.debug('No master coefs prediction')
            yfit = yi
        else:
            ypredi = ypred[:, j]
            yfit = yi - ypredi

        # fit but remove bit accounted for by previous fits
        model.fit(Xpoly[:samples], yfit[:samples])
        coefs = model.coef_.tolist()

        # this would be where you could filter
        filtered_indices = p_fit_indices
        filtered_coefs = coefs

        new_indices.append(filtered_indices)
        new_coefs.append(filtered_coefs)
    logger.debug('indices, coefs:')
    logger.debug('{0}'.format(str(new_indices)))
    logger.debug('{0}'.format(str(new_coefs)))

    return new_indices, new_coefs, p_shapes, p_analytic, p_params

def evaluate_fit(p_indices, p_params, y, ymodel, samples, logger):
    # chi2 on training and test samples, by shape
    train_chi2 = np.mean(np.square(y - ymodel)[:samples], axis=0)
    train_str = 'chi2 on train set:'
    for chi2 in train_chi2:
        train_str += ' {0:.3e}'.format(chi2)
    logger.info(train_str)
    # also print the variance of the y == what you would have gotten
    test_chi2 = np.mean(np.square(y - ymodel)[samples:], axis=0)
    test_str = 'chi2 on test  set:'
    for chi2 in test_chi2:
        test_str += ' {0:.3e}'.format(chi2)
    logger.info(test_str)
    train_var = np.var(y[:samples], axis=0)
    var_str = 'var  on train set:'
    for var in train_var:
        var_str += ' {0:.3e}'.format(var)
    logger.info(var_str)
    test_var = np.var(y[samples:], axis=0)
    var_str = 'var  on test  set:'
    for var in test_var:
        var_str += ' {0:.3e}'.format(var)
    logger.info(var_str)

    logger.debug('producing plots')
    # plots
    xlabels = ['', 'size', 'g1', 'g2'] + ['z{0:02d}'.format(zernike) for zernike in range(4, jmax_pupil + 1)]
    ylabels = ['flux', 'u0', 'v0', 'e0', 'e1', 'e2', 'zeta1', 'zeta2', 'delta1', 'delta2']
    fig, axs = plt.subplots(ncols=2 * len(p_indices), nrows=y.shape[1], squeeze=False,
                            figsize=(4 * 2 * len(p_indices), 3 * y.shape[1]))
    for j in range(y.shape[1]):
        ylabel = ylabels[j]
        for k in range(len(p_indices)):
            pk = p_indices[k]
            # data
            xlabel = xlabels[pk]
            ax = axs[j, 2 * k]  # ax is [row, col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # C0 = train; C1 = test, . = data, x = model
            ax.plot(p_params[:samples, k], y[:samples, j], 'o', color='C0', alpha=0.3)
            ax.plot(p_params[samples:, k], y[samples:, j], 'o', color='C1', alpha=0.3)
            ax.plot(p_params[:samples, k], ymodel[:samples, j], 'x', color='C2', alpha=0.3)
            ax.plot(p_params[samples:, k], ymodel[samples:, j], 'x', color='C3', alpha=0.3)

            # residual
            ax = axs[j, 2 * k + 1]  # ax is [row, col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel('data - model: ' + ylabel)
            ax.plot(p_params[:samples, k], y[:samples, j] - ymodel[:samples, j], '.', color='C0')
            ax.plot(p_params[samples:, k], y[samples:, j] - ymodel[samples:, j], '.', color='C1')
    fig.tight_layout()
    return fig

# generate all p indices
all_p_indices = []
for repeat in range(1, variables_at_once + 1):
    all_p_indices_i = [list(_) for _ in itertools.combinations([asdf for asdf in range(1, nparams + 1)], repeat)]
    all_p_indices += all_p_indices_i

logger.info('Fitting {0} combinations'.format(len(all_p_indices)))

def fit_and_evaluate(ith_fit, master_indices, master_coefs):
    p_indices = all_p_indices[ith_fit]
    logger.info('Fitting {0}: {1}'.format(ith_fit, p_indices))

    logger.info('Performing fit {0} for: {1}'.format(ith_fit, p_indices))
    new_indices, new_coefs, p_shapes, p_analytic, p_params = fit_model(p_indices, samples, samples_test, master_indices, master_coefs, logger)


    # filter
    shapes_range = np.max(p_shapes, axis=0) - np.min(p_shapes, axis=0)
    if do_filter:
        filtered_new_indices = []
        filtered_new_coefs = []
        for indices, coefs, shapes_range, min_shapes_range in zip(new_indices, new_coefs, shapes_range, min_shapes_ranges):
            if shapes_range < min_shapes_range:
                filtered_new_indices.append([])
                filtered_new_coefs.append([])
                logger.warning('Skipping Shape {0} because range {1} is less than min range {2}'.format(len(filtered_new_indices), shapes_range, min_shapes_range))
            else:
                filtered_new_indices.append(indices)
                filtered_new_coefs.append(coefs)
    else:
        filtered_new_indices = new_indices
        filtered_new_coefs = new_coefs

    master_indices, master_coefs = consolidate_indices(master_indices, master_coefs, filtered_new_indices, filtered_new_coefs, logger)

    for shape_i, coef in enumerate(master_coefs):
        logger.debug('Currently have {0} coefficients for shape {1}'.format(len(coef), shape_i))

    # and save ones with flux and centroid shifts for records
    full_final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
    np.save(out_file.replace('.npy', '_{0:04d}.npy'.format(ith_fit)), full_final_analytic_coefs)

    logger.debug('predicing shapes for chi2')
    # predict shapes to go into ith_shapes_model
    p_shapes_model = predict_shapes(p_analytic, master_coefs, master_indices)

    #####
    # evaluate fit
    #####
    fig = evaluate_fit(p_indices, p_params, p_shapes, p_shapes_model, samples, logger)
    out_file_i = out_file.replace('.npy', '_{0:04d}'.format(ith_fit))
    out_file_i += '_indices_'
    for i in p_indices:
        out_file_i += '_{0}'.format(i)
    out_file_i += '.pdf'
    fig.savefig(out_file_i)
    plt.close('all')
    plt.close()
    return master_indices, master_coefs

def merge():

    master_indices, master_coefs = make_empty_master_indices_coefs()

    logger.info('consolidating indices')

    # load up indices
    all_indices = []
    all_coefs = []
    for ith_fit in range(len(all_p_indices)):
        temp_file = out_file.replace('.npy', '_{0:04d}.npy'.format(ith_fit))
        temp_analytic_coefs = np.load(temp_file).item()
        new_coefs = temp_analytic_coefs['coefs']
        new_indices = temp_analytic_coefs['indices']

        all_coefs.append(new_coefs)
        all_indices.append(new_indices)

    master_indices, master_coefs = consolidate_indices_median(master_indices, master_coefs, all_indices, all_coefs, logger)

    # and save ones with flux and centroid shifts for records
    full_final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
    np.save(out_file, full_final_analytic_coefs)

    return full_final_analytic_coefs

def test(full_final_analytic_coefs):
    master_coefs = full_final_analytic_coefs['coefs']
    master_indices = full_final_analytic_coefs['indices']

    final_samples = 10 * (samples + samples_test)
    logger.info('Creating test sample from {0} samples'.format(final_samples))
    p_indices = range(1, nparams + 1)

    p_params, p_shapes, p_analytic = generate_sample(p_indices, final_samples, logger)
    logger.debug('predicing shapes for chi2')
    # predict shapes to go into ith_shapes_model
    p_shapes_model = predict_shapes(p_analytic, master_coefs, master_indices)
    fig = evaluate_fit(p_indices, p_params, p_shapes, p_shapes_model, 1, logger)  # not sure if the fitter would like having 0 entries for one of the plots
    fig.savefig(out_file.replace('.npy', '_all.pdf'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, dest='index', help='which fit to call', default=0)
    options = parser.parse_args()
    index = options.index - 1

    index_iter = 50  # do 50 with each call


    if index == -1:
        # run the fulllll thing adding each at a time
        master_indices, master_coefs = make_empty_master_indices_coefs()
        for j in range(0, len(all_p_indices)):
            master_indices, master_coefs = fit_and_evaluate(j, master_indices, master_coefs)
        full_final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
        np.save(out_file, full_final_analytic_coefs)
        test(full_final_analytic_coefs)
    elif index == -2:
        full_final_analytic_coefs = merge()
        np.save(out_file, full_final_analytic_coefs)
        test(full_final_analytic_coefs)
    elif index == -3:
        # run full but with median attachments
        all_indices = []
        all_coefs = []
        for j in range(0, len(all_p_indices)):
            master_indices, master_coefs = make_empty_master_indices_coefs()
            master_indices, master_coefs = fit_and_evaluate(j, master_indices, master_coefs)
        full_final_analytic_coefs = merge()
        full_final_analytic_coefs = {'coefs': master_coefs, 'indices': master_indices, 'after_burner': [[0, 1] * len(master_coefs)]}
        np.save(out_file, full_final_analytic_coefs)
        test(full_final_analytic_coefs)
    else:
        for j in range(0, index_iter):
            master_indices, master_coefs = make_empty_master_indices_coefs()
            master_indices, master_coefs = fit_and_evaluate(index * index_iter + j, master_indices, master_coefs)
