from __future__ import print_function, division

# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import copy

import os
import fitsio
import galsim
import piff

from piff.util import hsm_error, hsm_higher_order, measure_snr

def fit_interp(stars, test_stars, config_interp, psf, chicut=0, madcut=0, snrcut=0, logger=None):
    if chicut:
        new_stars = []
        for star in stars:
            chi2 = star.fit.chisq
            dof = star.image.array.size - (len(star.fit.params) + 3)
            if chi2 / dof < chicut:
                new_stars.append(star)
        stars = new_stars
    if madcut:
        # get mad of fit params
        params = np.array([star.fit.params for star in stars])
        med = np.median(params, axis=0)
        mad = np.median(np.abs(params - med[None]), axis=0) + 1e-8  # 1e-8 for any params that are constant
        new_stars = []
        for star in stars:
            mad_i = np.abs(star.fit.params - med)
            if np.all(mad_i < 1.5 * madcut * mad):  # convert mad to outlier
                new_stars.append(star)
        stars = new_stars
    if snrcut:
        new_stars = []
        for star in stars:
            if star.data.properties['snr'] < snrcut:
                new_stars.append(star)
        stars = new_stars

        # test stars also undergo snr cut
        new_stars = []
        for star in test_stars:
            if star.data.properties['snr'] < snrcut:
                new_stars.append(star)
        test_stars = new_stars
    # init interp
    atmo_interp = piff.Interp.process(copy.deepcopy(config_interp), logger=logger)
    logger.debug("Stripping star fit params down to just atmosphere params for fitting with the atmo_interp")
    stripped_stars = psf.stripStarList(stars, logger=logger)
    logger.debug('Initializing atmo interp')
    initialized_stars = atmo_interp.initialize(stripped_stars, logger=logger)
    logger.debug('Fitting atmo interp assuming no outlier model')
    atmo_interp.solve(initialized_stars, logger=logger)
    logger.debug('Updating PSF with atmo_interp object')
    psf.atmo_interp = atmo_interp
    psf.kwargs['atmo_interp'] = 0
    psf._enable_atmosphere = True

    # since we didn't use outliers, remove any we find
    logger.debug('removing outliers from psf, if present')
    psf.outliers = None
    psf.kwargs['outliers'] = 0

    return stars, test_stars

def write_stars(stars, file_name, extname='psf_test_stars', logger=None):
    fits = fitsio.FITS(file_name, mode='rw')
    piff.Star.write(stars, fits, extname, logger=logger)
    fits.close()

def read_stars(file_name, extname='psf_test_stars', logger=None):
    fits = fitsio.FITS(file_name, mode='rw')
    stars = piff.Star.read(fits, extname, logger=logger)
    fits.close()
    return stars

def load_star_images(stars, config, logger=None):
    inconfig = config['input']

    loaded_stars = []
    # divide up stars by ccdnum to get the specific file_name
    chipnums = np.array([star.data.properties['chipnum'] for star in stars])
    for chipnum in sorted(np.unique(chipnums)):
        ccd_stars = []
        conds = chipnums == chipnum
        for ci, cond in enumerate(conds):
            if cond:
                ccd_stars.append(stars[ci])

        # figure out file_name
        # image_file_name: /nfs/slac/g/ki/ki19/des/cpd/y3_piff/exposures_v29_grizY/512974/*.fits.fz
        # e.g. psf_im_512974_5.fits.fz
        expid = int(inconfig['image_file_name'].split('/')[-2])
        file_name = '/'.join(inconfig['image_file_name'].split('/')[:-1]) + '/psf_im_{0}_{1}.fits.fz'.format(expid, int(chipnum))

        ccd_loaded_stars = piff.Star.load_images(ccd_stars, file_name, image_hdu=inconfig['image_hdu'], weight_hdu=inconfig['weight_hdu'], badpix_hdu=inconfig['badpix_hdu'], logger=logger)
        loaded_stars += ccd_loaded_stars

    return loaded_stars

def measure_star_shape(stars, model_stars, logger=None):
    logger = galsim.config.LoggerWrapper(logger)
    shapes = []
    for i in range(len(stars)):
        if i % 100 == 0:
            logger.debug('Measuring shape of star {0} of {1}'.format(i, len(stars)))
        star = stars[i]
        model_star = model_stars[i]
        # returns a pandas series
        flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2 = hsm_higher_order(star)
        flux, u0, v0, e0, e1, e2, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, return_debug=False, return_error=True)
        model_flux, model_u0, model_v0, model_e0, model_e1, model_e2, model_zeta1, model_zeta2, model_delta1, model_delta2 = hsm_higher_order(model_star)

        properties = {key: star.data.properties[key] for key in ['chipnum', 'x', 'y', 'u', 'v', 'ra', 'dec']}

        properties['data_flux'] = flux
        properties['data_u0'] = u0
        properties['data_v0'] = v0
        properties['data_e0'] = e0
        properties['data_e1'] = e1
        properties['data_e2'] = e2
        properties['data_zeta1'] = zeta1
        properties['data_zeta2'] = zeta2
        properties['data_delta1'] = delta1
        properties['data_delta2'] = delta2

        properties['snr'] = measure_snr(star)

        properties['data_sigma_flux'] = sigma_flux
        properties['data_sigma_u0'] = sigma_u0
        properties['data_sigma_v0'] = sigma_v0
        properties['data_sigma_e0'] = sigma_e0
        properties['data_sigma_e1'] = sigma_e1
        properties['data_sigma_e2'] = sigma_e2

        properties['model_flux'] = model_flux
        properties['model_u0'] = model_u0
        properties['model_v0'] = model_v0
        properties['model_e0'] = model_e0
        properties['model_e1'] = model_e1
        properties['model_e2'] = model_e2
        properties['model_zeta1'] = model_zeta1
        properties['model_zeta2'] = model_zeta2
        properties['model_delta1'] = model_delta1
        properties['model_delta2'] = model_delta2

        # add delta
        shape_keys = ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']
        for key in shape_keys:
            properties['d{0}'.format(key)] = properties['data_{0}'.format(key)] - properties['model_{0}'.format(key)]

        shapes.append(pd.Series(properties))
    shapes = pd.DataFrame(shapes)
    return shapes

def plot_2dhist_shapes(shapes, keys, diff_mode=False, **kwargs):
    u = shapes['u']
    v = shapes['v']

    fig, axs = plt.subplots(nrows=len(keys), ncols=3, figsize=(4 * 3, 3 * len(keys)), squeeze=False)
    # left column gets the Y coordinate label
    for ax in axs:
        ax[0].set_ylabel('v')
    # bottom row gets the X coordinate label
    for i in range(3):
        axs[-1, i].set_xlabel('u')

    for ax_i in range(len(axs)):
        # determine min and max values based on shape key
        if diff_mode:
            x = np.array([shapes[keys[ax_i][0]], shapes[keys[ax_i][1]]]).flatten()
            dx = np.array([shapes[keys[ax_i][0]] - shapes[keys[ax_i][1]]]).flatten()
            x = np.abs(x[np.isfinite(x)])
            dx = np.abs(dx[np.isfinite(dx)])
            fvmax = np.nanpercentile(x, q=90)
            dvmax = np.nanpercentile(dx, q=90)

            if 'e0' in keys[ax_i][0]:
                fvmin = np.nanpercentile(x, q=10)
            else:
                fvmin = -fvmax

            dvmin = -dvmax

        for ax_j in range(len(axs[0])):
            ax = axs[ax_i, ax_j]
            label = keys[ax_i][ax_j]
            C = shapes[label]

            conds = np.isfinite(C)
            u_i = u[conds]
            v_i = v[conds]
            C_i = C[conds]

            if diff_mode:
                cmap = plt.cm.RdBu_r
                if ax_j == 2:
                    vmin = dvmin
                    vmax = dvmax
                else:
                    vmin = fvmin
                    vmax = fvmax
            else:
                vmin = None
                vmax = None
                cmap = None

            kwargs_in = {'gridsize': 30, 'reduce_C_function': np.median,
                         'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
            kwargs_in.update(kwargs)
            im = ax.hexbin(u_i, v_i, C=C_i, **kwargs_in)
            fig.colorbar(im, ax=ax)
            ax.set_title(label)
    plt.tight_layout()

    return fig, axs

def fit_psf(directory, config_file_name, print_log, meanify_file_path='', fit_interp_only=False):
    do_meanify = meanify_file_path != ''
    piff_name = config_file_name
    # load config file
    config = piff.read_config('{0}/{1}.yaml'.format(directory, config_file_name))
    is_optatmo = 'OptAtmo' in config['psf']['type']

    # do galsim modules
    if 'modules' in config:
        galsim.config.ImportModules(config)

    # create logger
    verbose = config.get('verbose', 3)
    if print_log:
        logger = piff.setup_logger(verbose=verbose)
    else:
        if do_meanify:
            logger = piff.setup_logger(verbose=verbose, log_file='{0}/{1}_fit_psf_meanify_logger.log'.format(directory, config_file_name))
        else:
            logger = piff.setup_logger(verbose=verbose, log_file='{0}/{1}_fit_psf_logger.log'.format(directory, config_file_name))

    if (do_meanify or fit_interp_only) and is_optatmo:
        # load base optics psf
        out_path = '{0}/{1}.piff'.format(directory, piff_name)
        logger.info('Loading saved PSF at {0}'.format(out_path))
        psf = piff.read(out_path)

        # load images for train stars
        logger.info('loading train stars')
        psf.stars = load_star_images(psf.stars, config, logger=logger)

        # load test stars and their images
        logger.info('loading test stars')
        test_stars = read_stars(out_path, logger=logger)
        test_stars = load_star_images(test_stars, config, logger=logger)

        # make output
        config['output']['dir'] = directory
        output = piff.Output.process(config['output'], logger=logger)

    elif (do_meanify or fit_interp_only) and not is_optatmo:
        # welp, not much to do here. shouldn't even have gotten here! :(
        logger.warning('Somehow passed the meanify to a non-optatmo argument. This should not happen.')
        return

    else:
        # load stars
        stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)

        # separate stars
        # set seed
        np.random.seed(12345)
        test_fraction = config.get('test_fraction', 0.2)
        test_indx = np.random.choice(len(stars), int(test_fraction * len(stars)), replace=False)
        test_stars = []
        train_stars = []
        # kludgey:
        for star_i, star in enumerate(stars):
            if star_i in test_indx:
                test_stars.append(star)
            else:
                train_stars.append(star)

        # initialize psf
        psf = piff.PSF.process(config['psf'], logger=logger)

        # piffify
        logger.info('Fitting PSF')
        psf.fit(train_stars, wcs, pointing, logger=logger)
        logger.info('Fitted PSF!')

        # fit atmosphere parameters
        if is_optatmo:
            logger.info('Fitting PSF atmosphere parameters')
            logger.info('getting param info for {0} stars'.format(len(psf.stars)))
            params = psf.getParamsList(psf.stars)
            psf._enable_atmosphere = False
            new_stars = []
            for star_i, star in zip(range(len(psf.stars)), psf.stars):
                if star_i % 100 == 0:
                    logger.info('Fitting star {0} of {1}'.format(star_i, len(psf.stars)))
                try:
                    model_fitted_star, results = psf.fit_model(star, params=params[star_i], vary_shape=True, vary_optics=False, logger=logger)
                    new_stars.append(model_fitted_star)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logger.warning('{0}'.format(str(e)))
                    logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
            psf.stars = new_stars

        # save psf
        # make sure the output is in the right directory
        config['output']['dir'] = directory
        output = piff.Output.process(config['output'], logger=logger)
        logger.info('Saving PSF')
        # save fitted PSF
        psf.write(output.file_name, logger=logger)

        # and write test stars
        write_stars(test_stars, output.file_name, logger=logger)

    shape_keys = ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']
    shape_plot_keys = []
    for key in shape_keys:
        shape_plot_keys.append(['data_' + key, 'model_' + key, 'd' + key])
    if is_optatmo:
        interps = config.pop('interps')
        interp_keys = interps.keys()
        if not (do_meanify or fit_interp_only):
            # do noatmo only when we do not have meanify
            interp_keys = ['noatmo'] + interp_keys
        train_stars = psf.stars
        for interp_key in interp_keys:
            piff_name = '{0}_{1}'.format(config_file_name, interp_key)
            logger.info('Fitting optatmo interpolate for {0}'.format(interp_key))
            if interp_key == 'noatmo':
                psf.atmo_interp = None
                psf._enable_atmosphere = False
                passed_test_stars = test_stars
            else:
                # fit interps
                config_interp = interps[interp_key]

                if do_meanify:
                    config_interp['average_fits'] = meanify_file_path
                    piff_name += '_meanified'

                # extract chicut, madcut, snrcut from interp config, if provided
                interp_chicut = config_interp.pop('chicut', 0)
                interp_madcut = config_interp.pop('madcut', 0)
                interp_snrcut = config_interp.pop('snrcut', 0)

                # test stars undergo snr cut, but no other cuts
                used_interp_stars, passed_test_stars = fit_interp(train_stars, test_stars, config_interp, psf, interp_chicut, interp_madcut, interp_snrcut, logger)
                psf.stars = used_interp_stars

            # save
            out_path = '{0}/{1}.piff'.format(directory, piff_name)
            psf.write(out_path, logger=logger)
            write_stars(passed_test_stars, out_path, logger=logger)

            # evaluate
            logger.info('Evaluating {0}'.format(piff_name))

            for stars, label in zip([psf.stars, passed_test_stars], ['train', 'test']):
                # get shapes
                logger.debug('drawing {0} model stars'.format(label))
                model_stars = psf.drawStarList(stars)
                shapes = measure_star_shape(stars, model_stars, logger=logger)

                # TODO: fix this with revised params
                param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2']
                if psf.atmosphere_model == 'vonkarman':
                    param_keys += ['atmo_L0']
                param_keys += ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
                if label == 'train':
                    # if train, plot fitted params
                    logger.info('Extracting training fit parameters')
                    params = np.array([star.fit.params for star in stars])
                    params_var = np.array([star.fit.params_var for star in stars])
                    for i in range(params.shape[1]):
                        shapes['{0}_fit'.format(param_keys[i])] = params[:, i]
                        shapes['{0}_var'.format(param_keys[i])] = params_var[:, i]
                    logger.info('Getting training fit parameters')
                    params = psf.getParamsList(stars)
                    for i in range(params.shape[1]):
                        shapes[param_keys[i]] = params[:, i]

                elif label == 'test':
                    # if test, plot predicted params
                    logger.info('Getting test parameters')
                    params = psf.getParamsList(stars)
                    for i in range(params.shape[1]):
                        shapes[param_keys[i]] = params[:, i]

                # save shapes
                shapes.to_hdf('{0}/shapes_{1}_{2}.h5'.format(directory, label, piff_name), 'data', mode='w')

                # plot shapes
                fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
                # save
                fig.savefig('{0}/plot_2dhist_shapes_{1}_{2}.pdf'.format(directory, label, piff_name))

                # plot params
                plot_keys = []
                plot_keys_i = []
                for i in range(params.shape[1]):
                    plot_keys_i.append(param_keys[i])
                    if len(plot_keys_i) == 3:
                        plot_keys.append(plot_keys_i)
                        plot_keys_i = []
                if len(plot_keys_i) > 0:
                    plot_keys_i += [plot_keys_i[0]] * (3 - len(plot_keys_i))
                    plot_keys.append(plot_keys_i)
                fig, axs = plot_2dhist_shapes(shapes, plot_keys, diff_mode=False)
                # save
                fig.savefig('{0}/plot_2dhist_params_{1}_{2}.pdf'.format(directory, label, piff_name))
                # and repeat for the fit params
                if label == 'train':
                    fig, axs = plot_2dhist_shapes(shapes, [[key + '_fit' for key in kp] for kp in plot_keys], diff_mode=False)
                    # save
                    fig.savefig('{0}/plot_2dhist_fit_params_{1}_{2}.pdf'.format(directory, label, piff_name))

                # if test, fit the flux and centering
                if label == 'test':
                    logger.info('Fitting the centers and fluxes of {0} test stars'.format(len(stars)))
                    # fit stars for stats
                    new_stars = []
                    for star, param in zip(stars, params):
                        new_star, res = psf.fit_model(star, param, vary_shape=False, vary_optics=False, logger=logger)
                        new_stars.append(new_star)
                    stars = new_stars

                # do the output processing
                logger.info('Writing Stats Output of {0} stars'.format(label))
                for stat in output.stats_list:
                    stat.compute(psf, stars, logger=logger)
                    file_name = '{0}/{1}_{2}_{3}'.format(directory, label, piff_name, os.path.split(stat.file_name)[1])
                    stat.write(file_name=file_name, logger=logger)

    else:
        logger.info('Evaluating {0}'.format(piff_name))

        for stars, label in zip([psf.stars, test_stars], ['train', 'test']):
            # get shapes
            shapes = measure_star_shape(stars, psf, logger=logger)
            # save shapes
            shapes.to_hdf('{0}/shapes_{1}_{2}.h5'.format(directory, label, piff_name), 'data', mode='w')

            # plot shapes
            fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
            # save
            fig.savefig('{0}/plot_2dhist_shapes_{1}_{2}.pdf'.format(directory, label, piff_name))

            logger.info('Writing Stats Output of {0} stars'.format(label))
            for stat in output.stats_list:
                stat.compute(psf, stars, logger=logger)
                file_name = '{0}/{1}_{2}_{3}'.format(directory, label, piff_name, os.path.split(stat.file_name)[1])
                stat.write(file_name=file_name, logger=logger)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', action='store', dest='directory',
                        default='.',
                        help='directory of the run. Default to current directory')
    parser.add_argument('--print_log', action='store_true', dest='print_log',
                        help='print logging instead of save')
    parser.add_argument('--fit_interp_only', action='store_true', dest='fit_interp_only',
                        help='load up piff file and stars and only do the fit_interp portion')
    parser.add_argument('--config_file_name', action='store', dest='config_file_name',
                        help='name of the config file (without .yaml)')
    parser.add_argument('--meanify_file_path', action='store', dest='meanify_file_path',
                        default='',
                        help='path to meanify file. If not specified, or if not using optatmo, then ignored')

    options = parser.parse_args()
    kwargs = vars(options)

    fit_psf(**kwargs)
