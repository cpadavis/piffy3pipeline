from __future__ import print_function, division

# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import copy

import os
import sys
import fitsio
import galsim
import piff

from piff.util import measure_snr

def fit_interp(stars, config_interp, psf, logger):
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

def write_stars(stars, file_name, extname='psf_test_stars'):
    fits = fitsio.FITS(file_name, mode='rw')
    piff.Star.write(stars, fits, extname)
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
        file_name = '/'.join(inconfig['image_file_name'].split('/')[:-1]) + '/psf_im_{0}_{1:02d}.fits.fz'.format(expid, int(chipnum))

        ccd_loaded_stars = piff.Star.load_images(ccd_stars, file_name, image_hdu=inconfig['image_hdu'], weight_hdu=inconfig['weight_hdu'], badpix_hdu=inconfig['badpix_hdu'], logger=logger)
        loaded_stars += ccd_loaded_stars

    for loaded_star_i, loaded_star in enumerate(loaded_stars):
        measured_snr = measure_snr(loaded_star)
        loaded_star.data.weight = loaded_star.data.weight * (loaded_star.data.properties['snr'] / measured_snr) ** 2

    return loaded_stars

def measure_star_shape(psf, stars, model_stars, logger=None):
    logger = galsim.config.LoggerWrapper(logger)
    shapes = []
    #for star in model_stars:
    #    if star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
    #        raise AttributeError('model_star that is not of the star class found!')
    for i in range(len(stars)):
        if i % 100 == 0:
            logger.debug('Measuring shape of star {0} of {1}'.format(i, len(stars)))
        star = stars[i]
        model_star = model_stars[i]
        # returns a pandas series
        flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, orth4, orth6, orth8 = psf.measure_shape_orthogonal(star)
        sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_orth4, sigma_orth6, sigma_orth8 = psf.measure_error_orthogonal(star)
        #if model_star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
        #    raise AttributeError('model_star that is not of the star class found!')
        model_flux, model_u0, model_v0, model_e0, model_e1, model_e2, model_zeta1, model_zeta2, model_delta1, model_delta2, model_orth4, model_orth6, model_orth8 = psf.measure_shape_orthogonal(model_star)

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
        properties['data_orth4'] = orth4
        properties['data_orth6'] = orth6
        properties['data_orth8'] = orth8

        properties['snr'] = measure_snr(star)

        properties['data_sigma_flux'] = sigma_flux
        properties['data_sigma_u0'] = sigma_u0
        properties['data_sigma_v0'] = sigma_v0
        properties['data_sigma_e0'] = sigma_e0
        properties['data_sigma_e1'] = sigma_e1
        properties['data_sigma_e2'] = sigma_e2
        properties['data_sigma_zeta1'] = sigma_zeta1
        properties['data_sigma_zeta2'] = sigma_zeta2
        properties['data_sigma_delta1'] = sigma_delta1
        properties['data_sigma_delta2'] = sigma_delta2
        properties['data_sigma_orth4'] = sigma_orth4
        properties['data_sigma_orth6'] = sigma_orth6
        properties['data_sigma_orth8'] = sigma_orth8

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
        properties['model_orth4'] = model_orth4
        properties['model_orth6'] = model_orth6
        properties['model_orth8'] = model_orth8

        # add delta
        shape_keys = ['e0', 'e1', 'e2', 'zeta1', 'zeta2', 'delta1', 'delta2', 'orth4', 'orth6', 'orth8']
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

def fit_psf(directory, config_file_name, print_log, meanify_file_path='', fit_interp_only=False, opt_only="False", no_opt="False"):
    if opt_only = "True":
        opt_only = True
    else:
        opt_only = False
    if no_opt = "True":
        no_opt = True
    else:
        no_opt = False
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
        psf.test_stars = test_stars

        # make output
        config['output']['dir'] = directory
        output = piff.Output.process(config['output'], logger=logger)

    elif no_opt and is_optatmo:
        # load base optics psf
        out_path = '{0}/{1}_opt_only.piff'.format(directory, piff_name)
        logger.info('Loading saved PSF at {0}'.format(out_path))
        psf = piff.read(out_path)

        # load images for train stars
        logger.info('loading train stars')
        psf.stars = load_star_images(psf.stars, config, logger=logger)

        # load test stars and their images
        logger.info('loading test stars')
        test_stars = read_stars(out_path, logger=logger)
        test_stars = load_star_images(test_stars, config, logger=logger)
        psf.test_stars = test_stars

    elif (do_meanify or fit_interp_only) and not is_optatmo:
        # welp, not much to do here. shouldn't even have gotten here! :(
        logger.warning('Somehow passed the meanify to a non-optatmo argument. This should not happen.')
        return

    else:
        if not (no_opt and is_optatmo):
            # load stars
            stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)

            # separate stars
            # set seed


            # initialize psf
            psf = piff.PSF.process(config['psf'], logger=logger)



            # piffify
            logger.info('Fitting PSF')
            if not is_optatmo:
                np.random.seed(12345)
                test_fraction = config.get('test_fraction', 0.2)
                test_indx = np.random.choice(len(stars), int(test_fraction * len(stars)), replace=False)
                test_stars = []
                train_stars = []
                for star_i, star in enumerate(stars):
                    if star_i in test_indx:
                        test_stars.append(star)
                    else:
                        train_stars.append(star)
                psf.fit(train_stars, wcs, pointing, logger=logger)
            else:
                psf.fit(stars, wcs, pointing, logger=logger)
                #test_stars = psf.test_stars
            logger.info('Fitted PSF!')


            # save optical pull cuts and other similar things; some of these will be used in doing cuts on exposures
            if is_optatmo and config['psf']['fit_optics_mode'] == "shape":
                logger.info('number of stars pre cut optical: {0}'.format(psf.number_of_stars_pre_cut_optical))
                logger.info('number of stars post cut optical: {0}'.format(psf.number_of_stars_post_cut_optical))
                number_of_outliers_optical = psf.number_of_outliers_optical
                pull_mean_optical = psf.pull_mean_optical
                pull_rms_optical = psf.pull_rms_optical
                pull_all_stars_optical = psf.pull_all_stars_optical
                chisq_all_stars_optical = psf.chisq_all_stars_optical
                total_redchi_across_iterations = psf.total_redchi_across_iterations
                iterations_indices = range(0,len(total_redchi_across_iterations))
                logger.info('Preparing to save npy files')
                np.save("{0}/number_of_outliers_optical.npy".format(directory),np.array(number_of_outliers_optical))
                np.save("{0}/pull_mean_optical.npy".format(directory),np.array(pull_mean_optical))    
                np.save("{0}/pull_rms_optical.npy".format(directory),np.array(pull_rms_optical))
                logger.info('Preparing to create optical pull histograms as well as chisq histogram')
                for m, moment in enumerate(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"]):
                    plt.figure()
                    plt.hist(pull_all_stars_optical[:,m])
                    plt.savefig("{0}/{1}_optical_pull_hist.png".format(directory, moment))
                plt.figure()
                plt.hist(chisq_all_stars_optical)
                plt.savefig("{0}/optical_chisq_hist.png".format(directory))
                plt.figure()
                plt.scatter(x=iterations_indices,y=total_redchi_across_iterations)
                plt.savefig("{0}/total_redchi_across_iterations.png".format(directory))
                logger.info('Finished creating optical pull histograms as well as chisq histogram')

            # save optatmo_psf_kwargs to an npy file for easy access to optics fit params
            if is_optatmo:
                try:
                    optatmo_psf_kwargs = psf.optatmo_psf_kwargs
                    if True:
                        print("making optatmo_psf_kwargs npy file")
                        logger.info("making optatmo_psf_kwargs npy file")
                        np.save("{0}/optatmo_psf_kwargs.npy".format(directory),optatmo_psf_kwargs)
                    else:
                        print("not making optatmo_psf_kwargs npy file") 
                        logger.info("not making optatmo_psf_kwargs npy file")        
                except:
                    print("failed in either making or not making optatmo_psf_kwargs npy file")
                    logger.info("failed in either making or not making optatmo_psf_kwargs npy file")

            # save optical psf in case you just want to stop after the optical fit and pick up where you left off later
            if is_optatmo:
                # save psf
                # make sure the output is in the right directory
                config['output']['dir'] = directory
                output = piff.Output.process(config['output'], logger=logger)
                logger.info('Saving PSF')
                # save fitted PSF
                output_filename_dummy = copy.deepcopy(output.file_name)
                output_filename_core = output_filename_dummy.split('.')[0]
                output_filename_filetype = output_filename_dummy.split('.')[1]
                output_filename_opt_only = output_filename_core + "_opt_only." + output_filename_filetype
                psf.write(output_filename_opt_only, logger=logger)

                # and write test stars
                test_stars = psf.test_stars
                write_stars(test_stars, output_filename_opt_only)


        # stop here right after the optical fit if is_optatmo and you request to do only the optical fit
        if opt_only and is_optatmo:
            sys.exit()
            

        # fit atmosphere parameters (for train stars)
        if is_optatmo:
            logger.info('Fitting PSF atmosphere parameters (with train stars)')
            logger.info('getting param info for {0} train stars'.format(len(psf.stars)))
            params = psf.getParamsList(psf.stars)
            psf._enable_atmosphere = False
            new_stars = []
            for star_i, star in zip(range(len(psf.stars)), psf.stars):
                if star_i % 100 == 0:
                    logger.info('Fitting star {0} of {1}'.format(star_i, len(psf.stars)))
                try:
                    model_fitted_star = psf.fit_model(star, params=params[star_i], logger=logger)
                    new_stars.append(model_fitted_star)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logger.warning('{0}'.format(str(e)))
                    logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
            psf.stars = new_stars

        # go through the motions of fitting atmosphere parameters for test stars, so we can throw out test stars for which this would fail
        if is_optatmo:
            logger.info('Seeing which test stars would successfully be able to fit PSF atmosphere parameters')
            logger.info('getting param info for {0} test stars'.format(len(psf.test_stars)))
            params = psf.getParamsList(psf.test_stars)
            psf._enable_atmosphere = False
            new_stars = []
            dummy_test_stars = []
            for star_i, star in zip(range(len(psf.test_stars)), psf.test_stars):
                if star_i % 100 == 0:
                    logger.info('Fitting star {0} of {1}'.format(star_i, len(psf.test_stars)))
                try:
                    model_fitted_star = psf.fit_model(star, params=params[star_i], logger=logger)
                    new_stars.append(star)
                    dummy_test_stars.append(model_fitted_star)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logger.warning('{0}'.format(str(e)))
                    logger.warning('Warning! Failed to fit atmosphere model for test star {0}. Removing star'.format(star_i))
            psf.test_stars = new_stars


        # cut (train) stars based on 5sig MAD cut
        if is_optatmo:
            logger.debug("Stripping (train) star fit params from copy stars down to just atmosphere params for fitting with the atmo_interp")
            stripped_stars = psf.stripStarList(copy.deepcopy(psf.stars), logger=logger)
            logger.info('Stripping MAD outliers from (train) star fit params of atmosphere')
            params = np.array([s.fit.params for s in stripped_stars])
            madp = np.abs(params - np.median(params, axis=0)[np.newaxis])
            madcut = np.all(madp <= 5 * 1.48 * np.median(madp)[np.newaxis] + 1e-8, axis=1)
            mad_stars = []
            for si, s, keep in zip(range(len(stripped_stars)), stripped_stars, madcut):
                if keep:
                    mad_stars.append(psf.stars[si])
                else:
                    logger.debug('Removing (train) star {0} based on MAD. params are {1}'.format(si, str(params[si])))
            if len(mad_stars) != len(stars):
                logger.info('Stripped (train) stars from {0} to {1} based on 5sig MAD cut'.format(len(stripped_stars), len(mad_stars)))
            psf.stars = mad_stars

        # cut (test) stars based on 5sig MAD cut
        if is_optatmo:
            logger.debug("Stripping (test) star fit params from copy stars down to just atmosphere params for fitting with the atmo_interp")
            stripped_stars = psf.stripStarList(copy.deepcopy(dummy_test_stars), logger=logger)
            logger.info('Stripping MAD outliers from (test) star fit params of atmosphere')
            params = np.array([s.fit.params for s in stripped_stars])
            madp = np.abs(params - np.median(params, axis=0)[np.newaxis])
            madcut = np.all(madp <= 5 * 1.48 * np.median(madp)[np.newaxis] + 1e-8, axis=1)
            mad_stars = []
            for si, s, keep in zip(range(len(stripped_stars)), stripped_stars, madcut):
                if keep:
                    mad_stars.append(psf.test_stars[si])
                else:
                    logger.debug('Removing (test) star {0} based on MAD. params are {1}'.format(si, str(params[si])))
            if len(mad_stars) != len(stars):
                logger.info('Stripped (test) stars from {0} to {1} based on 5sig MAD cut'.format(len(stripped_stars), len(mad_stars)))
            psf.test_stars = mad_stars


        # one extra round of outlier rejection using the pull from the moments (only up to third moments) after the individual star fit for the atmospheric parameters and the atmospheric parameter MAD cuts
        if is_optatmo:
            for s, stars in enumerate([psf.stars, psf.test_stars]):
                data_shapes_all_stars = []
                data_errors_all_stars = []
                model_shapes_all_stars = []
                for star in stars:
                    data_shapes_all_stars.append(psf.measure_shape_third_moments(star))
                    data_errors_all_stars.append(psf.measure_error_third_moments(star))
                    model_shapes_all_stars.append(psf.measure_shape_third_moments(psf.drawStar(star)))
                data_shapes_all_stars = np.array(data_shapes_all_stars)[:,3:]
                data_errors_all_stars = np.array(data_errors_all_stars)[:,3:]
                model_shapes_all_stars = np.array(model_shapes_all_stars)[:,3:]
                pull_all_stars = (data_shapes_all_stars - model_shapes_all_stars) / data_errors_all_stars #pull is (data-model)/error
                conds_pull = (np.all(np.abs(pull_all_stars) <= 4.0, axis=1)) #all stars with more than 4.0 pull are thrown out
                conds_pull_e0 = (np.abs(pull_all_stars[:,0]) <= 4.0)
                conds_pull_e1 = (np.abs(pull_all_stars[:,1]) <= 4.0)
                conds_pull_e2 = (np.abs(pull_all_stars[:,2]) <= 4.0)
                if s == 0:
                    psf.stars = np.array(psf.stars)[conds_pull].tolist()
                if s == 1:
                    psf.test_stars = np.array(psf.test_stars)[conds_pull].tolist()



        # save psf
        # make sure the output is in the right directory
        config['output']['dir'] = directory
        output = piff.Output.process(config['output'], logger=logger)
        logger.info('Saving PSF')
        # save fitted PSF
        psf.write(output.file_name, logger=logger)

        # and write test stars
        if is_optatmo:
            test_stars = psf.test_stars
        write_stars(test_stars, output.file_name)

    shape_keys = ['e0', 'e1', 'e2', 'zeta1', 'zeta2', 'delta1', 'delta2', 'orth4', 'orth6', 'orth8']
    shape_plot_keys = []
    for key in shape_keys:
        shape_plot_keys.append(['data_' + key, 'model_' + key, 'd' + key])
    if is_optatmo:
        interps = config.pop('interps')
        interp_keys = list(interps.keys())
        if not (do_meanify or fit_interp_only):
            # do noatmo only when we do not have meanify
            interp_keys = interp_keys + ['noatmo']
        for interp_key in interp_keys:
            piff_name = '{0}_{1}'.format(config_file_name, interp_key)
            logger.info('Fitting optatmo interpolate for {0}'.format(interp_key))
            if interp_key == 'noatmo':
                psf.atmo_interp = None
                psf._enable_atmosphere = False
            else:
                # fit interps
                config_interp = interps[interp_key]

                if do_meanify:
                    config_interp['average_fits'] = meanify_file_path
                    piff_name += '_meanified'

                fit_interp(psf.stars, config_interp, psf, logger)

            # save
            out_path = '{0}/{1}.piff'.format(directory, piff_name)
            psf.write(out_path, logger=logger)
            test_stars = psf.test_stars
            write_stars(test_stars, out_path)

            # evaluate
            logger.info('Evaluating {0}'.format(piff_name))


            # save a copy of the data stars (test and train) to npy files
            logger.info('Preparing to copy stars_test and stars_train')
            stars_test = copy.deepcopy(psf.test_stars)
            stars_train = copy.deepcopy(psf.stars)  
            logger.info('Preparing to enter test-train for loop')  
            for label in ["test", "train"]:
                if label == "test":
                    stars_label = stars_test
                    logger.info('stars_label = stars_test has been set')
                else:
                    stars_label = stars_train
                    logger.info('stars_label = stars_train has been set')
                np.save("{0}/stars_{1}_{2}.npy".format(directory, label, piff_name),np.array(stars_label))


########################################################################
########################################################################

            logger.info('Finished saving stars to npy files')

            logger.info('Preparing to draw model stars')
            for stars, label in zip([psf.stars, psf.test_stars], ['train', 'test']):

                # draw model stars in order to measure their shapes
                logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
                delete_list = []
                model_stars = psf.drawStarList(stars)
                logger.info('Preliminarily drew model stars. Will now identify stars failed to be drawn to throw them out.')
                for star_i, star in enumerate(model_stars):
                    #logger.info("star_i: {0}".format(star_i))
                    if star == None: #or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
                        delete_list.append(star_i)
                logger.info("delete_list: {0}".format(delete_list))
                stars = np.delete(stars, delete_list).tolist()
                model_stars = np.delete(model_stars, delete_list).tolist()
                #for star_i, star in enumerate(model_stars):
                #    if star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
                #        if star == None:
                #            logger.info('star == None')
                #        if str(type(star)) == "<class 'NoneType'>":
                #            logger.info('star type appears to be NoneType')
                #        if str(type(star)) == "<class 'piff.star.Star'>":
                #            logger.info('star type not of the star class')
                #        raise AttributeError('model_star that is not of the star class found!')
                logger.info('Finished drawing {0} {1} model stars. Some may have failed to be drawn and were thrown out.'.format(len(stars), label))

                # measure shapes
                logger.info('Preparing to extract and get params (& fit params if train stars) graphs for {0} stars'.format(label))
                logger.info('Using measure_star_shape()')
                shapes = measure_star_shape(psf, stars, model_stars, logger=logger)
                logger.info('Preparing to set up param keys')
                param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2']
                if psf.atmosphere_model == 'vonkarman':
                    param_keys += ['opt_L0']
                param_keys += ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
                logger.info('Finished setting up param keys')

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
                logger.info('Finished extracting and get params (& fit params if train stars) graphs for {0} stars'.format(label))

                logger.info('Creating h5 files for {0} stars'.format(label))
                # save shapes
                shapes.to_hdf('{0}/shapes_{1}_{2}.h5'.format(directory, label, piff_name), 'data', mode='w')

                logger.info('Creating plot 2dhist shapes graphs for {0} stars'.format(label))
                # plot shapes
                fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
                # save
                fig.savefig('{0}/plot_2dhist_shapes_{1}_{2}.pdf'.format(directory, label, piff_name))

                logger.info('Preparing to create plot 2dhist params (& fit params if train stars) graphs for {0} stars'.format(label))
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
                        try:
                            new_star = psf.reflux(star, param, logger=logger)
                            new_stars.append(new_star) 
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as e:
                            logger.warning('{0}'.format(str(e)))
                            logger.warning('Warning! Failed to fit flux and center for star {0}. Ignoring star'.format(star))
                    stars = new_stars
                logger.info('Finished creating plot 2dhist params (& fit params if train stars) graphs for {0} stars'.format(label))

                # do the output processing
                logger.info('Writing Stats Output of {0} stars'.format(label))
                for stat in output.stats_list:
                    stat.compute(psf, stars, logger=logger)
                    file_name = '{0}/{1}_{2}_{3}'.format(directory, label, piff_name, os.path.split(stat.file_name)[1])
                    stat.write(file_name=file_name, logger=logger)

    else:
        logger.info('Evaluating {0}'.format(piff_name))

        for stars, label in zip([psf.stars, test_stars], ['train', 'test']):
        
            np.save("{0}/stars_{1}_{2}.npy".format(directory, label, piff_name),np.array(stars))        
        
            # draw model stars in order to measure their shapes
            logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
            delete_list = []
            model_stars = psf.drawStarList(stars)
            for star_i, star in enumerate(stars):
                if star == None:
                    delete_list.append(star_i)
            stars = np.delete(stars, delete_list).tolist()
            model_stars = np.delete(model_stars, delete_list).tolist()
            logger.info('Finished drawing {0} {1} model stars. Some failed to be drawn and were thrown out.'.format(len(stars), label))

            # measure shapes
            shapes = measure_star_shape(psf, stars, model_stars, logger=logger)

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
    parser.add_argument('--opt_only', default="False")
    parser.add_argument('--no_opt', default="False")
    options = parser.parse_args()
    kwargs = vars(options)
    fit_psf(**kwargs)
