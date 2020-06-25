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

########################################################################
########################################################################


# function to do Gaussian Process Interpolation (or some other interpolation)
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

# extra function needed for writing test stars
def write_stars(stars, file_name, extname='psf_test_stars'):
    fits = fitsio.FITS(file_name, mode='rw')
    piff.Star.write(stars, fits, extname)
    fits.close()

# extra function needed for reading test stars
def read_stars(file_name, extname='psf_test_stars', logger=None):
    fits = fitsio.FITS(file_name, mode='rw')
    stars = piff.Star.read(fits, extname)
    fits.close()
    return stars

# extra function needed to load stars from a previous (and likely incomplete) fit
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

# this function graphs the stars' data, model, difference, error, and pull images in histograms.
def make_oned_hists_pdf(shapes, directory, label, piff_name):
    list_of_moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
    list_of_moment_arrays = []
    for moment in list_of_moments:
        data_moments = shapes["data_{0}".format(moment)].values
        model_moments = shapes["model_{0}".format(moment)].values
        difference_moments = shapes["d{0}".format(moment)].values
        error_moments = shapes["data_sigma_{0}".format(moment)].values
        pull_moments = difference_moments / error_moments
        moment_array = np.array([data_moments, model_moments, difference_moments, error_moments, pull_moments])
        list_of_moment_arrays.append(moment_array)


    fig, axss = plt.subplots(nrows=10, ncols=5, figsize=(50, 100))#, squeeze=False)
    for r, moment_array, moment in zip(list(range(0,len(list_of_moment_arrays))), list_of_moment_arrays, list_of_moments):
        
        data_moment_array = moment_array[0]
        model_moment_array = moment_array[1]  
        difference_moment_array = moment_array[2] 
        error_moment_array = moment_array[3]   
        pull_moment_array = moment_array[4]          
        
        if moment == "e0":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.003),0.003))                
            axss[r][0].set_xlim(0.36,0.48) 
        if moment == "e1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0035),0.0035))    
            axss[r][0].set_xlim(-0.07,0.07) 
        if moment == "e2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.003),0.003))    
            axss[r][0].set_xlim(-0.06,0.06)  
        if moment == "zeta1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0005),0.0005))    
            axss[r][0].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0005),0.0005))    
            axss[r][0].set_xlim(-0.005,0.015) 
        if moment == "delta1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.00075),0.00075))    
            axss[r][0].set_xlim(-0.015,0.015)  
        if moment == "delta2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.00075),0.00075))    
            axss[r][0].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.01),0.01))    
            axss[r][0].set_xlim(4.1,4.5) 
        if moment == "orth6":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0875),0.0875))    
            axss[r][0].set_xlim(13.0,16.5) 
        if moment == "orth8":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.875),0.875))    
            axss[r][0].set_xlim(55.0,90.0) 
        axss[r][0].set_title("data {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.003),0.003))    
            axss[r][1].set_xlim(0.36,0.48) 
        if moment == "e1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0035),0.0035))    
            axss[r][1].set_xlim(-0.07,0.07) 
        if moment == "e2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.003),0.003))    
            axss[r][1].set_xlim(-0.06,0.06)  
        if moment == "zeta1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0005),0.0005))    
            axss[r][1].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0005),0.0005))    
            axss[r][1].set_xlim(-0.005,0.015) 
        if moment == "delta1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.00075),0.00075))    
            axss[r][1].set_xlim(-0.015,0.015)  
        if moment == "delta2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.00075),0.00075))    
            axss[r][1].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.01),0.01))    
            axss[r][1].set_xlim(4.1,4.5) 
        if moment == "orth6":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0875),0.0875))    
            axss[r][1].set_xlim(13.0,16.5) 
        if moment == "orth8":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.875),0.875))    
            axss[r][1].set_xlim(55.0,90.0) 
        axss[r][1].set_title("model {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "e1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "e2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "zeta1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.0005),0.0005))    
            axss[r][2].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.0005),0.0005))    
            axss[r][2].set_xlim(-0.01,0.01) 
        if moment == "delta1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00075),0.00075))    
            axss[r][2].set_xlim(-0.015,0.015) 
        if moment == "delta2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00075),0.00075))    
            axss[r][2].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.01),0.01))    
            axss[r][2].set_xlim(-0.2,0.2) 
        if moment == "orth6":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.075),0.075))    
            axss[r][2].set_xlim(-1.5,1.5) 
        if moment == "orth8":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.75),0.75))    
            axss[r][2].set_xlim(-15.0,15.0) 
        axss[r][2].set_title("difference {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "e1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "e2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "zeta1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "zeta2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01)
        if moment == "delta1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "delta2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "orth4":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.00375/20.0),0.00375/20.0))    
            axss[r][3].set_xlim(0.0,0.075) 
        if moment == "orth6":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0375/20.0),0.0375/20.0))    
            axss[r][3].set_xlim(0.0,0.75) 
        if moment == "orth8":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.375/20.0),0.375/20.0))    
            axss[r][3].set_xlim(0.0,7.5) 
        axss[r][3].set_title("error {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.2),0.2))    
            axss[r][4].set_xlim(-4.0,4.0) 
        if moment == "e1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "e2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "zeta1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.1),0.1))    
            axss[r][4].set_xlim(-2.0,2.0) 
        if moment == "zeta2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.1),0.1))    
            axss[r][4].set_xlim(-2.0,2.0) 
        if moment == "delta1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "delta2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth4":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth6":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth8":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        axss[r][4].set_title("pull {0} histogram".format(moment))
        
    #plt.tight_layout()
    fig.savefig("{0}/{1}_stars_oned_hists_{2}.pdf".format(directory,label, piff_name))

# TODO: make sure this back-up function always matches the version in optatmo_psf.py from PIFF proper
def measure_shape_orthogonal(star, logger=None):
    """Measure the shape of a star using the HSM algorithm.

    Goes up to third moments plus orthogonal radial moments up to eighth moments.
    Does not return error.

    :param star:                Star we want to measure
    :param logger:              A logger object for logging debug info

    :returns:   Shape in unnormalized basis. Goes up to third moments plus orthogonal radial
                moments up to eighth moments
    """

    # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, xi4, xi6, xi8
    values = piff.util.calculate_moments(star, logger=logger, third_order=True, radial=True)
    values = np.array(values)

    if True:
        # This converts from natural moments to the version Ares had
        # The tests pass without this, but I think that just means they weren't really
        # sufficiently robust.  Probably should just disable this and redo the RF with the
        # new moment definitions.
        hsm = piff.util.hsm(star)
        values[0] *= hsm[0] * star.data.pixel_area * star.data.weight.array.mean()
        values[1] += hsm[1]
        values[2] += hsm[2]
        values[3:] *= 2

    # flux is underestimated empirically
    # MJ: I don't think this ^ is true.  But then, it isn't expected to return the real flux.
    #     For a Gaussian, M00 is flux / (4 pi sigma^2).
    #     Or for your version, it is flux^2 pixel_scale^2 mean(w) / (4 pi sigma^2).
    #     So probably that just happened to come out as 0.92 for whatever test you did.
    #values[0] = values[0] / 0.92

    return values

# TODO: make sure this back-up function always matches the version in optatmo_psf.py from PIFF proper
def measure_error_orthogonal(star, logger=None):
    """Measure the shape of a star using the HSM algorithm.

    Goes up to third moments plus orthogonal radial moments up to eighth moments.

    :param star:                Star we want to measure
    :param logger:              A logger object for logging debug info

    :returns:   Shape Error in unnormalized basis. Goes up to third moments plus orthogonal
                radial moments up to eighth moments.  to fourth moments.
    """

    # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
    #          sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2,
    #          sigma_orth4, sigma_orth6, sigma_orth8
    values = piff.util.calculate_moments(star, logger=logger, third_order=True, radial=True, errors=True)
    errors = np.array(values[13:])

    if True:
        hsm = piff.util.hsm(star)
        errors[0] *= (hsm[0] * star.data.pixel_area * star.data.weight.array.mean())**2
        errors[3:] *= 4

    return np.sqrt(errors)

# function for measuring the shape and snr of several stars
def measure_star_shape(psf, stars, model_stars, delete_list, is_optatmo, logger=None):
    logger = galsim.config.LoggerWrapper(logger)
    shapes = []
    #for star in model_stars:
    #    if star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
    #        raise AttributeError('model_star that is not of the star class found!')
    for i in range(len(stars)):
        try:
            if i % 100 == 0:
                logger.debug('Measuring shape of star {0} of {1}'.format(i, len(stars)))
            star = stars[i]
            model_star = model_stars[i]
            # returns a pandas series
            if is_optatmo:
                flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, orth4, orth6, orth8 = psf.measure_shape_orthogonal(star)
                sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_orth4, sigma_orth6, sigma_orth8 = psf.measure_error_orthogonal(star)
                #if model_star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
                #    raise AttributeError('model_star that is not of the star class found!')
                model_flux, model_u0, model_v0, model_e0, model_e1, model_e2, model_zeta1, model_zeta2, model_delta1, model_delta2, model_orth4, model_orth6, model_orth8 = psf.measure_shape_orthogonal(model_star)
            else:
                flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, orth4, orth6, orth8 = measure_shape_orthogonal(star)
                sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_orth4, sigma_orth6, sigma_orth8 = measure_error_orthogonal(star)
                #if model_star == None or str(type(star)) == "<class 'NoneType'>" or str(type(star)) != "<class 'piff.star.Star'>":
                #    raise AttributeError('model_star that is not of the star class found!')
                model_flux, model_u0, model_v0, model_e0, model_e1, model_e2, model_zeta1, model_zeta2, model_delta1, model_delta2, model_orth4, model_orth6, model_orth8 = measure_shape_orthogonal(model_star)

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
        except:
            delete_list.append(i)
    shapes = pd.DataFrame(shapes)
    return shapes, delete_list

# function for making moment residual across-the-focal-plane plots (aka "the color plots") 
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


########################################################################
########################################################################


# function for doing the fit.
def fit_psf(directory, config_file_name, print_log, meanify_file_path='', fit_interp_only=False, opt_only=False, no_opt=False, no_interp=False, no_final_graphs = False, final_graphs_only = False):
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

    import time

    if (do_meanify or fit_interp_only or final_graphs_only) and is_optatmo:
        if not final_graphs_only:
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
            if is_optatmo:
                logger.info('CPU time before optical fit: {0}'.format(time.clock()))
                np.random.seed(12345)
                psf.fit(stars, wcs, pointing, logger=logger)
                #test_stars = psf.test_stars
                logger.info('CPU time after optical fit: {0}'.format(time.clock()))
            else:
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
            logger.info('Fitted PSF!')




            # do stuff between optical fit and individual star fit
            if is_optatmo:
                # a round of outlier rejection using the pull from the moments for the optical fit
                for s, stars in enumerate([psf.stars, psf.test_stars, psf.fit_optics_stars]):
                    if s == 0:
                        logger.info("now prepping pull cuts for psf.stars; in other words the train stars")
                    if s == 1:
                        logger.info("now prepping pull cuts for psf.test_stars")
                    if s == 2:
                        logger.info("now prepping pull cuts for psf.fit_optics_stars; in other words the "
                                    "subset train stars specifically used in the optical fit")

                    # prep lists for shapes, model shapes, and shape errors
                    data_shapes_all_stars = []
                    data_errors_all_stars = []
                    model_shapes_all_stars = []

                   # Draw model stars in order to measure their shapes
                    logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
                    delete_list = []
                    model_stars = psf.drawStarList(stars)
                    logger.info('Preliminarily drew model stars. Will now identify stars failed to be drawn to throw them out.')
                    for star_i, star in enumerate(model_stars):
                        if star == None:
                            delete_list.append(star_i)
                    logger.info("delete_list: {0}".format(delete_list))
                    stars = np.delete(stars, delete_list).tolist()
                    model_stars = np.delete(model_stars, delete_list).tolist()
                    logger.info('Finished drawing {0} {1} model stars. Some may have failed to be drawn and were thrown out.'.format(len(stars), label))

                    # make orthogonal moments' optical pull histograms
                    for star_i, star, model_star in zip(list(range(0,len(stars))), stars, model_stars):
                        try:
                            data_shape = psf.measure_shape_orthogonal(star)
                            data_error = psf.measure_error_orthogonal(star)
                            model_shape = psf.measure_shape_orthogonal(model_star)
                            data_shapes_all_stars.append(data_shape)
                            data_errors_all_stars.append(data_error)
                            model_shapes_all_stars.append(model_shape)
                        except:
                            logger.info('failed to do orthogonal moments analysis for pull cuts for star {0}'.format(star_i))
                            data_shapes_all_stars.append(np.full(13,1000.0)) # throw out stars where pull cannot be measured
                            data_errors_all_stars.append(np.full(13,1.0))
                            model_shapes_all_stars.append(np.full(13,2.0))
                    data_shapes_all_stars = np.array(data_shapes_all_stars)[:,3:]
                    data_errors_all_stars = np.array(data_errors_all_stars)[:,3:]
                    model_shapes_all_stars = np.array(model_shapes_all_stars)[:,3:]
                    pull_all_stars_optical = ((data_shapes_all_stars - model_shapes_all_stars) /
                                        data_errors_all_stars)
                    # pull is (data-model)/error
                    logger.debug("data_shapes_all_stars: {0}".format(data_shapes_all_stars))
                    logger.debug("model_shapes_all_stars: {0}".format(model_shapes_all_stars))
                    logger.debug("data_errors_all_stars: {0}".format(data_errors_all_stars))
                    logger.debug("pull_all_stars_optical: {0}".format(pull_all_stars_optical))
                    med = np.nanmedian(pull_all_stars_optical, axis=0)
                    mad = np.nanmedian(np.abs(pull_all_stars_optical - med[None]), axis=0)
                    madx = np.abs(pull_all_stars_optical - med[None])
                    # all stars with pull more than 4 sigma equivalent MAD away from the median pull are thrown out
                    conds_pull_mad = (np.all(madx <= 1.48 * 4 * mad, axis=1))
                    conds_pull_mad_e0 = (madx[:,0] <= 1.48 * 4 * mad[0])
                    conds_pull_mad_e1 = (madx[:,1] <= 1.48 * 4 * mad[1])
                    conds_pull_mad_e2 = (madx[:,2] <= 1.48 * 4 * mad[2])
                    if s == 0:
                        psf.stars = np.array(psf.stars)[conds_pull_mad].tolist()
                    if s == 1:
                        psf.test_stars = np.array(psf.test_stars)[conds_pull_mad].tolist()
                    if s == 2:
                        number_of_outliers_optical = np.array(
                            [len(psf.fit_optics_stars) - np.sum(conds_pull_mad_e0),
                             len(psf.fit_optics_stars) - np.sum(conds_pull_mad_e1),
                             len(psf.fit_optics_stars) - np.sum(conds_pull_mad_e2)])
                        number_of_stars_pre_cut_optical = len(psf.fit_optics_stars)
                        psf.fit_optics_stars = np.array(psf.fit_optics_stars)[conds_pull_mad].tolist()
                        number_of_stars_post_cut_optical = len(psf.fit_optics_stars)
                        logger.info('number of stars pre cut optical: {0}'.format(number_of_stars_pre_cut_optical))
                        logger.info('number of stars post cut optical: {0}'.format(number_of_stars_post_cut_optical))
                        pull_mean_optical = np.nanmean(pull_all_stars_optical[:,:3], axis=0)
                        # the mean pull (only second moments) for stars used in the fit is later used to
                        # find outliers among exposures
                        pull_rms_optical = np.sqrt(np.nanmean(np.square(pull_all_stars_optical[:,:3]),axis=0))
                        logger.info('Preparing to create optical pull histograms')
                        for m, moment in enumerate(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]):
                            plt.figure()
                            plt.hist(pull_all_stars_optical[:,m], bins=20)
                            plt.savefig("{0}/{1}_optical_pull_hist.png".format(directory, moment))




                # save optical pull cuts (some of these will be used in doing cuts on exposures) and make chisq histogram
                logger.info('Preparing to create chisq histogram')
                chisq_all_stars_optical = psf.chisq_all_stars_optical
                plt.figure()
                plt.hist(chisq_all_stars_optical, bins=20)
                plt.savefig("{0}/optical_chisq_hist.png".format(directory))
                logger.info('Preparing to save outlier npy files')
                np.save("{0}/number_of_outliers_optical.npy".format(directory),np.array(number_of_outliers_optical))
                np.save("{0}/pull_mean_optical.npy".format(directory),np.array(pull_mean_optical))    
                np.save("{0}/pull_rms_optical.npy".format(directory),np.array(pull_rms_optical))




                # make optical trace plots and plot redchi across opt fit iterations
                total_redchi_across_iterations = psf.total_redchi_across_iterations
                iterations_indices = range(0,len(total_redchi_across_iterations))
                plt.figure()
                plt.scatter(x=iterations_indices,y=total_redchi_across_iterations)
                plt.savefig("{0}/total_optical_redchi_across_iterations.png".format(directory))
                optical_fit_params_across_iterations = np.array(psf.optical_fit_params_across_iterations)
                optical_fit_keys = psf.optical_fit_keys
                for o, optical_fit_key in enumerate(optical_fit_keys):
                    plt.figure()
                    plt.scatter(x=list(range(0,len(optical_fit_params_across_iterations))),y=optical_fit_params_across_iterations[:,o])
                    plt.savefig("{0}/{1}_across_iterations.png".format(directory, optical_fit_key))
                logger.info('Note: Optics fit took {0} function evals to finish'.format(len(optical_fit_params_across_iterations)))
                np.save("{0}/full_optical_chisq_vector.npy".format(directory), np.array(psf.final_optical_chi))
                logger.info('Finished saving the full optical chisq vector is')
                logger.info('Finished creating optical param trace plots and (if appropriate) optical pull histograms and chisq histogram')




                # save optatmo_psf_kwargs to an npy file for easy access to optics fit params
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




                # Create graphs and h5 files for both train and test stars for the opt_only case (also save the stars and the PSF itself)
                logger.info('Preparing to create graphs and h5 files for both train and test stars for the opt_only case (also preparing to save the stars and the PSF itself)')
                for stars, label in zip([psf.stars, psf.test_stars], ['train', 'test']):

                    # Draw model stars in order to measure their shapes
                    logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
                    delete_list = []
                    model_stars = psf.drawStarList(stars)
                    logger.info('Preliminarily drew model stars. Will now identify stars failed to be drawn to throw them out.')
                    for star_i, star in enumerate(model_stars):
                        if star == None:
                            delete_list.append(star_i)
                    logger.info("delete_list: {0}".format(delete_list))
                    stars = np.delete(stars, delete_list).tolist()
                    model_stars = np.delete(model_stars, delete_list).tolist()
                    logger.info('Finished drawing {0} {1} model stars. Some may have failed to be drawn and were thrown out.'.format(len(stars), label))

                    # Measure shapes of both stars and model stars
                    logger.info('Preparing to extract and get params (& fit params if train stars) graphs for {0} stars'.format(label))
                    logger.info('Using measure_star_shape()')
                    delete_list = []
                    shapes, delete_list = measure_star_shape(psf, stars, model_stars, delete_list, is_optatmo, logger=logger)
                    logger.info("delete_list: {0}".format(delete_list))
                    stars = np.delete(stars, delete_list).tolist()
                    model_stars = np.delete(model_stars, delete_list).tolist()
                    if label == "train":
                        psf.stars = np.delete(psf.stars, delete_list).tolist() 
                    if label == "test":
                        psf.test_stars = np.delete(psf.test_stars, delete_list).tolist()                                              

                    # Set up param keys
                    logger.info('Preparing to set up param keys')
                    param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2']
                    if psf.atmosphere_model == 'vonkarman':
                        param_keys += ['opt_L0']
                    param_keys += ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
                    logger.info('Finished setting up param keys')

                    # Get params (& fit params if train stars) to be used later in graphs
                    logger.info('Preparing to get params for {0} stars'.format(label))
                    if label == 'train':
                        # If train, get params (from getParamsList())
                        logger.info('Preparing to get train params (from getParamsList())')
                        params = psf.getParamsList(stars)
                        for i in range(params.shape[1]):
                            shapes[param_keys[i]] = params[:, i]
                        logger.info('Finished getting train params (from getParamsList())')

                    elif label == 'test':
                        # If test, get params (from getParamsList())
                        logger.info('Preparing to get test params (from getParamsList())')
                        params = psf.getParamsList(stars)
                        for i in range(params.shape[1]):
                            shapes[param_keys[i]] = params[:, i]
                        logger.info('Finished getting test params (from getParamsList())')
                    logger.info('Finished getting params (& fit params if train stars) for {0} stars'.format(label))

                    # Create h5 files for stars using shapes, model shapes, data-model shapes, and params (& fit params if train stars)
                    logger.info('Preparing to create h5 files for {0} stars using shapes, model shapes, data-model shapes, and params'.format(label))
                    shapes.to_hdf('{0}/shapes_{1}_{2}_opt_only.h5'.format(directory, label, piff_name), 'data', mode='w')
                    logger.info('Finished creating h5 files for {0} stars using shapes, model shapes, data-model shapes, and params'.format(label))

                    # Create oned moment histograms
                    make_oned_hists_pdf(shapes, directory, label, "{0}_opt_only".format(piff_name))

                    # Create plot_2dhist_shapes graphs
                    logger.info('Preparing to create plot_2dhist_shapes graphs for {0} stars'.format(label))
                    # Make plot_2dhist_shapes graphs
                    shape_keys = ['e0', 'e1', 'e2', 'zeta1', 'zeta2', 'delta1', 'delta2', 'orth4', 'orth6', 'orth8']
                    shape_plot_keys = []
                    for key in shape_keys:
                        shape_plot_keys.append(['data_' + key, 'model_' + key, 'd' + key])
                    fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
                    # Save plot_2dhist_shapes graphs
                    fig.savefig('{0}/plot_2dhist_shapes_{1}_{2}_opt_only.pdf'.format(directory, label, piff_name))
                    logger.info('Finished creating plot_2dhist_shapes graphs for {0} stars'.format(label))

                # Save a copy of the stars used in the optical fit that were not thrown out after as outliers
                logger.info('Preparing to copy stars_fit_optics')
                stars_fit_optics = copy.deepcopy(psf.fit_optics_stars)
                np.save("{0}/stars_fit_optics.npy".format(directory),np.array(stars_fit_optics))

                # Save a copy of the train stars
                logger.info('Preparing to copy stars_train_psf_optatmo_opt_only')
                stars_train_copy = copy.deepcopy(psf.stars)
                np.save("{0}/stars_train_psf_optatmo_opt_only.npy".format(directory),np.array(stars_train_copy))

                # Save a copy of the test stars
                logger.info('Preparing to copy stars_test_psf_optatmo_opt_only')
                stars_test_copy = copy.deepcopy(psf.test_stars)
                np.save("{0}/stars_test_psf_optatmo_opt_only.npy".format(directory),np.array(stars_test_copy))


                # Save optical psf (with train and test stars) in case you just want to stop after the optical fit and pick up where you left off later
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
                logger.info('Finished creating graphs and h5 files for both train and test stars for the opt_only case (also preparing to save the stars and the PSF itself)')




        # if no_opt and is_optatmo, simply load the stuff from the previous-finished optical fit and start the fit run from there
        else:
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




        # stop here right after the optical fit if is_optatmo and you request to do only the optical fit
        if opt_only and is_optatmo:
            sys.exit()




########################################################################
########################################################################

        # Do Individual Atmo Star Fit and Associated Star Cuts
        # fit atmosphere parameters (for train stars)
        if is_optatmo:
            logger.info('Fitting PSF atmosphere parameters (with train stars)')
            logger.info('getting param info for {0} train stars'.format(len(psf.stars)))
            params = psf.getParamsList(psf.stars)
            psf._enable_atmosphere = False
            new_stars = []
            cpu_time_across_stars = []
            number_of_successful_fits = 0
            for star_i, star in zip(range(len(psf.stars)), psf.stars):
                if star_i == len(psf.stars) - 1 and number_of_successful_fits == 0:
                    logger.info('Failed individual atmo star fit for all stars but the last! Will now attempt to do the last without a try-except statement.')
                    start_time = time.clock()
                    if star_i % 100 == 0:
                        logger.info('Fitting star {0} of {1}'.format(star_i, len(psf.stars)))
                    model_fitted_star = psf.fit_model(star, params=params[star_i], logger=logger)
                    new_stars.append(model_fitted_star)
                    logger.info('chisq for star {0}: {1}'.format(star_i, model_fitted_star.fit.chisq))
                    logger.info('dof for star {0}: {1}'.format(star_i, model_fitted_star.fit.dof))
                    logger.info('atmo_size for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[0]))
                    logger.info('atmo_g1 for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[1]))
                    logger.info('atmo_g2 for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[2]))
                    logger.info('clock time for star {0}: {1}'.format(star_i, time.clock() - start_time))
                    cpu_time_across_stars.append(time.clock() - start_time)
                    number_of_successful_fits = number_of_successful_fits + 1

                else:
                    start_time = time.clock()
                    if star_i % 100 == 0:
                        logger.info('Fitting star {0} of {1}'.format(star_i, len(psf.stars)))
                    try:
                        model_fitted_star = psf.fit_model(star, params=params[star_i], logger=logger)
                        new_stars.append(model_fitted_star)
                        logger.info('chisq for star {0}: {1}'.format(star_i, model_fitted_star.fit.chisq))
                        logger.info('dof for star {0}: {1}'.format(star_i, model_fitted_star.fit.dof))
                        logger.info('atmo_size for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[0]))
                        logger.info('atmo_g1 for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[1]))
                        logger.info('atmo_g2 for star {0}: {1}'.format(star_i, model_fitted_star.fit.params[2]))
                        logger.info('clock time for star {0}: {1}'.format(star_i, time.clock() - start_time))
                        cpu_time_across_stars.append(time.clock() - start_time)
                        number_of_successful_fits = number_of_successful_fits + 1
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:
                        logger.warning('{0}'.format(str(e)))
                        logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
            plt.figure()
            plt.scatter(x=list(range(0,number_of_successful_fits)),y=cpu_time_across_stars, alpha=0.1, s=0.2)
            plt.savefig("{0}/cpu_time_across_stars.png".format(directory))
            psf.stars = new_stars




            # go through the motions of fitting atmosphere parameters for test stars, so we can throw out test stars for which this would fail
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
            if len(mad_stars) != len(psf.stars):
                logger.info('Stripped (train) stars from {0} to {1} based on 5sig MAD cut'.format(len(stripped_stars), len(mad_stars)))
            psf.stars = mad_stars




            # cut (test) stars based on 5sig MAD cut
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
            if len(mad_stars) != len(psf.test_stars):
                logger.info('Stripped (test) stars from {0} to {1} based on 5sig MAD cut'.format(len(stripped_stars), len(mad_stars)))
            psf.test_stars = mad_stars




            # one extra round of outlier rejection using the pull from the moments (only up to third moments) after the individual star fit for the atmospheric parameters and the atmospheric parameter MAD cuts
            for s, stars in enumerate([psf.stars, psf.test_stars]):
                if s == 0:
                    logger.info('Number of Train Stars Before Atmo Pull Cuts: {0}'.format(len(stars)))
                if s == 1:
                    logger.info('Number of Test Stars Before Atmo Pull Cuts: {0}'.format(len(stars)))

                # prep lists for shapes, model shapes, and shape errors
                data_shapes_all_stars = []
                data_errors_all_stars = []
                model_shapes_all_stars = []


               # Draw model stars in order to measure their shapes
                logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
                delete_list = []
                model_stars = psf.drawStarList(stars)
                logger.info('Preliminarily drew model stars. Will now identify stars failed to be drawn to throw them out.')
                for star_i, star in enumerate(model_stars):
                    if star == None:
                        delete_list.append(star_i)
                logger.info("delete_list: {0}".format(delete_list))
                stars = np.delete(stars, delete_list).tolist()
                model_stars = np.delete(model_stars, delete_list).tolist()
                logger.info('Finished drawing {0} {1} model stars. Some may have failed to be drawn and were thrown out.'.format(len(stars), label))

                # make orthogonal moments' optical pull histograms
                for star_i, star, model_star in zip(list(range(0,len(stars))), stars, model_stars):
                    try:
                        data_shape = psf.measure_shape_orthogonal_moments(star)
                        data_error = psf.measure_error_orthogonal_moments(star)
                        model_shape = psf.measure_shape_orthogonal_moments(model_star)
                        data_shapes_all_stars.append(data_shape)
                        data_errors_all_stars.append(data_error)
                        model_shapes_all_stars.append(model_shape)
                    except:
                        logger.info('failed to do orthogonal moments analysis for pull cuts for star {0}'.format(star_i))
                        data_shapes_all_stars.append(np.full(13,1000.0)) # throw out stars where pull cannot be measured
                        data_errors_all_stars.append(np.full(13,1.0))
                        model_shapes_all_stars.append(np.full(13,2.0))
                data_shapes_all_stars = np.array(data_shapes_all_stars)[:,3:]
                data_errors_all_stars = np.array(data_errors_all_stars)[:,3:]
                model_shapes_all_stars = np.array(model_shapes_all_stars)[:,3:]
                pull_all_stars = (data_shapes_all_stars - model_shapes_all_stars) / data_errors_all_stars # pull is (data-model)/error
                med = np.nanmedian(pull_all_stars, axis=0)
                mad = np.nanmedian(np.abs(pull_all_stars - med[None]), axis=0)
                madx = np.abs(pull_all_stars - med[None])
                conds_pull_mad = (np.all(madx <= 1.48 * 4 * mad, axis=1)) # all stars with pull more than 4 sigma equivalent MAD away from the median pull are thrown out
                if s == 0:
                    psf.stars = np.array(psf.stars)[conds_pull_mad].tolist()
                    logger.info('Number of Train Stars After Atmo Pull Cuts: {0}'.format(len(psf.stars)))
                if s == 1:
                    psf.test_stars = np.array(psf.test_stars)[conds_pull_mad].tolist()
                    logger.info('Number of Test Stars After Atmo Pull Cuts: {0}'.format(len(psf.test_stars)))




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




        # stop here right before the atmospheric interpolation if is_optatmo and you request to not do the atmospheric interpolation
        if no_interp and is_optatmo:
            sys.exit()




########################################################################
########################################################################

    # For all atmospheric interpolations as specified in the interp_keys from the config file (in addition to the "noatmo" interp), do the atmospheric interpolation, make a copy of the PSF that has it, save that PSF, and save the data stars (test and train)
    shape_keys = ['e0', 'e1', 'e2', 'zeta1', 'zeta2', 'delta1', 'delta2', 'orth4', 'orth6', 'orth8']
    shape_plot_keys = []
    for key in shape_keys:
        shape_plot_keys.append(['data_' + key, 'model_' + key, 'd' + key])
    if is_optatmo:
        interps = config.pop('interps')
        interp_keys = list(interps.keys())
        if False:
        #if not (do_meanify or fit_interp_only):
            # do noatmo only when we do not have meanify
            interp_keys = interp_keys + ['noatmo']
        for interp_key in interp_keys:
            piff_name = '{0}_{1}'.format(config_file_name, interp_key)
            if not final_graphs_only:
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

                # Save the PSF, and the data stars (test and train)
                out_path = '{0}/{1}.piff'.format(directory, piff_name)
                psf.write(out_path, logger=logger)
                test_stars = psf.test_stars
                write_stars(test_stars, out_path)
            else:
                # retrieve PSF with atmo-interpolation and stars
                logger.info('Retrieving optatmo interpolate for {0}'.format(interp_key))
                if interp_key == 'noatmo':
                    psf.atmo_interp = None
                    psf._enable_atmosphere = False
                else:
                    if do_meanify:
                        piff_name += '_meanified'

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



########################################################################
########################################################################

            # Create graphs, h5 files, rho statistics (and other Stats Outputs) for both train and test stars  (also save a copy of the stars as npy files that includes the atmo params from this interpolation, includes the refluxing for the test stars, and excludes all stars that could not be used for graph-making).
            # Create graphs, h5 files, rho statistics (and other Stats Outputs) for both train and test stars.
            logger.info('Evaluating {0}'.format(piff_name))
            logger.info('Preparing to create graphs, h5 files, rho statistics (and other Stats Outputs) for both train and test stars.')
            for stars, label in zip([psf.stars, psf.test_stars], ['train', 'test']):

                # Draw model stars in order to measure their shapes
                logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
                delete_list = []
                stars_copy = copy.deepcopy(stars)   
                model_stars, stars = psf.drawStarList(stars, return_stars_with_atmo_params_from_psf=True)
                logger.info('Preliminarily drew model stars. Will now identify stars failed to be drawn to throw them out.')
                for star_i, star in enumerate(model_stars):
                    if star == None:
                        delete_list.append(star_i)
                logger.info("delete_list: {0}".format(delete_list))
                stars = np.delete(stars, delete_list).tolist()
                model_stars = np.delete(model_stars, delete_list).tolist()
                stars_copy = np.delete(stars_copy, delete_list).tolist() 
                if label == "train":
                    psf.stars = np.delete(psf.stars, delete_list).tolist() 
                if label == "test":
                    psf.test_stars = np.delete(psf.test_stars, delete_list).tolist() 
                logger.info('Finished drawing {0} {1} model stars. Some may have failed to be drawn and were thrown out.'.format(len(stars), label))

                # Measure shapes of both stars and model stars
                logger.info('Preparing to extract and get params (& fit params if train stars) graphs for {0} stars'.format(label))
                logger.info('Using measure_star_shape()')
                delete_list = []
                shapes, delete_list = measure_star_shape(psf, stars, model_stars, delete_list, is_optatmo, logger=logger)
                logger.info("delete_list: {0}".format(delete_list))
                stars = np.delete(stars, delete_list).tolist()
                model_stars = np.delete(model_stars, delete_list).tolist()
                if label == "train":
                    psf.stars = np.delete(psf.stars, delete_list).tolist() 
                if label == "test":
                    psf.test_stars = np.delete(psf.test_stars, delete_list).tolist() 

                # Set up param keys
                logger.info('Preparing to set up param keys')
                param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2']
                if psf.atmosphere_model == 'vonkarman':
                    param_keys += ['opt_L0']
                param_keys += ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
                logger.info('Finished setting up param keys')

                # Get params (& fit params if train stars) to be used later in graphs
                logger.info('Preparing to get params (& fit params if train stars) for {0} stars'.format(label))
                if label == 'train':
                    # If train, get params (from getParamsList()) & fit params that were found in the individual atmo star fit
                    logger.info('Preparing to get train fit params that were found in the individual atmo star fit')
                    params = np.array([star.fit.params for star in stars_copy])
                    params_var = np.array([star.fit.params_var for star in stars_copy])
                    for i in range(params.shape[1]):
                        shapes['{0}_fit'.format(param_keys[i])] = params[:, i]
                        shapes['{0}_var'.format(param_keys[i])] = params_var[:, i]
                    logger.info('Finished getting train fit params that were found in the individual atmo star fit')
                    logger.info('Preparing to get train params (from getParamsList())')
                    params = psf.getParamsList(stars, trust_atmo_params_stored_in_stars=True)
                    for i in range(params.shape[1]):
                        shapes[param_keys[i]] = params[:, i]
                    logger.info('Finished getting train params (from getParamsList())')

                elif label == 'test':
                    # If test, get params (from getParamsList())
                    logger.info('Preparing to get test params (from getParamsList())')
                    params = psf.getParamsList(stars, trust_atmo_params_stored_in_stars=True)
                    for i in range(params.shape[1]):
                        shapes[param_keys[i]] = params[:, i]
                    logger.info('Finished getting test params (from getParamsList())')
                logger.info('Finished getting params (& fit params if train stars) for {0} stars'.format(label))

                # Create h5 files for stars using shapes, model shapes, data-model shapes, and params (& fit params if train stars)
                logger.info('Preparing to create h5 files for {0} stars using shapes, model shapes, data-model shapes, and params (& fit params if train stars)'.format(label))
                shapes.to_hdf('{0}/shapes_{1}_{2}.h5'.format(directory, label, piff_name), 'data', mode='w')
                logger.info('Finished creating h5 files for {0} stars using shapes, model shapes, data-model shapes, and params (& fit params if train stars)'.format(label))

                # Create oned moment histograms
                make_oned_hists_pdf(shapes, directory, label, piff_name)

                # Create plot_2dhist_shapes graphs
                logger.info('Preparing to create plot_2dhist_shapes graphs for {0} stars'.format(label))
                # Make plot_2dhist_shapes graphs
                fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
                # Save plot_2dhist_shapes graphs
                fig.savefig('{0}/plot_2dhist_shapes_{1}_{2}.pdf'.format(directory, label, piff_name))
                logger.info('Finished creating plot_2dhist_shapes graphs for {0} stars'.format(label))

                # Create plot_2dhist_params (& plot_2dhist_fit_params if train stars) graphs
                logger.info('Preparing to create plot_2dhist_params (& plot_2dhist_fit_params if train stars) graphs for {0} stars'.format(label))
                # Make plot_2dhist_params graphs
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
                # Save plot_2dhist_params graphs
                fig.savefig('{0}/plot_2dhist_params_{1}_{2}.pdf'.format(directory, label, piff_name))
                # If train, create plot_2dhist_fit_params graphs
                if label == 'train':
                    # Make plot_2dhist_fit_params graphs
                    fig, axs = plot_2dhist_shapes(shapes, [[key + '_fit' for key in kp] for kp in plot_keys], diff_mode=False)
                    # Save plot_2dhist_fit_params graphs
                    fig.savefig('{0}/plot_2dhist_fit_params_{1}_{2}.pdf'.format(directory, label, piff_name))
                logger.info('Finished creating plot 2dhist params (& plot_2dhist_fit_params if train stars) graphs for {0} stars'.format(label))

                # If test, reflux (re-find the fluxes and centers) the stars in preparation for writing the Stats Outputs (writing these requires the use of the stars)
                if label == 'test':
                    logger.info('Preparing to fit the fluxes and centers of {0} test stars (this section is skipped if train stars)'.format(len(stars)))
                    new_stars = []
                    star_i = 0
                    for star, param in zip(stars, params):
                        try:
                            new_star = psf.reflux(star, param, logger=logger)
                            new_stars.append(new_star) 
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as e:
                            logger.warning('{0}'.format(str(e)))
                            logger.warning('Warning! Failed to fit flux and center for star {0}. Ignoring star'.format(star_i))
                        star_i = star_i + 1
                    stars = new_stars
                    logger.info('Finished fitting the fluxes and centers of {0} test stars (this section is skipped if train stars)'.format(len(stars)))

                if label == "train":
                    psf.stars = stars
                if label == "test":
                    psf.test_stars = stars    
            logger.info('Finished creating graphs, h5 files, rho statistics (and other Stats Outputs) for both train and test stars.')



            # Save a copy of the data stars (test and train) to npy files; note that unlike the copy of these stars saved in the PSF, these have the atmo params from whatever atmospheric interpolation method being used (rather than taking the atmo params from the individual atmo star fit for train stars, for example); another thing to note is that stars saved here exclude stars that could not be used for graph-making; also note the test stars here have been refluxed
            logger.info('Preparing to save {0} stars to npy files'.format(label))
            np.save("{0}/stars_{1}_{2}.npy".format(directory, label, piff_name),np.array(stars))
            logger.info('Finished saving {0} stars to npy files'.format(label))


########################################################################
########################################################################

    # For pixelgrid, create graphs, h5 files, rho statistics (and other Stats Outputs) for both train and test stars (also save a copy of the stars as npy files).
    else:
        logger.info('Evaluating {0}'.format(piff_name))

        for stars, label in zip([psf.stars, test_stars], ['train', 'test']):     
        
            # draw model stars in order to measure their shapes
            logger.info('Preparing to draw {0} {1} model stars'.format(len(stars), label))
            delete_list = []
            model_stars = psf.drawStarList(stars)
            for star_i, star in enumerate(stars):
                if star == None:
                    delete_list.append(star_i)
            stars = np.delete(stars, delete_list).tolist()
            model_stars = np.delete(model_stars, delete_list).tolist()
            if label == "train":
                psf.stars = np.delete(psf.stars, delete_list).tolist() 
            if label == "test":
                test_stars = np.delete(test_stars, delete_list).tolist()  
            logger.info('Finished drawing {0} {1} model stars. Some failed to be drawn and were thrown out.'.format(len(stars), label))

            # measure shapes
            delete_list = []
            shapes, delete_list = measure_star_shape(psf, stars, model_stars, delete_list, logger=logger)
            stars = np.delete(stars, delete_list).tolist()
            model_stars = np.delete(model_stars, delete_list).tolist()
            if label == "train":
                psf.stars = np.delete(psf.stars, delete_list).tolist() 
            if label == "test":
                test_stars = np.delete(test_stars, delete_list).tolist()  

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

            # Save a copy of the data stars (test and train) to npy files; note that stars saved here exclude stars that could not be used for graph-making
            logger.info('Preparing to save {0} stars to npy files'.format(label))
            np.save("{0}/stars_{1}_{2}.npy".format(directory, label, piff_name),np.array(stars))
            logger.info('Finished saving {0} stars to npy files'.format(label))   

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', action='store', dest='directory',
                        default='.',
                        help='directory of the run. Default to current directory')
    parser.add_argument('--print_log', action='store_true', dest='print_log',
                        help='print logging instead of save')
    parser.add_argument('--fit_interp_only', action='store_true', dest='fit_interp_only',
                        help='start fit after (previously done) individual atmo star fit')
    parser.add_argument('--config_file_name', action='store', dest='config_file_name',
                        help='name of the config file (without .yaml)')
    parser.add_argument('--meanify_file_path', action='store', dest='meanify_file_path',
                        default='',
                        help='path to meanify file. If not specified, or if not using optatmo, then ignored')
    parser.add_argument('--opt_only', action='store_true', dest='opt_only',
                        help='only do the optical fit')
    parser.add_argument('--no_opt', action='store_true', dest='no_opt',
                        help='start fit after (previously done) optical fit')
    parser.add_argument('--no_interp', action='store_true', dest='no_interp',
                        help='stop fit right before the atmo interpolation')
    parser.add_argument('--no_final_graphs', action='store_true', dest='no_final_graphs',
                        help='stop fit right before the final graphs are made')
    parser.add_argument('--final_graphs_only', action='store_true', dest='final_graphs_only',
                        help='(having previously done everything else), do just the final graphs')
    options = parser.parse_args()
    kwargs = vars(options)
    fit_psf(**kwargs)
