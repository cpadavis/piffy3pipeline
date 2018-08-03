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
import sys

#from piff.stats import Stats
#from piff.star import Star, StarFit
from piff.util import hsm_error, hsm_higher_order, measure_snr
#from piff.star_stats import fit_star_alternate, _fit_residual_alternate

def fit_star_alternate(star, psf, logger=None):
    """Adjust star.fit.flux and star.fit.center

    :param star:        Star we want to adjust
    :param psf:         PSF with which we adjust
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: Star with modified fit and center
    """
    import lmfit
    # create lmfit
    lmparams = lmfit.Parameters()
    # put in initial guesses for flux, du, dv if they exist
    flux = star.fit.flux
    if flux == 1.0:
        # provide a reasonable starting guess
        flux = star.image.array.sum()
    du, dv = star.fit.center
    # Order of params is important!
    lmparams.add('flux', value=flux, vary=True, min=0.0)
    lmparams.add('du', value=du, vary=True, min=-1, max=1)
    lmparams.add('dv', value=dv, vary=True, min=-1, max=1)

    # run lmfit
    results = lmfit.minimize(_fit_residual_alternate, lmparams,
                             args=(star, psf, logger,),
                             method='leastsq', epsfcn=1e-8,
                             maxfev=500)

    # report results
    #logger.debug('Adjusted Star Fit Results:')
    #logger.debug(lmfit.fit_report(results))
    #print(lmfit.fit_report(results))
    #import time
    #time.sleep(10)

    # create new star with new fit
    flux = results.params['flux'].value
    du = results.params['du'].value
    dv = results.params['dv'].value
    center = (du, dv)
    # also update the chisq, but keep the rest of the parameters from model fit
    chisq = results.chisqr
    fit = piff.star.StarFit(star.fit.params, params_var=star.fit.params_var,
            flux=flux, center=center, chisq=chisq, dof=star.fit.dof,
            alpha=star.fit.alpha, beta=star.fit.beta,
            worst_chisq=star.fit.worst_chisq)
    star_fit = piff.star.Star(star.data, fit)

    return star_fit

def _fit_residual_alternate(lmparams, star, psf, logger=None):
    # modify star's fit values
    flux, du, dv = lmparams.valuesdict().values()
    star.fit.flux = flux
    star.fit.center = (du, dv)

    # draw star
    image_model = psf.drawStar(star).image

    # get chi
    image, weight, image_pos = star.data.getImage()
    chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()

    return chi

def convert_two_d_plot_to_radial_plot(two_d_plot):
    stamp_size = len(two_d_plot[0])
    max_r = np.sqrt(np.square((stamp_size-1.0)/2.0)+np.square((stamp_size-1.0)/2.0))
    number_of_radial_bins = np.int(np.floor(max_r) + 1)
    x = range(0,number_of_radial_bins)
    y = np.zeros(number_of_radial_bins)
    y_bin_entries = np.zeros(number_of_radial_bins,int)
    r_flux_pairs = []

    for i in range(0,stamp_size):
        for j in range(0,stamp_size):
            r = np.sqrt(np.square((stamp_size-1.0)/2.0-i)+np.square((stamp_size-1.0)/2.0-j))
            r_flux_pairs.append([np.int(np.floor(r)),two_d_plot[i][j]])
    for r_flux_pair in r_flux_pairs:
        y_bin_number = r_flux_pair[0]
        y[y_bin_number] = y[y_bin_number] + r_flux_pair[1]
        y_bin_entries[y_bin_number] = y_bin_entries[y_bin_number] + 1
    for y_index in range(0,number_of_radial_bins):
        if y_bin_entries[y_index] > 0:
            y[y_index] = y[y_index]/y_bin_entries[y_index]
    return x,y

def make_star_residual_plots():
    directory = "{0}/00{1}".format(core_directory, exposure)
    print("directory: {0}".format(directory))
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)
    # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"

    stars_test = np.load("{0}/stars_test.npy".format(directory))
    stars_train = np.load("{0}/stars_train.npy".format(directory))
    psf = piff.read("{0}/psf_{1}.piff".format(directory, psf_type))   

    is_optatmo=True
    if psf_type.split("_")[0]!="optatmo":
	is_optatmo=False
    print(is_optatmo)

    for star_test in stars_test:
	star_test.fit.params = None

    if is_optatmo==True:
        params = psf.getParamsList(stars_test)
        pre_drawn_stars = [psf.fit_model(star_test, param, vary_shape=False)[0] for star_test, param in zip(stars_test, params)] 
        stars_test_fitted = psf.drawStarList(pre_drawn_stars)
    else:
        #pre_drawn_stars = psf.drawStarList(stars_test)
	#plt.figure()
	#plt.imshow(stars_test[0].image.array)
        #plt.savefig('{0}/dummy_data_star.png'.format(directory))
	#print("finished making dummy star")
	#plt.figure()
	#plt.imshow(pre_drawn_stars[0].image.array)
        #plt.savefig('{0}/dummy_model_star.png'.format(directory))
	#print("finished making dummy star")
        pre_drawn_stars = [fit_star_alternate(star_test, psf) for star_test in stars_test]
        stars_test_fitted = psf.drawStarList(pre_drawn_stars)
	#plt.figure()
	#plt.imshow(pre_drawn_stars[0].image.array)
        #plt.savefig('{0}/dummy_model_star.png'.format(directory))
	#print("finished making dummy star")	 

    delete_list = []
    for star_i, star in enumerate(stars_test):
        if np.sum(stars_test[star_i].image.array)>2.0*np.sum(stars_test_fitted[star_i].image.array):
	    if is_optatmo==True:
                delete_list.append(star_i)    

    stars_test = np.delete(stars_test, delete_list)
    stars_test_fitted = np.delete(stars_test_fitted, delete_list)

    for star_train in stars_train:
	star_train.fit.params = None

    if is_optatmo==True:
        params = psf.getParamsList(stars_train)
        pre_drawn_stars = [psf.fit_model(star_train, param, vary_shape=False)[0] for star_train, param in zip(stars_train, params)] 
        stars_train_fitted = psf.drawStarList(pre_drawn_stars)
    else:
        #pre_drawn_stars = psf.drawStarList(stars_train)
        #stars_train_fitted = [fit_star_alternate(pre_drawn_star, psf) for pre_drawn_star in pre_drawn_stars] 
        pre_drawn_stars = [fit_star_alternate(star_train, psf) for star_train in stars_train]
        stars_train_fitted = psf.drawStarList(pre_drawn_stars)

    delete_list = []
    for star_i, star in enumerate(stars_train):
        if np.sum(stars_train[star_i].image.array)>2.0*np.sum(stars_train_fitted[star_i].image.array):
	    if is_optatmo==True:
                delete_list.append(star_i)   

    stars_train = np.delete(stars_train, delete_list)
    stars_train_fitted = np.delete(stars_train_fitted, delete_list)

    print("finished fitting stars")
    print("finished fitting stars")
    #import sys
    #sys.exit()



    star_plots = np.array([star_test.image.array for star_test in stars_test])
    norm = np.array([star_test.image.array.sum() for star_test in stars_test])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    test_real_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    plt.imshow(test_real_average_plot, vmin=np.percentile(test_real_average_plot, q=2),vmax=np.percentile(test_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Data, Test Stars normalized by data star flux')
    np.save("{0}/test_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",test_real_average_plot)
    print("preparing to save first figure")
    plt.savefig('{0}/stars_test_data_'.format(directory) + psf_type + '.png')
    print('{0}/stars_test_data_'.format(directory) + psf_type + '.png')
    print("finished saving first figure")

    star_plots = np.array([star_test_fitted.image.array for star_test_fitted in stars_test_fitted])
    norm = np.array([star_test.image.array.sum() for star_test in stars_test])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    test_fitted_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    plt.imshow(test_fitted_average_plot, vmin=np.percentile(test_real_average_plot, q=2),vmax=np.percentile(test_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Model, Test Stars normalized by data star flux')
    np.save("{0}/test_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",test_fitted_average_plot)
    plt.savefig('{0}/stars_test_model_'.format(directory) + psf_type + '.png')

    star_plots = np.array([stars_test[index].image.array-stars_test_fitted[index].image.array for index in range(0,len(stars_test))])
    norm = np.array([star_test.image.array.sum() for star_test in stars_test])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    test_difference_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    #plt.imshow(test_difference_average_plot, vmin=np.percentile(test_difference_average_plot, q=2),vmax=np.percentile(test_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.imshow(test_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Difference, Test Stars normalized by data star flux')
    np.save("{0}/test_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",test_difference_average_plot)
    plt.savefig('{0}/stars_test_difference_'.format(directory) + psf_type + '.png')

    print("preparing to do radial plot")
    x_test_real, y_test_real = convert_two_d_plot_to_radial_plot(test_real_average_plot)
    x_test_fitted, y_test_fitted = convert_two_d_plot_to_radial_plot(test_fitted_average_plot)
    x_test_difference, y_test_difference = convert_two_d_plot_to_radial_plot(test_difference_average_plot)

    plt.figure()
    plt.scatter(x_test_real,y_test_real, label="Data")
    plt.scatter(x_test_fitted,y_test_fitted, label="Model")
    plt.scatter(x_test_difference,y_test_difference, label="Difference")
    plt.legend()
    plt.title('Average Data, Model, and Difference for Test Stars normalized by data star flux')
    np.save("{0}/x_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_test_real)
    np.save("{0}/y_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_test_real)
    np.save("{0}/x_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_test_fitted)
    np.save("{0}/y_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_test_fitted)
    np.save("{0}/x_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_test_difference)
    np.save("{0}/y_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_test_difference)
    plt.savefig('{0}/stars_test_radial_'.format(directory) + psf_type + '.png')
    print("finished radial plot")



    stars_test_snr_level_0 = []
    stars_test_snr_level_1 = []
    stars_test_snr_level_2 = []

    stars_test_fitted_snr_level_0 = []
    stars_test_fitted_snr_level_1 = []
    stars_test_fitted_snr_level_2 = []

    real_divided_by_fitted_fluxes_snr_level_0 = []
    real_divided_by_fitted_fluxes_snr_level_1 = []
    real_divided_by_fitted_fluxes_snr_level_2 = []


    for star_test_i, star_test in enumerate(stars_test):
        snr = psf.measure_snr(star_test)
        if snr < 70.0:
            stars_test_snr_level_0.append(star_test)
            stars_test_fitted_snr_level_0.append(stars_test_fitted[star_test_i])
        if snr >= 70.0 and snr < 99.9:
            stars_test_snr_level_1.append(star_test)
            stars_test_fitted_snr_level_1.append(stars_test_fitted[star_test_i])      
        if snr >= 99.9:
            stars_test_snr_level_2.append(star_test)
            stars_test_fitted_snr_level_2.append(stars_test_fitted[star_test_i])        

    stars_test_snr_level_0 = np.array(stars_test_snr_level_0)
    stars_test_snr_level_1 = np.array(stars_test_snr_level_1)
    stars_test_snr_level_2 = np.array(stars_test_snr_level_2)

    stars_test_fitted_snr_level_0 = np.array(stars_test_fitted_snr_level_0)
    stars_test_fitted_snr_level_1 = np.array(stars_test_fitted_snr_level_1)
    stars_test_fitted_snr_level_2 = np.array(stars_test_fitted_snr_level_2)

    real_divided_by_fitted_fluxes_snr_level_0 = np.array(real_divided_by_fitted_fluxes_snr_level_0)
    real_divided_by_fitted_fluxes_snr_level_1 = np.array(real_divided_by_fitted_fluxes_snr_level_1)
    real_divided_by_fitted_fluxes_snr_level_2 = np.array(real_divided_by_fitted_fluxes_snr_level_2)


    print("len(stars_test_snr_level_0): {0}".format(len(stars_test_snr_level_0)))
    print("len(stars_test_snr_level_1): {0}".format(len(stars_test_snr_level_1)))
    print("len(stars_test_snr_level_2): {0}".format(len(stars_test_snr_level_2)))

    for l in [0, 1, 2]:
        if l ==0:
            stars_test_copy = stars_test_snr_level_0
            stars_test_fitted_copy = stars_test_fitted_snr_level_0 
        if l ==1:
            stars_test_copy = stars_test_snr_level_1
            stars_test_fitted_copy = stars_test_fitted_snr_level_1  
        if l ==2:
            stars_test_copy = stars_test_snr_level_2
            stars_test_fitted_copy = stars_test_fitted_snr_level_2   


	#try:
        star_plots = np.array([star_test.image.array for star_test in stars_test_copy])
        norm = np.array([star_test.image.array.sum() for star_test in stars_test_copy])
        norm_blown_up = norm[:, np.newaxis, np.newaxis]
        star_plots = star_plots / norm_blown_up
        test_real_average_plot = np.mean(star_plots, axis=0)
        plt.figure()
        plt.imshow(test_real_average_plot, vmin=np.percentile(test_real_average_plot, q=2),vmax=np.percentile(test_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title('Average Data, Test Stars normalized by data star flux for snr level {0}'.format(l))
	np.save("{0}/test_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", test_real_average_plot)
        plt.savefig('{0}/stars_test_data_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

        star_plots = np.array([star_test_fitted.image.array for star_test_fitted in stars_test_fitted_copy])
        norm = np.array([star_test.image.array.sum() for star_test in stars_test_copy])
        norm_blown_up = norm[:, np.newaxis, np.newaxis]
    	star_plots = star_plots / norm_blown_up
        test_fitted_average_plot = np.mean(star_plots, axis=0)
        plt.figure()
        plt.imshow(test_fitted_average_plot, vmin=np.percentile(test_real_average_plot, q=2),vmax=np.percentile(test_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title('Average Model, Test Stars normalized by data star flux for snr level {0}'.format(l))
	np.save("{0}/test_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", test_fitted_average_plot)
        plt.savefig('{0}/stars_test_model_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

        star_plots = np.array([stars_test_copy[index].image.array-stars_test_fitted_copy[index].image.array for index in range(0,len(stars_test_copy))])
        norm = np.array([star_test.image.array.sum() for star_test in stars_test_copy])
        norm_blown_up = norm[:, np.newaxis, np.newaxis]
        star_plots = star_plots / norm_blown_up
        test_difference_average_plot = np.mean(star_plots, axis=0)
        plt.figure()
        #plt.imshow(test_difference_average_plot, vmin=np.percentile(test_difference_average_plot, q=2),vmax=np.percentile(test_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
        plt.imshow(test_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title('Average Difference, Test Stars normalized by data star flux for snr level {0}'.format(l))
	np.save("{0}/test_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", test_difference_average_plot)
        plt.savefig('{0}/stars_test_difference_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

        x_test_real, y_test_real = convert_two_d_plot_to_radial_plot(test_real_average_plot)
        x_test_fitted, y_test_fitted = convert_two_d_plot_to_radial_plot(test_fitted_average_plot)
        x_test_difference, y_test_difference = convert_two_d_plot_to_radial_plot(test_difference_average_plot)

        plt.figure()
        plt.scatter(x_test_real,y_test_real, label="Data")
        plt.scatter(x_test_fitted,y_test_fitted, label="Model")
        plt.scatter(x_test_difference,y_test_difference, label="Difference")
        plt.legend()
        plt.title('Average Data, Model, and Difference for Test Stars normalized by data star flux for snr level {0}'.format(l))
	np.save("{0}/x_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_test_real)
	np.save("{0}/y_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_test_real)
	np.save("{0}/x_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_test_fitted)
	np.save("{0}/y_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_test_fitted)
	np.save("{0}/x_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_test_difference)
	np.save("{0}/y_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_test_difference)
        plt.savefig('{0}/stars_test_radial_snr_level_{1}_'.format(directory, l) + psf_type + '.png')
	#except:
	#    pass



    star_plots = np.array([star_train.image.array for star_train in stars_train])
    norm = np.array([star_train.image.array.sum() for star_train in stars_train])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    train_real_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    plt.imshow(train_real_average_plot, vmin=np.percentile(train_real_average_plot, q=2),vmax=np.percentile(train_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Data, train Stars normalized by data star flux')
    np.save("{0}/train_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",train_real_average_plot)
    plt.savefig('{0}/stars_train_data_'.format(directory) + psf_type + '.png')

    star_plots = np.array([star_train_fitted.image.array for star_train_fitted in stars_train_fitted])
    norm = np.array([star_train.image.array.sum() for star_train in stars_train])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    train_fitted_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    plt.imshow(train_fitted_average_plot, vmin=np.percentile(train_real_average_plot, q=2),vmax=np.percentile(train_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Model, train Stars normalized by data star flux')
    np.save("{0}/train_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",train_fitted_average_plot)
    plt.savefig('{0}/stars_train_model_'.format(directory) + psf_type + '.png')

    star_plots = np.array([stars_train[index].image.array-stars_train_fitted[index].image.array for index in range(0,len(stars_train))])
    norm = np.array([star_train.image.array.sum() for star_train in stars_train])
    norm_blown_up = norm[:, np.newaxis, np.newaxis]
    star_plots = star_plots / norm_blown_up
    train_difference_average_plot = np.mean(star_plots, axis=0)
    plt.figure()
    #plt.imshow(train_difference_average_plot, vmin=np.percentile(train_difference_average_plot, q=2),vmax=np.percentile(train_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.imshow(train_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Difference, train Stars normalized by data star flux')
    np.save("{0}/train_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy",train_difference_average_plot)
    plt.savefig('{0}/stars_train_difference_'.format(directory) + psf_type + '.png')

    x_train_real, y_train_real = convert_two_d_plot_to_radial_plot(train_real_average_plot)
    x_train_fitted, y_train_fitted = convert_two_d_plot_to_radial_plot(train_fitted_average_plot)
    x_train_difference, y_train_difference = convert_two_d_plot_to_radial_plot(train_difference_average_plot)

    plt.figure()
    plt.scatter(x_train_real,y_train_real, label="Data")
    plt.scatter(x_train_fitted,y_train_fitted, label="Model")
    plt.scatter(x_train_difference,y_train_difference, label="Difference")
    plt.legend()
    plt.title('Average Data, Model, and Difference for train Stars normalized by data star flux')
    np.save("{0}/x_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_train_real)
    np.save("{0}/y_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_train_real)
    np.save("{0}/x_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_train_fitted)
    np.save("{0}/y_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_train_fitted)
    np.save("{0}/x_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", x_train_difference)
    np.save("{0}/y_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy", y_train_difference)
    plt.savefig('{0}/stars_train_radial_'.format(directory) + psf_type + '.png')



    stars_train_snr_level_0 = []
    stars_train_snr_level_1 = []
    stars_train_snr_level_2 = []

    stars_train_fitted_snr_level_0 = []
    stars_train_fitted_snr_level_1 = []
    stars_train_fitted_snr_level_2 = []

    real_divided_by_fitted_fluxes_snr_level_0 = []
    real_divided_by_fitted_fluxes_snr_level_1 = []
    real_divided_by_fitted_fluxes_snr_level_2 = []


    for star_train_i, star_train in enumerate(stars_train):
        snr = psf.measure_snr(star_train)
        if snr < 70.0:
            stars_train_snr_level_0.append(star_train)
            stars_train_fitted_snr_level_0.append(stars_train_fitted[star_train_i])
        if snr >= 70.0 and snr < 99.9:
            stars_train_snr_level_1.append(star_train)
            stars_train_fitted_snr_level_1.append(stars_train_fitted[star_train_i])      
        if snr >= 99.9:
            stars_train_snr_level_2.append(star_train)
            stars_train_fitted_snr_level_2.append(stars_train_fitted[star_train_i])        

    stars_train_snr_level_0 = np.array(stars_train_snr_level_0)
    stars_train_snr_level_1 = np.array(stars_train_snr_level_1)
    stars_train_snr_level_2 = np.array(stars_train_snr_level_2)

    stars_train_fitted_snr_level_0 = np.array(stars_train_fitted_snr_level_0)
    stars_train_fitted_snr_level_1 = np.array(stars_train_fitted_snr_level_1)
    stars_train_fitted_snr_level_2 = np.array(stars_train_fitted_snr_level_2)

    real_divided_by_fitted_fluxes_snr_level_0 = np.array(real_divided_by_fitted_fluxes_snr_level_0)
    real_divided_by_fitted_fluxes_snr_level_1 = np.array(real_divided_by_fitted_fluxes_snr_level_1)
    real_divided_by_fitted_fluxes_snr_level_2 = np.array(real_divided_by_fitted_fluxes_snr_level_2)


    print(len(stars_train_snr_level_0))
    print(len(stars_train_snr_level_1))
    print(len(stars_train_snr_level_2))

    for l in [0, 1, 2]:
        if l ==0:
            stars_train_copy = stars_train_snr_level_0
            stars_train_fitted_copy = stars_train_fitted_snr_level_0 
        if l ==1:
            stars_train_copy = stars_train_snr_level_1
            stars_train_fitted_copy = stars_train_fitted_snr_level_1  
        if l ==2:
            stars_train_copy = stars_train_snr_level_2
            stars_train_fitted_copy = stars_train_fitted_snr_level_2 


	try:
            star_plots = np.array([star_train.image.array for star_train in stars_train_copy])
            norm = np.array([star_train.image.array.sum() for star_train in stars_train_copy])
            norm_blown_up = norm[:, np.newaxis, np.newaxis]
            star_plots = star_plots / norm_blown_up
            train_real_average_plot = np.mean(star_plots, axis=0)
            plt.figure()
            plt.imshow(train_real_average_plot, vmin=np.percentile(train_real_average_plot, q=2),vmax=np.percentile(train_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Data, train Stars normalized by data star flux for snr level {0}'.format(l))
	    np.save("{0}/train_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", train_real_average_plot)
            plt.savefig('{0}/stars_train_data_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

            star_plots = np.array([star_train_fitted.image.array for star_train_fitted in stars_train_fitted_copy])
            norm = np.array([star_train.image.array.sum() for star_train in stars_train_copy])
            norm_blown_up = norm[:, np.newaxis, np.newaxis]
    	    star_plots = star_plots / norm_blown_up
            train_fitted_average_plot = np.mean(star_plots, axis=0)
            plt.figure()
            plt.imshow(train_fitted_average_plot, vmin=np.percentile(train_real_average_plot, q=2),vmax=np.percentile(train_real_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Model, train Stars normalized by data star flux for snr level {0}'.format(l))
	    np.save("{0}/train_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", train_fitted_average_plot)
            plt.savefig('{0}/stars_train_model_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

            star_plots = np.array([stars_train_copy[index].image.array-stars_train_fitted_copy[index].image.array for index in range(0,len(stars_train_copy))])
            norm = np.array([star_train.image.array.sum() for star_train in stars_train_copy])
            norm_blown_up = norm[:, np.newaxis, np.newaxis]
            star_plots = star_plots / norm_blown_up
            train_difference_average_plot = np.mean(star_plots, axis=0)
            plt.figure()
            #plt.imshow(train_difference_average_plot, vmin=np.percentile(train_difference_average_plot, q=2),vmax=np.percentile(train_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.imshow(train_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Difference, train Stars normalized by data star flux for snr level {0}'.format(l))
	    np.save("{0}/train_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", train_difference_average_plot)
            plt.savefig('{0}/stars_train_difference_snr_level_{1}_'.format(directory, l) + psf_type + '.png')

            x_train_real, y_train_real = convert_two_d_plot_to_radial_plot(train_real_average_plot)
            x_train_fitted, y_train_fitted = convert_two_d_plot_to_radial_plot(train_fitted_average_plot)
            x_train_difference, y_train_difference = convert_two_d_plot_to_radial_plot(train_difference_average_plot)

            plt.figure()
            plt.scatter(x_train_real,y_train_real, label="Data")
            plt.scatter(x_train_fitted,y_train_fitted, label="Model")
            plt.scatter(x_train_difference,y_train_difference, label="Difference")
            plt.legend()
            plt.title('Average Data, Model, and Difference for train Stars normalized by data star flux for snr level {0}'.format(l))
	    np.save("{0}/x_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_train_real)
	    np.save("{0}/y_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_train_real)
	    np.save("{0}/x_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_train_fitted)
	    np.save("{0}/y_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_train_fitted)
	    np.save("{0}/x_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", x_train_difference)
	    np.save("{0}/y_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy", y_train_difference)
            plt.savefig('{0}/stars_train_radial_snr_level_{1}_'.format(directory, l) + psf_type + '.png')
	except:
	    pass





if __name__ == '__main__':
    print("preparing to parse arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exposure')
    parser.add_argument('--core_directory')
    parser.add_argument('--psf_type')
    options = parser.parse_args()
    exposure = options.exposure
    core_directory = options.core_directory
    psf_type = options.psf_type
    kwargs = vars(options)
    del kwargs['exposure']
    del kwargs['core_directory']
    del kwargs['psf_type']
    make_star_residual_plots()
