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

from piff.util import hsm_error, measure_snr

def make_pngs(directory, label, plot_information, radial_information):
    #This function graphs the stars' data, model, and difference images.
    fig, axss = plt.subplots(nrows=4, ncols=5, figsize=(4 * 4, 4 * 5), squeeze=False)
    # left column gets the Y coordinate label
    for i in range(0,4):
        axss[i][0].imshow(plot_information[i*3+0], vmin=np.percentile(plot_information[i*3+0], q=2),vmax=np.percentile(plot_information[i*3+0], q=98), cmap=plt.cm.RdBu_r)
        if i==0:
            axss[i][0].set_title("data for all snr levels")
        else:
            axss[i][0].set_title("data for snr_level {0}".format(i-1))
        axss[i][1].imshow(plot_information[i*3+1], vmin=np.percentile(plot_information[i*3+0], q=2),vmax=np.percentile(plot_information[i*3+0], q=98), cmap=plt.cm.RdBu_r)
        if i==0:
            axss[i][1].set_title("model for all snr levels")
        else:
            axss[i][1].set_title("model for snr_level {0}".format(i-1))
        axss[i][2].imshow(plot_information[i*3+2], vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
        if i==0:
            axss[i][2].set_title("difference for all snr levels")
        else:
            axss[i][2].set_title("difference for snr_level {0}".format(i-1))

        axss[i][3].scatter(radial_information[i*6+0], radial_information[i*6+1], label="Data")
        axss[i][3].scatter(radial_information[i*6+2], radial_information[i*6+3], label="Model")
        if i==0:
            axss[i][3].set_title("radial data and model for all snr levels")
        else:
            axss[i][3].set_title("radial data and model for snr_level {0}".format(i-1))
        axss[i][4].scatter(radial_information[i*6+4], radial_information[i*6+5], label="Difference")
        if i==0:
            axss[i][4].set_title("radial difference for all snr levels")
            axss[i][4].set_ylim(-0.005,0.005)
        else:
            axss[i][4].set_title("radial difference for snr_level {0}".format(i-1))
            axss[i][4].set_ylim(-0.005,0.005)
    plt.tight_layout()
    fig.savefig("{0}/{1}_{2}_star_residuals.png".format(directory, label, psf_type))

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

def make_star_residual_plots(exposure, core_directory, psf_type):
    directory = "{0}/00{1}".format(core_directory, exposure)
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)
    
    # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
    psf = piff.read("{0}/psf_{1}.piff".format(directory, psf_type))   

    is_optatmo=True
    if psf_type.split("_")[0]!="optatmo":
        is_optatmo=False
    print(is_optatmo)
    
    
    for label in ["test", "train"]:
        stars_label = np.load("{0}/stars_{1}_psf_{2}.npy".format(directory, label, psf_type))
        for star_label in stars_label:
            star_label.fit.params = None
        early_delete_list = []    
        pre_drawn_stars = []        
        if is_optatmo==True:
            params = psf.getParamsList(stars_label)
            for sl, param in zip(range(0,len(stars_label)), params):
                star_label = stars_label[sl]
                try:
                    pre_drawn_star = psf.fit_model(star_label, param, vary_shape=False, estimated_errorbars_not_required=True)[0]
                    pre_drawn_stars.append(pre_drawn_star)
                except:
                    early_delete_list.append(sl)
            stars_label = np.delete(stars_label, early_delete_list)                    
            stars_label_fitted = psf.drawStarList(pre_drawn_stars)
        else:
            for sl, star_label in enumerate(stars_label):
                try:
                    pre_drawn_star = fit_star_alternate(star_label, psf)
                    pre_drawn_stars.append(pre_drawn_star)
                except:
                    early_delete_list.append(sl)
            stars_label = np.delete(stars_label, early_delete_list)               
            stars_label_fitted = psf.drawStarList(pre_drawn_stars)
        delete_list = []
        for star_i, star in enumerate(stars_label):
            if np.sum(stars_label[star_i].image.array)>2.0*np.sum(stars_label_fitted[star_i].image.array):
                if is_optatmo==True:
                    delete_list.append(star_i)    
        stars_label = np.delete(stars_label, delete_list)
        stars_label_fitted = np.delete(stars_label_fitted, delete_list)
        print("finished fitting {0} stars".format(label))
        
        side_length = len(stars_label[0].image.array)
        plot_information = np.empty([12, side_length, side_length])
        x_label_dummy, y_label_dummy = convert_two_d_plot_to_radial_plot(stars_label[0].image.array)
        radial_length = len(x_label_dummy)
        radial_information = np.empty([24, radial_length])
        
        norm = np.array([star_label.image.array.sum() for star_label in stars_label])
        norm_blown_up = norm[:, np.newaxis, np.newaxis]
        capital_kind_names = ["Data", "Model", "Difference"]
        kind_names = ["data", "model", "difference"]
        label_kind_average_plots = []
        stars_label_kinds = []
        stars_label_kinds.append(np.array([star_label.image.array for star_label in stars_label]))
        stars_label_kinds.append(np.array([star_label_fitted.image.array for star_label_fitted in stars_label_fitted]))
        stars_label_kinds.append(np.array([stars_label[index].image.array-stars_label_fitted[index].image.array for index in range(0,len(stars_label))]))
        
        for slk, stars_label_kind in enumerate(stars_label_kinds):
            star_plots = stars_label_kind
            star_plots = star_plots / norm_blown_up
            label_kind_average_plot = np.mean(star_plots, axis=0)
            label_kind_average_plots.append(label_kind_average_plot)
            plt.figure()
            plot_information[slk] = label_kind_average_plot
            if slk==2:
                plt.imshow(label_kind_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
            else:
                plt.imshow(label_kind_average_plot, vmin=np.percentile(label_kind_average_plots[0], q=2),vmax=np.percentile(label_kind_average_plots[0], q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average {0}, T{1} Stars normalized by data star flux'.format(capital_kind_names[slk], label[1:]))
            np.save("{0}/{1}_{2}_average_plot_{3}_".format(graph_values_directory, label, kind_names[slk], exposure) + psf_type + ".npy",label_kind_average_plot)
            #plt.savefig('{0}/stars_{1}_{2}_'.format(directory, label, kind_names[slk]) + psf_type + '.png')
            #note: this plot is never saved
            
        plt.figure()
        for k, label_kind_average_plot in enumerate(label_kind_average_plots):
            x_label_kind, y_label_kind = convert_two_d_plot_to_radial_plot(label_kind_average_plot)
            radial_information[2*k] = x_label_kind
            radial_information[1+2*k] = y_label_kind
            plt.scatter(x_label_kind,y_label_kind, label=capital_kind_names[k])
            np.save("{0}/x_{1}_{2}_{3}_".format(graph_values_directory, label, kind_names[k], exposure) + psf_type + ".npy", x_label_kind)
            np.save("{0}/y_{1}_{2}_{3}_".format(graph_values_directory, label, kind_names[k], exposure) + psf_type + ".npy", y_label_kind)          
        plt.legend()
        plt.title('Average Data, Model, and Difference for T{0} Stars normalized by data star flux'.format(label[1:]))
        #plt.savefig('{0}/stars_{1}_radial_'.format(directory, label) + psf_type + '.png')
        #note: this radial plot here does not split difference from data and model, but this is ok since this plot is never saved

        if label=="train":
            make_pngs(directory=directory, label=label, plot_information=plot_information, radial_information=radial_information)
            break

        stars_label_kind_snr_level_l_dictionary = {}
        for kind in ["data", "model"]:
            for l in ["0", "1", "2"]:
                stars_label_kind_snr_level_l_dictionary["{0}_{1}".format(kind, l)] = []

        for star_label_i, star_label in enumerate(stars_label):
            snr = psf.measure_snr(star_label)
            if snr < 70.0:
                stars_label_kind_snr_level_l_dictionary["data_0"].append(star_label)
                stars_label_kind_snr_level_l_dictionary["model_0"].append(stars_label_fitted[star_label_i])
            if snr >= 70.0 and snr < 90.0:
                stars_label_kind_snr_level_l_dictionary["data_1"].append(star_label)
                stars_label_kind_snr_level_l_dictionary["model_1"].append(stars_label_fitted[star_label_i])   
            if snr >= 90.0:
                stars_label_kind_snr_level_l_dictionary["data_2"].append(star_label)
                stars_label_kind_snr_level_l_dictionary["model_2"].append(stars_label_fitted[star_label_i])        

        for kind in ["data", "model"]:
            for l in ["0", "1", "2"]:
                stars_label_kind_snr_level_l_dictionary["{0}_{1}".format(kind, l)] = np.array(stars_label_kind_snr_level_l_dictionary["{0}_{1}".format(kind, l)])

        for l in [0, 1, 2]:
            if l ==0:
                stars_label_copy = stars_label_kind_snr_level_l_dictionary["data_0"]
                stars_label_fitted_copy = stars_label_kind_snr_level_l_dictionary["model_0"]
            if l ==1:
                stars_label_copy = stars_label_kind_snr_level_l_dictionary["data_1"]
                stars_label_fitted_copy = stars_label_kind_snr_level_l_dictionary["model_1"]
            if l ==2:
                stars_label_copy = stars_label_kind_snr_level_l_dictionary["data_2"]
                stars_label_fitted_copy = stars_label_kind_snr_level_l_dictionary["model_2"]
                
            norm = np.array([star_label.image.array.sum() for star_label in stars_label_copy])
            norm_blown_up = norm[:, np.newaxis, np.newaxis]
            label_kind_average_plots = []
            stars_label_kinds = []
            stars_label_kinds.append(np.array([star_label.image.array for star_label in stars_label_copy]))
            stars_label_kinds.append(np.array([star_label_fitted.image.array for star_label_fitted in stars_label_fitted_copy]))
            stars_label_kinds.append(np.array([stars_label_copy[index].image.array-stars_label_fitted_copy[index].image.array for index in range(0,len(stars_label_copy))]))
                
            for slk, stars_label_kind in enumerate(stars_label_kinds):
                star_plots = stars_label_kind
                star_plots = star_plots / norm_blown_up
                label_kind_average_plot = np.mean(star_plots, axis=0)
                label_kind_average_plots.append(label_kind_average_plot)
                plt.figure()
                plot_information[(l+1)*3+slk] = label_kind_average_plot
                if slk==2:
                    plt.imshow(label_kind_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
                else:
                    plt.imshow(label_kind_average_plot, vmin=np.percentile(label_kind_average_plots[0], q=2),vmax=np.percentile(label_kind_average_plots[0], q=98), cmap=plt.cm.RdBu_r)
                plt.colorbar()
                plt.title('Average {0}, T{1} Stars normalized by data star flux for snr level {2}'.format(capital_kind_names[slk], label[1:], l))
                np.save("{0}/{1}_{2}_average_plot_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_names[slk], l, exposure) + psf_type + ".npy", label_kind_average_plot)
                #plt.savefig('{0}/stars_{1}_{2}_snr_level_{3}_'.format(directory, label, kind_names[slk], l) + psf_type + '.png')     
                #note: this plot is never saved           
                                
            plt.figure()
            for k, label_kind_average_plot in enumerate(label_kind_average_plots):
                x_label_kind, y_label_kind = convert_two_d_plot_to_radial_plot(label_kind_average_plot)
                radial_information[(l+1)*6+2*k] = x_label_kind
                radial_information[(l+1)*6+1+2*k] = y_label_kind
                plt.scatter(x_label_kind,y_label_kind, label=capital_kind_names[k])
                np.save("{0}/x_{1}_{2}_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_names[k], l, exposure) + psf_type + ".npy", x_label_kind)
                np.save("{0}/y_{1}_{2}_snr_level_{3}_{4}".format(graph_values_directory, label, kind_names[k], l, exposure) + psf_type + ".npy", y_label_kind)          
            plt.legend()
            plt.title('Average Data, Model, and Difference for T{0} Stars normalized by data star flux for snr level {1}'.format(label[1:], l))
            #plt.savefig('{0}/stars_{1}_radial_snr_level_{2}_'.format(directory, label, l) + psf_type + '.png')
            #note: this radial plot here does not split difference from data and model, but this is ok since this plot is never saved


        make_pngs(directory=directory, label=label, plot_information=plot_information, radial_information=radial_information)



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
    make_star_residual_plots(exposure=exposure, core_directory=core_directory, psf_type=psf_type)
