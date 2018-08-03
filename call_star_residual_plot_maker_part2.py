from __future__ import print_function, division

# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy import stats
from astropy.io import fits
import fitsio
import matplotlib.pyplot as plt
import os
import glob

import lmfit
import galsim
import piff

#from zernike import Zernike
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure




def make_call():
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = core_directory + "/graph_values_npy_storage"
    graph_directory = "{0}/multi_exposure_graphs/{1}_star_residual_plots_averaged_across_exposures".format(core_directory, psf_type)
    terminal_command = os.system("mkdir {0}".format(graph_directory))
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    original_exposures = glob.glob("{0}/*".format(source_directory))
    original_exposures = [original_exposure.split("/")[-1] for original_exposure in original_exposures]
    if band=="all":
	exposures = original_exposures
    else:
	exposures = []
	for original_exposure in original_exposures:
	    skip=False
	    for index in range(1,63):
		try:
		    band_test_file = "{0}/{1}/psf_cat_{1}_{2}.fits".format(psf_dir, original_exposure, index)
		    hdu = fits.open(band_test_file)
		    break
		except:
		    if index==62:
			skip = True
		    else:
		        pass
	    if skip==True:
		continue
    	    filter_name = hdu[3].data['band'][0]
    	    if filter_name==band:
                exposures.append(original_exposure)  
	graph_directory = graph_directory + "/star_residual_plots_just_for_filter_{0}".format(band)
	os.system("mkdir {0}".format(graph_directory))

    test_data_average_plot = []
    test_model_average_plot = []
    test_difference_average_plot = []

    x_test_data = []
    y_test_data = []
    x_test_model = []
    y_test_model = []
    x_test_difference = []
    y_test_difference = []

    for exposure_i, exposure in enumerate(exposures):
        try:
	    # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
            if not np.any(np.isnan(np.load("{0}/test_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                test_data_average_plot.append(np.load("{0}/test_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))
            if not np.any(np.isnan(np.load("{0}/test_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                test_model_average_plot.append(np.load("{0}/test_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))
            if not np.any(np.isnan(np.load("{0}/test_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                test_difference_average_plot.append(np.load("{0}/test_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))

            if not np.any(np.isnan(np.load("{0}/x_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_test_data.append(np.load("{0}/x_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))                 
            if not np.any(np.isnan(np.load("{0}/y_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_test_data.append(np.load("{0}/y_test_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy")) 
            if not np.any(np.isnan(np.load("{0}/x_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_test_model.append(np.load("{0}/x_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))  
            if not np.any(np.isnan(np.load("{0}/y_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_test_model.append(np.load("{0}/y_test_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))   
            if not np.any(np.isnan(np.load("{0}/x_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_test_difference.append(np.load("{0}/x_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))                 
            if not np.any(np.isnan(np.load("{0}/y_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_test_difference.append(np.load("{0}/y_test_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy")) 

        except:
            pass

    test_data_average_plot = np.nanmean(np.array(test_data_average_plot),axis=0)
    test_model_average_plot = np.nanmean(np.array(test_model_average_plot),axis=0)
    test_difference_average_plot = np.nanmean(np.array(test_difference_average_plot),axis=0)

    x_test_data = np.nanmean(np.array(x_test_data),axis=0)
    y_test_data = np.nanmean(np.array(y_test_data),axis=0)
    x_test_model = np.nanmean(np.array(x_test_model),axis=0)
    y_test_model = np.nanmean(np.array(y_test_model),axis=0)
    x_test_difference = np.nanmean(np.array(x_test_difference),axis=0)
    y_test_difference = np.nanmean(np.array(y_test_difference),axis=0)

    plt.figure()
    plt.imshow(test_data_average_plot, vmin=np.percentile(test_data_average_plot, q=2),vmax=np.percentile(test_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Data, Test Stars normalized by data star flux')
    plt.savefig('{0}/stars_test_data_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.imshow(test_model_average_plot, vmin=np.percentile(test_data_average_plot, q=2),vmax=np.percentile(test_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Model, Test Stars normalized by data star flux')
    plt.savefig('{0}/stars_test_model_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.imshow(test_difference_average_plot, vmin=np.percentile(test_difference_average_plot, q=2),vmax=np.percentile(test_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.imshow(test_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Difference, Test Stars normalized by data star flux')
    plt.savefig('{0}/stars_test_difference_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.scatter(x_test_data,y_test_data, label="Data")
    plt.scatter(x_test_model,y_test_model, label="Model")
    plt.scatter(x_test_difference,y_test_difference, label="Difference")
    plt.legend()
    plt.title('Average Data, Model, and Difference for Test Stars normalized by data star flux')
    plt.savefig('{0}/stars_test_radial_'.format(graph_directory) + psf_type + '.png')



    for l in [0, 1, 2]:
        try:
            test_data_average_plot = []
            test_model_average_plot = []
            test_difference_average_plot = []

            x_test_data = []
            y_test_data = []
            x_test_model = []
            y_test_model = []
            x_test_difference = []
            y_test_difference = []

            for exposure_i, exposure in enumerate(exposures):
                try:
                    if not np.any(np.isnan(np.load("{0}/test_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        test_data_average_plot.append(np.load("{0}/test_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))
                    if not np.any(np.isnan(np.load("{0}/test_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        test_model_average_plot.append(np.load("{0}/test_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))
                    if not np.any(np.isnan(np.load("{0}/test_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        test_difference_average_plot.append(np.load("{0}/test_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))

                    if not np.any(np.isnan(np.load("{0}/x_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_test_data.append(np.load("{0}/x_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_test_data.append(np.load("{0}/y_test_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                    if not np.any(np.isnan(np.load("{0}/x_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_test_model.append(np.load("{0}/x_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_test_model.append(np.load("{0}/y_test_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                    if not np.any(np.isnan(np.load("{0}/x_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_test_difference.append(np.load("{0}/x_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_test_difference.append(np.load("{0}/y_test_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                except:
                    pass

            test_data_average_plot = np.nanmean(np.array(test_data_average_plot),axis=0)
            test_model_average_plot = np.nanmean(np.array(test_model_average_plot),axis=0)
            test_difference_average_plot = np.nanmean(np.array(test_difference_average_plot),axis=0)

            x_test_data = np.nanmean(np.array(x_test_data),axis=0)
            y_test_data = np.nanmean(np.array(y_test_data),axis=0)
            x_test_model = np.nanmean(np.array(x_test_model),axis=0)
            y_test_model = np.nanmean(np.array(y_test_model),axis=0)
            x_test_difference = np.nanmean(np.array(x_test_difference),axis=0)
            y_test_difference = np.nanmean(np.array(y_test_difference),axis=0)

            plt.figure()
            plt.imshow(test_data_average_plot, vmin=np.percentile(test_data_average_plot, q=2),vmax=np.percentile(test_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Data, Test Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_test_data_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.imshow(test_model_average_plot, vmin=np.percentile(test_data_average_plot, q=2),vmax=np.percentile(test_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Model, Test Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_test_model_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.imshow(test_difference_average_plot, vmin=np.percentile(test_difference_average_plot, q=2),vmax=np.percentile(test_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.imshow(test_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Difference, Test Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_test_difference_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.scatter(x_test_data,y_test_data, label="Data")
            plt.scatter(x_test_model,y_test_model, label="Model")
            plt.scatter(x_test_difference,y_test_difference, label="Difference")
            plt.legend()
            plt.title('Average Data, Model, and Difference for Test Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_test_radial_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')
        except:
            pass





    train_data_average_plot = []
    train_model_average_plot = []
    train_difference_average_plot = []

    x_train_data = []
    y_train_data = []
    x_train_model = []
    y_train_model = []
    x_train_difference = []
    y_train_difference = []

    for exposure_i, exposure in enumerate(exposures):
        try:

            if not np.any(np.isnan(np.load("{0}/train_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                train_data_average_plot.append(np.load("{0}/train_data_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))
            if not np.any(np.isnan(np.load("{0}/train_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                train_model_average_plot.append(np.load("{0}/train_model_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))
            if not np.any(np.isnan(np.load("{0}/train_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):                
                train_difference_average_plot.append(np.load("{0}/train_difference_average_plot_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))

            if not np.any(np.isnan(np.load("{0}/x_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_train_data.append(np.load("{0}/x_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))                 
            if not np.any(np.isnan(np.load("{0}/y_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_train_data.append(np.load("{0}/y_train_data_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy")) 
            if not np.any(np.isnan(np.load("{0}/x_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_train_model.append(np.load("{0}/x_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))  
            if not np.any(np.isnan(np.load("{0}/y_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_train_model.append(np.load("{0}/y_train_model_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))   
            if not np.any(np.isnan(np.load("{0}/x_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                x_train_difference.append(np.load("{0}/x_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))                 
            if not np.any(np.isnan(np.load("{0}/y_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy"))):
                y_train_difference.append(np.load("{0}/y_train_difference_{1}_".format(graph_values_directory, exposure) + psf_type + ".npy")) 

        except:
            pass

    train_data_average_plot = np.nanmean(np.array(train_data_average_plot),axis=0)
    train_model_average_plot = np.nanmean(np.array(train_model_average_plot),axis=0)
    train_difference_average_plot = np.nanmean(np.array(train_difference_average_plot),axis=0)

    x_train_data = np.nanmean(np.array(x_train_data),axis=0)
    y_train_data = np.nanmean(np.array(y_train_data),axis=0)
    x_train_model = np.nanmean(np.array(x_train_model),axis=0)
    y_train_model = np.nanmean(np.array(y_train_model),axis=0)
    x_train_difference = np.nanmean(np.array(x_train_difference),axis=0)
    y_train_difference = np.nanmean(np.array(y_train_difference),axis=0)

    plt.figure()
    plt.imshow(train_data_average_plot, vmin=np.percentile(train_data_average_plot, q=2),vmax=np.percentile(train_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Data, Train Stars normalized by data star flux')
    plt.savefig('{0}/stars_train_data_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.imshow(train_model_average_plot, vmin=np.percentile(train_data_average_plot, q=2),vmax=np.percentile(train_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Model, Train Stars normalized by data star flux')
    plt.savefig('{0}/stars_train_model_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.imshow(train_difference_average_plot, vmin=np.percentile(train_difference_average_plot, q=2),vmax=np.percentile(train_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
    plt.imshow(train_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title('Average Difference, Train Stars normalized by data star flux')
    plt.savefig('{0}/stars_train_difference_'.format(graph_directory) + psf_type + '.png')

    plt.figure()
    plt.scatter(x_train_data,y_train_data, label="Data")
    plt.scatter(x_train_model,y_train_model, label="Model")
    plt.scatter(x_train_difference,y_train_difference, label="Difference")
    plt.legend()
    plt.title('Average Data, Model, and Difference for Train Stars normalized by data star flux')
    plt.savefig('{0}/stars_train_radial_'.format(graph_directory) + psf_type + '.png')



    for l in [0, 1, 2]:
        try:
            train_data_average_plot = []
            train_model_average_plot = []
            train_difference_average_plot = []

            x_train_data = []
            y_train_data = []
            x_train_model = []
            y_train_model = []
            x_train_difference = []
            y_train_difference = []

            for exposure_i, exposure in enumerate(exposures):
                try:
                    if not np.any(np.isnan(np.load("{0}/train_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        train_data_average_plot.append(np.load("{0}/train_data_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))
                    if not np.any(np.isnan(np.load("{0}/train_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        train_model_average_plot.append(np.load("{0}/train_model_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))
                    if not np.any(np.isnan(np.load("{0}/train_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):                
                        train_difference_average_plot.append(np.load("{0}/train_difference_average_plot_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))

                    if not np.any(np.isnan(np.load("{0}/x_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_train_data.append(np.load("{0}/x_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_train_data.append(np.load("{0}/y_train_data_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                    if not np.any(np.isnan(np.load("{0}/x_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_train_model.append(np.load("{0}/x_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_train_model.append(np.load("{0}/y_train_model_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                    if not np.any(np.isnan(np.load("{0}/x_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        x_train_difference.append(np.load("{0}/x_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy"))):
                        y_train_difference.append(np.load("{0}/y_train_difference_snr_level_{1}_{2}_".format(graph_values_directory, l, exposure) + psf_type + ".npy")) 
                except:
                    pass

            train_data_average_plot = np.nanmean(np.array(train_data_average_plot),axis=0)
            train_model_average_plot = np.nanmean(np.array(train_model_average_plot),axis=0)
            train_difference_average_plot = np.nanmean(np.array(train_difference_average_plot),axis=0)

            x_train_data = np.nanmean(np.array(x_train_data),axis=0)
            y_train_data = np.nanmean(np.array(y_train_data),axis=0)
            x_train_model = np.nanmean(np.array(x_train_model),axis=0)
            y_train_model = np.nanmean(np.array(y_train_model),axis=0)
            x_train_difference = np.nanmean(np.array(x_train_difference),axis=0)
            y_train_difference = np.nanmean(np.array(y_train_difference),axis=0)

            plt.figure()
            plt.imshow(train_data_average_plot, vmin=np.percentile(train_data_average_plot, q=2),vmax=np.percentile(train_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Data, Train Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_train_data_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.imshow(train_model_average_plot, vmin=np.percentile(train_data_average_plot, q=2),vmax=np.percentile(train_data_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Model, Train Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_train_model_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.imshow(train_difference_average_plot, vmin=np.percentile(train_difference_average_plot, q=2),vmax=np.percentile(train_difference_average_plot, q=98), cmap=plt.cm.RdBu_r)
            plt.imshow(train_difference_average_plot, vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average Difference, Train Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_train_difference_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')

            plt.figure()
            plt.scatter(x_train_data,y_train_data, label="Data")
            plt.scatter(x_train_model,y_train_model, label="Model")
            plt.scatter(x_train_difference,y_train_difference, label="Difference")
            plt.legend()
            plt.title('Average Data, Model, and Difference for Train Stars normalized by data star flux for snr level {0}'.format(l))
            plt.savefig('{0}/stars_train_radial_snr_level_{1}_'.format(graph_directory, l) + psf_type + '.png')
        except:
            pass




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--band')
    parser.add_argument('--psf_type')
    options = parser.parse_args()
    band = options.band
    psf_type = options.psf_type
    kwargs = vars(options)
    del kwargs['band']
    del kwargs['psf_type']
    make_call()
