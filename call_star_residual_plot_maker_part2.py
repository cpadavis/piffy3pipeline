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
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip




def make_call(band, psf_type):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = core_directory + "/graph_values_npy_storage"
    graph_directory = "{0}/multi_exposure_graphs/{1}_star_residual_plots_averaged_across_exposures".format(core_directory, psf_type)
    os.system("mkdir {0}".format(graph_directory))
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    original_exposures = glob.glob("{0}/*".format(source_directory))
    original_exposures = [original_exposure.split("/")[-1] for original_exposure in original_exposures]
    if band=="all":
        exposures = original_exposures
    else:
        exposures = []
        for original_exposure in original_exposures:
            filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=original_exposure)
            if filter_name_and_skip_dictionary['skip'] == True:
                continue
            else:
                filter_name = filter_name_and_skip_dictionary['filter_name']       
            if filter_name in band:
                exposures.append(original_exposure)  
        graph_directory = graph_directory + "/star_residual_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    
    for label in ["test", "train"]:
        capital_kind_names = ["Data", "Model", "Difference"]
        kind_names = ["data", "model", "difference"]
        label_kind_average_plot_dictionary = {}
        x_label_kind_dictionary = {}
        y_label_kind_dictionary = {}
        for kind_name in kind_names:
            label_kind_average_plot_dictionary[kind_name] = []
            x_label_kind_dictionary[kind_name] = []
            y_label_kind_dictionary[kind_name] = []

        for exposure_i, exposure in enumerate(exposures):
            try:
                # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
                for kind_name in kind_names:
                    if not np.any(np.isnan(np.load("{0}/{1}_{2}_average_plot_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))):                
                        label_kind_average_plot_dictionary[kind_name].append(np.load("{0}/{1}_{2}_average_plot_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))
                    if not np.any(np.isnan(np.load("{0}/x_{1}_{2}_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))):
                        x_label_kind_dictionary[kind_name].append(np.load("{0}/x_{1}_{2}_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))                 
                    if not np.any(np.isnan(np.load("{0}/y_{1}_{2}_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))):
                        y_label_kind_dictionary[kind_name].append(np.load("{0}/y_{1}_{2}_{3}_".format(graph_values_directory, label, kind_name, exposure) + psf_type + ".npy"))                 
            except:
                pass

        for k, kind_name in enumerate(kind_names):
            label_kind_average_plot_dictionary[kind_name] = np.nanmean(np.array(label_kind_average_plot_dictionary[kind_name]),axis=0)
            x_label_kind_dictionary[kind_name] = np.nanmean(np.array(x_label_kind_dictionary[kind_name]),axis=0)
            y_label_kind_dictionary[kind_name] = np.nanmean(np.array(y_label_kind_dictionary[kind_name]),axis=0)

            plt.figure()
            if kind_name == "difference":
                plt.imshow(label_kind_average_plot_dictionary[kind_name], vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
            else:
                plt.imshow(label_kind_average_plot_dictionary[kind_name], vmin=np.percentile(label_kind_average_plot_dictionary["data"], q=2),vmax=np.percentile(label_kind_average_plot_dictionary["data"], q=98), cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.title('Average {0}, T{1} Stars normalized by data star flux'.format(capital_kind_names[k], label[1:]))
            plt.savefig('{0}/stars_{1}_{2}_'.format(graph_directory, label, kind_name) + psf_type + '.png')

        plt.figure()
        for k, kind_name in enumerate(kind_names):
            plt.scatter(x_label_kind_dictionary[kind_name], y_label_kind_dictionary[kind_name], label=capital_kind_names[k])
        plt.legend()
        plt.title('Average Data, Model, and Difference for T{0} Stars normalized by data star flux'.format(label[1:]))
        plt.savefig('{0}/stars_{1}_radial_'.format(graph_directory, label) + psf_type + '.png')



        for l in [0, 1, 2]:
            try:
                label_kind_average_plot_dictionary = {}
                x_label_kind_dictionary = {}
                y_label_kind_dictionary = {}
                for kind_name in kind_names:
                    label_kind_average_plot_dictionary[kind_name] = []
                    x_label_kind_dictionary[kind_name] = []
                    y_label_kind_dictionary[kind_name] = []

                for exposure_i, exposure in enumerate(exposures):
                    try:
                        for kind_name in kind_names:
                            if not np.any(np.isnan(np.load("{0}/{1}_{2}_average_plot_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))):
                                label_kind_average_plot_dictionary[kind_name].append(np.load("{0}/{1}_{2}_average_plot_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))
                            if not np.any(np.isnan(np.load("{0}/x_{1}_{2}_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))):
                                x_label_kind_dictionary[kind_name].append(np.load("{0}/x_{1}_{2}_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))                 
                            if not np.any(np.isnan(np.load("{0}/y_{1}_{2}_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))):
                                y_label_kind_dictionary[kind_name].append(np.load("{0}/y_{1}_{2}_snr_level_{3}_{4}_".format(graph_values_directory, label, kind_name, l, exposure) + psf_type + ".npy"))
                    except:
                        pass

                for k, kind_name in enumerate(kind_names):
                    label_kind_average_plot_dictionary[kind_name] = np.nanmean(np.array(label_kind_average_plot_dictionary[kind_name]),axis=0)
                    x_label_kind_dictionary[kind_name] = np.nanmean(np.array(x_label_kind_dictionary[kind_name]),axis=0)
                    y_label_kind_dictionary[kind_name] = np.nanmean(np.array(y_label_kind_dictionary[kind_name]),axis=0)


                    plt.figure()
                    if kind_name == "difference":
                        plt.imshow(label_kind_average_plot_dictionary[kind_name], vmin=-0.005,vmax=0.005, cmap=plt.cm.RdBu_r)
                    else:
                        plt.imshow(label_kind_average_plot_dictionary[kind_name], vmin=np.percentile(label_kind_average_plot_dictionary["data"], q=2),vmax=np.percentile(label_kind_average_plot_dictionary["data"], q=98), cmap=plt.cm.RdBu_r)
                    plt.colorbar()
                    plt.title('Average {0}, T{1} Stars normalized by data star flux for snr level {2}'.format(capital_kind_names[k], label[1:], l))
                    plt.savefig('{0}/stars_{1}_{2}_snr_level_{3}_'.format(graph_directory, label, kind_name, l) + psf_type + '.png')

                plt.figure()
                for k, kind_name in enumerate(kind_names):
                    plt.scatter(x_label_kind_dictionary[kind_name], y_label_kind_dictionary[kind_name], label=capital_kind_names[k])
                plt.legend()
                plt.title('Average Data, Model, and Difference for T{0} Stars normalized by data star flux for snr level {1}'.format(label, l))
                plt.savefig('{0}/stars_{1}_radial_snr_level_{2}_'.format(graph_directory, label, l) + psf_type + '.png')
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
    make_call(band=band, psf_type=psf_type)
