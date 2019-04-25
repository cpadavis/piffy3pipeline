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




def make_call(band):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = core_directory + "/graph_values_npy_storage"
    graph_directory = "{0}/multi_exposure_graphs/opt_L0_plots_across_exposures".format(core_directory)
    os.system("mkdir {0}".format(graph_directory))
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    very_original_exposures = glob.glob("{0}/*".format(source_directory))
    very_original_exposures = [very_original_exposure.split("/")[-1] for very_original_exposure in very_original_exposures]
    try:
        acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
        original_exposures = []
        for very_original_exposure in very_original_exposures:
            if very_original_exposure in acceptable_exposures:
                original_exposures.append(very_original_exposure)
    except:
        original_exposures = very_original_exposures
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
        graph_directory = graph_directory + "/opt_L0_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    

    opt_L0s = []
    exposures_with_opt_L0 = []
    for exposure_i, exposure in enumerate(exposures):
        try:
            #print("preparing to get load_string")
            load_string = "{0}/00{1}/optatmo_psf_kwargs.npy".format(core_directory,exposure)
            #print("load_string: {0}".format(load_string))
            #print("getting optatmo_psf_kwargs")
            optatmo_psf_kwargs = np.load(load_string)
            #print("getting item")
            optatmo_psf_kwargs_item = optatmo_psf_kwargs.item()
            #print("getting opt_L0 from item")
            #print(optatmo_psf_kwargs_item)
            opt_L0 = optatmo_psf_kwargs_item["L0"]
            #print("opt_L0: {0}".format(opt_L0))
            opt_L0s.append(opt_L0)
            #print("appended opt_L0")
            exposures_with_opt_L0.append(int(exposure))
            #print("appended exposure")
        except:
            pass

    #print("opt_L0s: {0}".format(opt_L0s))
    #print("exposures_with_opt_L0: {0}".format(exposures_with_opt_L0))
    plt.figure()
    plt.scatter(exposures_with_opt_L0, opt_L0s)
    #plt.ylim(0.0, 200.0)
    plt.title('opt_L0 across exposures')
    plt.savefig('{0}/opt_L0_across_exposures'.format(graph_directory) + '.png')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--band')
    options = parser.parse_args()
    band = options.band
    make_call(band=band)
