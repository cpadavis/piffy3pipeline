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
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip, find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory




def make_call(band):
    core_directory, source_directory, very_original_exposures, graph_values_directory, graph_directory = find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory(psf_type="anytype", graph_type="opt_L0_plots_across_exposures", set_up_graph_directory_and_graph_values_directory=True)
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

    
    #collect data on opt_L0s
    opt_L0s = []
    exposures_with_opt_L0 = []
    for exposure_i, exposure in enumerate(exposures):
        try:
            load_string = "{0}/00{1}/optatmo_psf_kwargs.npy".format(core_directory,exposure)
            optatmo_psf_kwargs = np.load(load_string)
            optatmo_psf_kwargs_item = optatmo_psf_kwargs.item()
            opt_L0 = optatmo_psf_kwargs_item["L0"]
            opt_L0s.append(opt_L0)
            exposures_with_opt_L0.append(int(exposure))
        except:
            pass

    #graph the opt_L0s
    plt.figure()
    plt.scatter(exposures_with_opt_L0, opt_L0s)
    plt.title('opt_L0 across exposures')
    plt.savefig('{0}/opt_L0_across_exposures'.format(graph_directory) + '.png')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--band')
    options = parser.parse_args()
    band = options.band
    make_call(band=band)
