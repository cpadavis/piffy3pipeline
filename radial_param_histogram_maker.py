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
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip




def make_call(band):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = core_directory + "/graph_values_npy_storage"
    graph_directory = "{0}/multi_exposure_graphs/radial_param_histogram_plots_across_exposures".format(core_directory)
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
        graph_directory = graph_directory + "/radial_param_histogram_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    
    #load up the radial data
    opt_L0s = []
    opt_sizes = []
    z4ds = []
    z11ds = []
    for exposure_i, exposure in enumerate(exposures):
        try:
            load_string = "{0}/00{1}/optatmo_psf_kwargs.npy".format(core_directory,exposure)
            optatmo_psf_kwargs = np.load(load_string)
            optatmo_psf_kwargs_item = optatmo_psf_kwargs.item()
            opt_L0 = optatmo_psf_kwargs_item["L0"]
            opt_size = optatmo_psf_kwargs_item["size"]
            z4d = optatmo_psf_kwargs_item["zPupil004_zFocal001"]
            z11d = optatmo_psf_kwargs_item["zPupil011_zFocal001"]
            opt_L0s.append(opt_L0)
            opt_sizes.append(opt_size)
            z4ds.append(z4d)
            z11ds.append(z11ds)
        except:
            pass

    #make the graphs
    plt.figure()
    plt.hist(opt_L0s)
    plt.title('opt_L0 across exposures')
    plt.savefig('{0}/opt_L0_across_exposures'.format(graph_directory) + '.png')

    plt.figure()
    plt.hist(opt_sizes)
    plt.title('opt_size across exposures')
    plt.savefig('{0}/opt_size_across_exposures'.format(graph_directory) + '.png')

    plt.figure()
    plt.hist(z4ds)
    plt.title('z4d across exposures')
    plt.savefig('{0}/z4d_across_exposures'.format(graph_directory) + '.png')

    plt.figure()
    plt.hist(z11ds)
    plt.title('z11d across exposures')
    plt.savefig('{0}/z11d_across_exposures'.format(graph_directory) + '.png')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--band')
    options = parser.parse_args()
    band = options.band
    make_call(band=band)
