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
from call_angular_moment_residual_plot_maker_part2 import find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory



def make_call(psf_type, band):
    core_directory, source_directory, very_original_exposures, graph_values_directory, graph_directory = find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory(psf_type=psf_type, graph_type="snr_moment_residual_plots_averaged_across_exposures", set_up_graph_directory_and_graph_values_directory=True)
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
        graph_directory = graph_directory + "/snr_moment_residual_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    for label in ["test", "train"]: #here, aggregate angular moment residual graphs are made
        
        moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
        kinds = ["data_", "model_", "d"]
        kind_final_moment_dictionary = {}
        
        for moment in moments:
            for kind in kinds:   
                kind_final_moment_dictionary['{0}{1}'.format(kind,moment)] = []

        got_number_of_snr_bins = False
        number_of_snr_bins = 0
        for exposure_i, exposure in enumerate(exposures):
            try:
                for m, moment in enumerate(moments):
                    for k, kind in enumerate(kinds):
                        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
                        kind_final_moment_dictionary['{0}{1}'.format(kind,moment)].append(np.load("{0}/{1}_{2}_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure))[m+k*len(moments)])                 
            except:
                pass
            if got_number_of_snr_bins == False:
                number_of_snr_bins = np.load("{0}/{1}_{2}_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure)).shape[1]
                got_number_of_snr_bins = True

        bin_width = 360.0 / number_of_snr_bins
        half_bin_width = bin_width / 2.0
        final_snrs_edges = np.linspace(0.0, 100.0, num=number_of_snr_bins)
        final_snrs = final_snrs_edges[1:] - half_bin_width
        print(kind_final_moment_dictionary['data_e0']) 

        for entry in kind_final_moment_dictionary:
            kind_final_moment_dictionary[entry] = np.nanmean(np.array(kind_final_moment_dictionary[entry]),axis=0) 

        print(final_snrs)
        print(kind_final_moment_dictionary['data_e0'])    

        for moment in moments:
            plt.figure()
            plt.scatter(final_snrs,kind_final_moment_dictionary['data_{0}'.format(moment)], label="data")
            plt.scatter(final_snrs,kind_final_moment_dictionary['model_{0}'.format(moment)], label="model")
            plt.scatter(final_snrs,kind_final_moment_dictionary['d{0}'.format(moment)], label="difference")
            plt.title("{0} snrs".format(moment))
            plt.legend()
            plt.savefig("{0}/{1}_{2}_across_snrs.png".format(graph_directory, label, moment))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type')
    parser.add_argument('--band')
    options = parser.parse_args()
    psf_type = options.psf_type
    band = options.band
    make_call(psf_type=psf_type, band=band)
