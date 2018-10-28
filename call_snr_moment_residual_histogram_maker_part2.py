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



def make_call(psf_type, band):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)
    graph_directory = "{0}/multi_exposure_graphs/{1}_snr_moment_residual_histograms_averaged_across_exposures".format(core_directory, psf_type)
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
        graph_directory = graph_directory + "/snr_moment_residual_histograms_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    for label in ["test", "train"]:
        
        moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"]
        kinds = ["data_", "model_", "d"]
        kind_final_moment_dictionary = {}
        
        for moment in moments:
            for kind in kinds:   
                kind_final_moment_dictionary['{0}{1}'.format(kind,moment)] = []

        for exposure_i, exposure in enumerate(exposures):
            try:
                for m, moment in enumerate(moments):
                    for k, kind in enumerate(kinds):
                        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
                        kind_final_moment_dictionary['{0}{1}'.format(kind,moment)].append(np.load("{0}/{1}_{2}_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure))[m+k*len(moments)])                 
            except:
                pass

        final_snrs = np.arange(5, 96, 10, dtype=np.float)
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
