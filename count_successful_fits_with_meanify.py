from __future__ import print_function, division

# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
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
from call_angular_moment_residual_plot_maker_part2 import find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory


def count_successful_fits_with_meanify():
    core_directory, source_directory, exposures = find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory()
    acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
    print("number of acceptable exposures: {0}".format(len(acceptable_exposures)))
    number_of_successful_fits_with_meanify = 0
    for exposure_i, exposure in enumerate(acceptable_exposures):
        try:
            directory = "{0}/00{1}".format(core_directory, exposure)
            shapes = piff.read("{0}/psf_optatmo_const_gpvonkarman_meanified.piff".format(directory))
            number_of_successful_fits_with_meanify = number_of_successful_fits_with_meanify + 1
            #if os.path.isfile("{0}/psf_optatmo_const_gpvonkarman_meanified.piff".format(directory)):
            #    number_of_successful_fits_with_meanify = number_of_successful_fits_with_meanify + 1
            #else:
            #    print("failed fit for {0}".format(exposure))
        except:
            print("failed fit for {0}".format(exposure))
    print("number_of_successful_fits_with_meanify: {0}".format(number_of_successful_fits_with_meanify))
    number_of_successful_h5_files_with_meanify = 0
    for exposure_i, exposure in enumerate(acceptable_exposures):
        try:
            directory = "{0}/00{1}".format(core_directory, exposure)
            shapes = pd.read_hdf("{0}/shapes_test_psf_optatmo_const_gpvonkarman_meanified.h5".format(directory))
            number_of_successful_h5_files_with_meanify = number_of_successful_h5_files_with_meanify + 1
        except:
            print("failed h5 file for {0}".format(exposure))
    print("number_of_successful_h5_files_with_meanify: {0}".format(number_of_successful_h5_files_with_meanify))
        





if __name__ == '__main__':
    count_successful_fits_with_meanify()
