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
import subprocess
import glob

import lmfit
import galsim
import piff

#from zernike import Zernike
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from call_angular_moment_residual_plot_maker_part2 import find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory



def make_call(psf_type, number_of_snr_bins):
    core_directory, source_directory, exposures = find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory()
    for exposure_i, exposure in enumerate(exposures):
        try:
            # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
            terminal_command = "bsub -R rhel60 -o {0}/log_file_dump/{1}_call_snr_moment_residual_histogram_maker_{2}.txt python {0}/snr_moment_residual_histogram_maker.py --core_directory {0} --psf_type {1} --exposure {2} --number_of_snr_bins {3}".format(core_directory, psf_type, exposure, number_of_snr_bins)
            os.system(terminal_command)
            #subprocess.call(terminal_command.split(" "))
            print(terminal_command)
        except:
            pass



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type')
    parser.add_argument('--number_of_snr_bins', default=10)
    options = parser.parse_args()
    psf_type = options.psf_type
    make_call(psf_type=psf_type, number_of_snr_bins=number_of_snr_bins)
