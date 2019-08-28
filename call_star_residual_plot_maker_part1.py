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



def make_call(psf_type):
    core_directory, source_directory, exposures = find_core_directory_source_directory_glob_exposures_and_possibly_set_up_graph_directory_and_graph_values_directory()
    for exposure_i, exposure in enumerate(exposures):
        try:
            # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
            terminal_command = "bsub -W 300 -R rhel60 -o {0}/log_file_dump/{1}_call_star_residual_plot_maker_{2}.txt python {0}/star_residual_plot_maker.py --core_directory {0} --psf_type {1} --exposure {2}".format(core_directory, psf_type, exposure)
            os.system(terminal_command)
            #subprocess.call(terminal_command.split(" "))
            print(terminal_command)
        except(KeyboardInterrupt, SystemExit):
            raise
        else:
            pass



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type')
    options = parser.parse_args()
    psf_type = options.psf_type
    make_call(psf_type=psf_type)
