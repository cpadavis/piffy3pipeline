from __future__ import print_function, division

# fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import copy

import os
import fitsio
import galsim
import piff
import sys

from piff.util import hsm_error, measure_snr

def make_pngs(directory, label, information, moments):

    fig, axss = plt.subplots(nrows=4, ncols=3, figsize=(4 * 4, 4 * 4), squeeze=False)
    # left column gets the Y coordinate label
    for axs in axss:
        for ax in axs:
            ax.set(xlabel='angle', ylabel='average moment')

    for i in range(0,4):
        for j in range(0,3):
            if i == 3 and j > 0:
                pass
            else:
                index = i*3 + j
                axss[i][j].scatter(np.arange(5, 356, 10, dtype=np.float), information[index], label="data")
                axss[i][j].scatter(np.arange(5, 356, 10, dtype=np.float), information[index+len(moments)], label="model")
                axss[i][j].scatter(np.arange(5, 356, 10, dtype=np.float), information[index+2*len(moments)], label="difference")
                axss[i][j].set_title(moments[index])
                axss[i][j].legend()
    plt.tight_layout()
    fig.savefig("{0}/{1}_{2}_across_angles.png".format(directory, label, psf_type))

def make_plots(exposure, core_directory, psf_type):

    directory = "{0}/00{1}".format(core_directory, exposure)
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)

    for label in ["test", "train"]:
        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
        filename = "{0}/shapes_{1}_psf_{2}.h5".format(directory, label, psf_type)
        shapes = pd.read_hdf(filename)

        u = shapes['u'].values
        v = shapes['v'].values
        angles = np.degrees(np.arctan2(v,u))
        angles[angles < 0.0] = angles[angles < 0.0] + 360.0

        moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
        information = np.empty([3*len(moments),36])
        kinds = ["data_", "model_", "d"]
        for m, moment in enumerate(moments):
            for k, kind in enumerate(kinds):
                kind_moment = shapes['{0}{1}'.format(kind,moment)].values
                kind_final_moment, bin_edges, binnumber = stats.binned_statistic(x=angles,values=kind_moment, statistic="mean",bins=36, range=(0.0,360.0))   
                information[m+k*len(moments)] = kind_final_moment

        np.save("{0}/{1}_{2}_angular_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure),information)
        make_pngs(directory=directory, label=label, information=information, moments=moments)
  


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exposure')
    parser.add_argument('--core_directory')
    parser.add_argument('--psf_type')
    options = parser.parse_args()
    exposure = options.exposure
    core_directory = options.core_directory
    psf_type = options.psf_type
    make_plots(exposure=exposure, core_directory=core_directory, psf_type=psf_type)
