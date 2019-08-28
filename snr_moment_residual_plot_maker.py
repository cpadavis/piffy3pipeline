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

def make_pngs(directory, label, information, moments, bin_centers):
    #This function graphs the moments' data, model, and difference values across various snrs. Note that these snrs are after rescaling down stars with snr > 100.
    fig, axss = plt.subplots(nrows=4, ncols=3, figsize=(4 * 4, 4 * 4), squeeze=False)
    # left column gets the Y coordinate label
    for axs in axss:
        for ax in axs:
            ax.set(xlabel='snr', ylabel='average moment')

    for i in range(0,4):
        for j in range(0,3):
            if i == 3 and j > 0:
                pass
            else:        
                index = i*3 + j
                axss[i][j].scatter(bin_centers, information[index], label="data")
                axss[i][j].scatter(bin_centers, information[index+len(moments)], label="model")
                axss[i][j].scatter(bin_centers, information[index+2*len(moments)], label="difference")
                axss[i][j].set_title(moments[index])
                axss[i][j].legend()
    plt.tight_layout()
    fig.savefig("{0}/{1}_{2}_across_snrs.png".format(directory, label, psf_type))

def make_plots(exposure, core_directory, psf_type, number_of_snr_bins):

    directory = "{0}/00{1}".format(core_directory, exposure)
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)

    for label in ["test", "train"]:
        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
        filename = "{0}/shapes_{1}_psf_{2}.h5".format(directory, label, psf_type)
        shapes = pd.read_hdf(filename)

        snrs = shapes['snr'].values

        moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
        information = np.empty([3*len(moments),number_of_snr_bins])
        kinds = ["data_", "model_", "d"]
        bin_centers = np.empty(number_of_angular_bins)
        for m, moment in enumerate(moments):
            for k, kind in enumerate(kinds):
                kind_moment = shapes['{0}{1}'.format(kind,moment)].values
                kind_final_moment, bin_edges, binnumber = stats.binned_statistic(x=snrs,values=kind_moment, statistic="mean",bins=number_of_snr_bins, range=(0.0,100.0)) #Here the average moment values inside snr bins are computed.  
                if m == 0 and k == 0:
                    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2.0
                information[m+k*len(moments)] = kind_final_moment

        np.save("{0}/{1}_{2}_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure),information)
        make_pngs(directory=directory, label=label, information=information, moments=moments, bin_centers=bin_centers)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exposure')
    parser.add_argument('--core_directory')
    parser.add_argument('--psf_type')
    parser.add_argument('--number_of_snr_bins', default=10)
    options = parser.parse_args()
    exposure = options.exposure
    core_directory = options.core_directory
    psf_type = options.psf_type
    make_plots(exposure=exposure, core_directory=core_directory, psf_type=psf_type, number_of_snr_bins=number_of_snr_bins)
