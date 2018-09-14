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

from piff.util import hsm_error, hsm_higher_order, measure_snr

def make_pngs(directory, label, information):

    fig, axss = plt.subplots(nrows=3, ncols=3, figsize=(4 * 4, 4 * 4), squeeze=False)
    # left column gets the Y coordinate label
    for axs in axss:
        for ax in axs:
            ax.set(xlabel='angle', ylabel='average moment')

    moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"]
    for i in range(0,3):
        for j in range(0,3):
            if i==2 and j>0:
                pass
            else:
                index = i*3 + j
                axss[i][j].scatter(information[0], information[index+1], label="data")
                axss[i][j].scatter(information[0], information[index+8], label="model")
                axss[i][j].scatter(information[0], information[index+15], label="difference")
                axss[i][j].set_title(moments[index])
                axss[i][j].legend()
    plt.tight_layout()
    fig.savefig("{0}/{1}_{2}_across_angles.png".format(directory, label, psf_type))

def make_plots():

    directory = "{0}/00{1}".format(core_directory, exposure)
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)



    for label in ["test", "train"]:
        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
        filename = "{0}/shapes_{1}_psf_{2}.h5".format(directory, label, psf_type)

        shapes = pd.read_hdf(filename)

        u = np.array(shapes['u'].tolist())
        v = np.array(shapes['v'].tolist())
        raw_angles = np.degrees(np.arctan2(v,u))
        angles = []
        for raw_angle in raw_angles:
            if raw_angle < 0:
                angle = raw_angle + 360.0
            else:
                angle = raw_angle
            angles.append(angle)
        angles = np.array(angles)

        data_e0 = np.array(shapes['data_e0'].tolist())
        model_e0 = np.array(shapes['model_e0'].tolist())
        difference_e0 = np.array(shapes['de0'].tolist())

        data_e1 = np.array(shapes['data_e1'].tolist())
        model_e1 = np.array(shapes['model_e1'].tolist())
        difference_e1 = np.array(shapes['de1'].tolist())

        data_e2 = np.array(shapes['data_e2'].tolist())
        model_e2 = np.array(shapes['model_e2'].tolist())
        difference_e2 = np.array(shapes['de2'].tolist())


        data_zeta1 = np.array(shapes['data_zeta1'].tolist())
        model_zeta1 = np.array(shapes['model_zeta1'].tolist())
        difference_zeta1 = np.array(shapes['dzeta1'].tolist())

        data_zeta2 = np.array(shapes['data_zeta2'].tolist())
        model_zeta2 = np.array(shapes['model_zeta2'].tolist())
        difference_zeta2 = np.array(shapes['dzeta2'].tolist())

        data_delta1 = np.array(shapes['data_delta1'].tolist())
        model_delta1 = np.array(shapes['model_delta1'].tolist())
        difference_delta1 = np.array(shapes['ddelta1'].tolist())

        data_delta2 = np.array(shapes['data_delta2'].tolist())
        model_delta2 = np.array(shapes['model_delta2'].tolist())
        difference_delta2 = np.array(shapes['ddelta2'].tolist())

        information = np.empty([22,36])
        information[0] = np.array([5.0,15.0,25.0,35.0,45.0,55.0,65.0,75.0,85.0,95.0,105.0,115.0,125.0,135.0,145.0,155.0,165.0,175.0,185.0,195.0,205.0,215.0,225.0,235.0,245.0,255.0,265.0,275.0,285.0,295.0,305.0,315.0,325.0,335.0,345.0,355.0])

        data_final_e0, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_e0, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_e1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_e1, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_e2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_e2, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_zeta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_zeta1, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_zeta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_zeta2, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_delta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_delta1, statistic="mean",bins=36, range=(0.0,360.0))
        data_final_delta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=data_delta2, statistic="mean",bins=36, range=(0.0,360.0))

        model_final_e0, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_e0, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_e1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_e1, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_e2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_e2, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_zeta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_zeta1, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_zeta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_zeta2, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_delta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_delta1, statistic="mean",bins=36, range=(0.0,360.0))
        model_final_delta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=model_delta2, statistic="mean",bins=36, range=(0.0,360.0))

        difference_final_e0, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_e0, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_e1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_e1, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_e2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_e2, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_zeta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_zeta1, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_zeta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_zeta2, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_delta1, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_delta1, statistic="mean",bins=36, range=(0.0,360.0))
        difference_final_delta2, bin_edges, binnumber = stats.binned_statistic(x=angles,values=difference_delta2, statistic="mean",bins=36, range=(0.0,360.0))


        information[1] = data_final_e0
        information[2] = data_final_e1
        information[3] = data_final_e2
        information[4] = data_final_zeta1
        information[5] = data_final_zeta2
        information[6] = data_final_delta1
        information[7] = data_final_delta2

        information[8] = model_final_e0
        information[9] = model_final_e1
        information[10] = model_final_e2
        information[11] = model_final_zeta1
        information[12] = model_final_zeta2
        information[13] = model_final_delta1
        information[14] = model_final_delta2

        information[15] = difference_final_e0
        information[16] = difference_final_e1
        information[17] = difference_final_e2
        information[18] = difference_final_zeta1
        information[19] = difference_final_zeta2
        information[20] = difference_final_delta1
        information[21] = difference_final_delta2

        np.save("{0}/{1}_{2}_angular_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure),information)

        make_pngs(directory=directory, label=label, information=information)
  


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
    kwargs = vars(options)
    del kwargs['exposure']
    del kwargs['core_directory']
    del kwargs['psf_type']
    make_plots()
