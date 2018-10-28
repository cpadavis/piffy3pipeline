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
#from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip



def flag_exposure_outliers():
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    original_exposures = glob.glob("{0}/*".format(source_directory))
    original_exposures = [original_exposure.split("/")[-1] for original_exposure in original_exposures]
    exposures = original_exposures

    moments = ["e0", "e1", "e2"]
    #labels = ["optical", "optatmo_test", "optatmo_train"]
    quality_control_list_dictionary = {}
    for moment in moments:
        #for label in labels:
        quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)] = []
        quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)] = []
        quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)] = []
    exposures_without_all_metrics = []
    for exposure_i, exposure in enumerate(exposures):
        try:
            #print("exposure: {0}".format(exposure))
            directory = "{0}/00{1}".format(core_directory, exposure)
            for m, moment in enumerate(moments):
                #for l, label in enumerate(labels):
                number_of_outliers_test_value = np.load("{0}/number_of_outliers_optical.npy".format(directory))[m]
                pull_mean_test_value = np.load("{0}/pull_mean_optical.npy".format(directory))[m]
                pull_rms_test_value = np.load("{0}/pull_rms_optical.npy".format(directory))[m]
            for m, moment in enumerate(moments):
                #for l, label in enumerate(labels):
                    #print("m: {0}, moment: {1}, l: {2}, label: {3}".format(m, moment, l, label))
                    #print("number of outliers array:")
                    #print(np.load("{0}/number_of_outliers_{1}.npy".format(directory, label))[m])
                    #print("pull mean array:")
                    #print(np.load("{0}/pull_mean_{1}.npy".format(directory, label))[m])
                    #print("pull rms array:")
                    #print(np.load("{0}/pull_rms_{1}.npy".format(directory, label))[m])
                quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)].append(np.load("{0}/number_of_outliers_optical.npy".format(directory))[m])
                quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)].append(np.load("{0}/pull_mean_optical.npy".format(directory))[m])
                quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)].append(np.load("{0}/pull_rms_optical.npy".format(directory))[m])
        except:
            exposures_without_all_metrics.append(exposure)
    quality_control_list = []
    for m, moment in enumerate(moments):
        #for l, label in enumerate(labels):
        quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)] = np.array(quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)])
        quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)] = np.array(quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)])
        quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)] = np.array(quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)])
            
        plt.figure()
        plt.hist(quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)])
        plt.title("number_of_outliers_{0}_optical".format(moment))
        plt.savefig("{0}/number_of_outliers_{1}_optical.png".format(core_directory, moment))
        plt.figure()
        plt.hist(quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)])
        plt.title("pull_mean_{0}_optical".format(moment))
        plt.savefig("{0}/pull_mean_{1}_optical.png".format(core_directory, moment))
        plt.figure()
        plt.hist(quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)])
        plt.title("pull_rms_{0}_optical.png".format(moment))
        plt.savefig("{0}/pull_rms_{1}_optical.png".format(core_directory, moment))
            
        quality_control_list.append(quality_control_list_dictionary["number_of_outliers_{0}_optical".format(moment)])
        quality_control_list.append(quality_control_list_dictionary["pull_mean_{0}_optical".format(moment)])
        quality_control_list.append(quality_control_list_dictionary["pull_rms_{0}_optical".format(moment)])
    print(quality_control_list_dictionary)
    #print(quality_control_list)
    quality_control_array = np.column_stack(quality_control_list)
    medians = np.nanmedian(quality_control_array, axis=0)
    madxs = np.abs(quality_control_array-medians)
    mads = np.nanmedian(madxs, axis=0)
    conds_mad = (np.all(madxs <= 6.0 * mads, axis=1))
    exposures_with_all_metrics = []
    for exposure in exposures:
        if exposure not in exposures_without_all_metrics:
            exposures_with_all_metrics.append(exposure)
    acceptable_exposures = np.array(exposures_with_all_metrics)[conds_mad]
    print("acceptable_exposures: ")
    print(acceptable_exposures)
    np.save("{0}/acceptable_exposures.npy".format(core_directory),acceptable_exposures)


if __name__ == '__main__':
    flag_exposure_outliers()
