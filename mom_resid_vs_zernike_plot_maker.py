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

#from zernike import Zernike
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip




def make_call(band, psf_type_difference, psf_type_zernike):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = core_directory + "/graph_values_npy_storage"
    graph_directory = "{0}/multi_exposure_graphs/{1}_difference_{2}_zernike_moment_residual_vs_zernike_plots_across_exposures".format(core_directory, psf_type_difference, psf_type_zernike)
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
        graph_directory = graph_directory + "/moment_residual_vs_zernike_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    
    #create dictionaries of saved moment residual and zernike data
    exposures = sorted(exposures)
    type_name_label_list_dictionary = {}
    type_names = ["de0", "de1", "de2", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "atmo_size", "atmo_g1", "atmo_g2"]
    for type_name in type_names:
        type_name_label_list_dictionary[type_name+"_test"] = []
        type_name_label_list_dictionary[type_name+"_train"] = []
    print("preparing to enter type_name for loop")
    for type_name in type_names:
        print("preparing to enter exposure for loop for type_name {0}".format(type_name))
        print("exposures: {0}".format(exposures))
        for exposure_i, exposure in enumerate(exposures):
            if int(exposure) < 229332: #TODO: find a way to make this work for more than one (or a few) exposure
                continue
            if int(exposure) > 229332:
                break
            try:
                directory = "{0}/00{1}".format(core_directory, exposure)
                for label in ["test", "train"]:
                    # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
                    filename_difference = "{0}/shapes_{1}_psf_{2}.h5".format(directory, label, psf_type_difference)
                    filename_zernike = "{0}/shapes_{1}_psf_{2}.h5".format(directory, label, psf_type_zernike)
                    shapes_difference = pd.read_hdf(filename_difference)
                    shapes_zernike = pd.read_hdf(filename_zernike)
                    if len(shapes_difference["de0"].values) != len(shapes_zernike["z04"].values):
                        print("failed for exposure {0} for {1} stars because difference and zernike arrays are a different length".format(exposure, label))
                        continue
                    if type_name in ["de0", "de1", "de2"]:
                       type_name_label_list_dictionary[type_name+"_"+label].append(shapes_difference[type_name].values)
                    if type_name in ["z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "atmo_size", "atmo_g1", "atmo_g2"]:
                       type_name_label_list_dictionary[type_name+"_"+label].append(shapes_zernike[type_name].values)
                    print("succeeded for exposure {0} for {1} stars".format(exposure, label))
                print("succeeded for exposure {0}".format(exposure))
            except:
                print("failed for exposure {0}".format(exposure))
    for type_name in type_names:
        print(type_name+"_test")
        print(type_name_label_list_dictionary[type_name+"_test"])
        type_name_label_list_dictionary[type_name+"_test"] = np.concatenate(type_name_label_list_dictionary[type_name+"_test"], axis=None)
        type_name_label_list_dictionary[type_name+"_train"] = np.concatenate(type_name_label_list_dictionary[type_name+"_train"], axis=None)

    #make graphs comparing moment residuals to zernikes
    difference_array_names = ["de0", "de1", "de2"]
    for difference_array_name in difference_array_names:
        for label in ["test", "train"]:
            difference_array = type_name_label_list_dictionary[difference_array_name+"_"+label]
            zernike_array_dictionary = {}
            zernike_array_dictionary["defocus"] = type_name_label_list_dictionary["z04_"+label]
            zernike_array_dictionary["astigmatism"] = np.sqrt(   np.square(type_name_label_list_dictionary["z05_"+label]) + np.square(type_name_label_list_dictionary["z06_"+label])   )
            zernike_array_dictionary["coma"] = np.sqrt(   np.square(type_name_label_list_dictionary["z07_"+label]) + np.square(type_name_label_list_dictionary["z08_"+label])   )
            zernike_array_dictionary["trefoil"] = np.sqrt(   np.square(type_name_label_list_dictionary["z09_"+label]) + np.square(type_name_label_list_dictionary["z10_"+label])   )
            zernike_array_dictionary["spherical"] = type_name_label_list_dictionary["z11_"+label]
            zernike_array_dictionary["atmo_size"] = type_name_label_list_dictionary["atmo_size_"+label]
            zernike_array_dictionary["atmo_g1"] = type_name_label_list_dictionary["atmo_g1_"+label]
            zernike_array_dictionary["atmo_g2"] = type_name_label_list_dictionary["atmo_g2_"+label]                        
            for zernike_array_dictionary_key in list(zernike_array_dictionary.keys()):
                plt.figure()
                plt.xlabel(zernike_array_dictionary_key)
                plt.ylabel(difference_array_name)
                plt.title("{0} vs {1}".format(zernike_array_dictionary_key,difference_array_name))
                plt.scatter(zernike_array_dictionary[zernike_array_dictionary_key],difference_array, alpha=0.3, s=2.0)
                if zernike_array_dictionary_key == "atmo_g1": #TODO: make limits not hardcoded but specifiable via argparse arguments
                    plt.xlim(-0.04,0.04)
                if zernike_array_dictionary_key == "atmo_g2":
                    plt.xlim(-0.075,0.075)
                elif zernike_array_dictionary_key == "atmo_size":
                    plt.xlim(-0.01,0.04)
                plt.ylim(-0.06,0.06)
                plt.savefig("{0}/{1}_vs_{2}_{3}_stars".format(graph_directory,zernike_array_dictionary_key,difference_array_name,label)+".png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--band')
    parser.add_argument('--psf_type_difference')
    parser.add_argument('--psf_type_zernike')
    options = parser.parse_args()
    band = options.band
    psf_type_difference = options.psf_type_difference
    psf_type_zernike = options.psf_type_zernike
    make_call(band=band, psf_type_difference=psf_type_difference, psf_type_zernike=psf_type_zernike)
