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



def make_graph(csv, psf_type, band, parameters):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_directory = "{0}/multi_exposure_graphs/{1}_aos_vs_piff_plots_averaged_across_exposures".format(core_directory, psf_type)
    os.system("mkdir {0}".format(graph_directory))
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    original_exposures = glob.glob("{0}/*".format(source_directory))
    original_exposures = [original_exposure.split("/")[-1] for original_exposure in original_exposures]
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
        graph_directory = graph_directory + "/aos_vs_piff_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    #parameters = ["r0oroptsize", "z4d", "z5d", "z5x", "z5y", "z6d", "z6x", "z6y", "z7d", "z7x", "z7y", "z8d", "z8x", "z8y", "z9d", "z10d", "z11d", "z14d", "z15d"]
    parameters = parameters.split("_")
    aos_parameter_list_dictionary = {}
    piff_parameter_list_dictionary = {}
    aos = pd.read_csv(csv)
    expid_list = aos['expid'].values  
    for parameter in parameters:
            if parameter == "r0oroptsize":
                aos_parameter_list_dictionary["r0oroptsize"] =  aos['rzero '].values
            else:
                aos_parameter_list_dictionary[parameter] = aos[parameter].values
    for parameter in parameters:
            piff_parameter_list_dictionary[parameter] = []
                   
    piff_key_dictionary = {}
    for parameter in parameters:
        if parameter == "r0oroptsize":
            piff_key_dictionary[parameter] = "size"
        elif parameter[-1]=="d":
            pupil_number = parameter[1:-1]
            focal_number = "001"
            if len(pupil_number) == 2:
                pupil_number = "0" + pupil_number
            if len(pupil_number) == 1:
                pupil_number = "00" + pupil_number
            piff_key_dictionary[parameter] = "zPupil{0}_zFocal{1}".format(pupil_number, focal_number)
        elif parameter[-1]=="x":
            pupil_number = parameter[1:-1]
            focal_number = "002"
            if len(pupil_number) == 2:
                pupil_number = "0" + pupil_number
            if len(pupil_number) == 1:
                pupil_number = "00" + pupil_number
            piff_key_dictionary[parameter] = "zPupil{0}_zFocal{1}".format(pupil_number, focal_number)
        elif parameter[-1]=="y":
            pupil_number = parameter[1:-1]
            focal_number = "003"
            if len(pupil_number) == 2:
                pupil_number = "0" + pupil_number
            if len(pupil_number) == 1:
                pupil_number = "00" + pupil_number
            piff_key_dictionary[parameter] = "zPupil{0}_zFocal{1}".format(pupil_number, focal_number)
            
    delete_list = []
    for exposure_i, exposure in enumerate(expid_list):
        try:
            # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
            psf = piff.read("{0}/00{1}/psf_{2}.piff".format(core_directory, exposure, psf_type))        
            optatmo_psf_kwargs = psf.optatmo_psf_kwargs        
            print(optatmo_psf_kwargs)
        except:
            delete_list.append(exposure_i)
            continue
        for parameter in parameters:
                piff_parameter_list_dictionary[parameter].append(optatmo_psf_kwargs[piff_key_dictionary[parameter]])        
    for entry in piff_parameter_list_dictionary:
        piff_parameter_list_dictionary[entry] = np.array(piff_parameter_list_dictionary[entry])

    for entry in aos_parameter_list_dictionary:        
        if entry[-1] == "x" or entry[-1] == "y":
            aos_parameter_list_dictionary[entry] = np.delete(aos_parameter_list_dictionary[entry], delete_list)*129.0
        else:
            aos_parameter_list_dictionary[entry] = np.delete(aos_parameter_list_dictionary[entry], delete_list)
    
    for parameter in parameters:
        try:
            plt.figure()
            plt.scatter(aos_parameter_list_dictionary[parameter], piff_parameter_list_dictionary[parameter])
            plt.xlabel("aos")
            plt.ylabel("piff")
            plt.title(parameter)
            plt.savefig("{0}/aos_vs_piff_{1}_{2}.png".format(graph_directory, parameter, psf_type))
        except:
            pass



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--psf_type')
    parser.add_argument('--band')
    parser.add_argument('--parameters')
    options = parser.parse_args()
    csv = options.csv
    psf_type = options.psf_type
    band = options.band
    parameters = options.parameters
    make_graph(csv=csv, psf_type=psf_type, band=band, parameters=parameters)
