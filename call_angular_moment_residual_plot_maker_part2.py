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



def find_filter_name_or_skip(source_directory, exposure):
    #This function finds the filter of the given exposure
    filter_name_and_skip_dictionary = {}
    filter_name_and_skip_dictionary['skip'] = False
    for index in range(1,63):
        try:
            band_test_file = "{0}/{1}/psf_cat_{1}_{2}.fits".format(source_directory, exposure, index)
            print("first band_test_file: {0}".format(band_test_file))
            hdu = fits.open(band_test_file)
            print("hdu found!")
            break
        except:
            print("failed to find hdu")
            if index==62:
                print("failed to find any hdu")
                filter_name_and_skip_dictionary['skip'] = True
                return filter_name_and_skip_dictionary
            else:
                pass
    try:
        band_test_file = "{0}/{1}/exp_psf_cat_{1}.fits".format(source_directory, exposure)
        print("second band_test_file: {0}".format(band_test_file))
        hdu_c = fits.open(band_test_file)
        print("hdu_c found!")
        filter_name = hdu_c[2].data['band'][0][0]
        print(filter_name)
        filter_name_and_skip_dictionary['filter_name'] = filter_name
    except:
        print("failed when dealing with hdu_c somehow")
        try:
            print("preparing to get filter_name from hdu")
            filter_name = hdu[3].data['band'][0]
            print(filter_name)
            filter_name_and_skip_dictionary['filter_name'] = filter_name
        except:
            print("failed_everything")
            filter_name_and_skip_dictionary['skip'] = True
            return filter_name_and_skip_dictionary
    try:
        hdu.close()
    except:
        pass
    try:
        hdu_c.close()
    except:
        pass
    print("finished closing files")    
    return filter_name_and_skip_dictionary
    

def make_call(psf_type, band):
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)
    graph_directory = "{0}/multi_exposure_graphs/{1}_angular_moment_residual_plots_averaged_across_exposures".format(core_directory, psf_type)
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
            print("original_exposure: {0}".format(original_exposure))
            filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=original_exposure)
            print("filter_name_and_skip_dictionary: {0}".format(filter_name_and_skip_dictionary))
            if filter_name_and_skip_dictionary['skip'] == True:
                print("skip is True!")
                continue
            else:
                print("skip is False!")
                filter_name = filter_name_and_skip_dictionary['filter_name'] 
                print("filter_name: {0}".format(filter_name))                  
            if filter_name in band:
                print("filter_name in band!")
                exposures.append(original_exposure)  
        graph_directory = graph_directory + "/angular_moment_residual_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))

    print("preparing to enter label for loop")
    for label in ["test", "train"]: #here, aggregate angular moment residual graphs are made
        print("entered label for loop")

        moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
        kinds = ["data_", "model_", "d"]
        kind_final_moment_dictionary = {}
        
        print("preparing to fill up kind_final_moment_dictionary")
        for moment in moments:
            for kind in kinds:   
                kind_final_moment_dictionary['{0}{1}'.format(kind,moment)] = []
        print("finished filling up kind_final_moment_dictionary")
        print("kind_final_moment_dictionary: {0}".format(kind_final_moment_dictionary))

        print("preparing to run through exposures")
        for exposure_i, exposure in enumerate(exposures):
            try:
                print("exposure_i, exposure: {0}, {1}".format(exposure_i, exposure))
                for m, moment in enumerate(moments):
                    print("m, moment: {0}, {1}".format(m, moment))
                    for k, kind in enumerate(kinds):
                        print("k, kind: {0}, {1}".format(k, kind))
                        # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
                        kind_final_moment_dictionary['{0}{1}'.format(kind,moment)].append(np.load("{0}/{1}_{2}_angular_information_{3}.npy".format(graph_values_directory, label, psf_type, exposure))[m+k*len(moments)])   
                        print("just appended")
                        print(kind_final_moment_dictionary['{0}{1}'.format(kind,moment)])  
            except:
                pass

        final_angles = np.arange(5, 356, 10, dtype=np.float)
        print(kind_final_moment_dictionary['data_e0']) 

        print("preparing to take mean")
        for entry in kind_final_moment_dictionary:
            kind_final_moment_dictionary[entry] = np.nanmean(np.array(kind_final_moment_dictionary[entry]),axis=0) 
        print("finished taking mean")

        print(final_angles)
        print(kind_final_moment_dictionary['data_e0'])    


        print("preparing to enter final for loop for making graphs for various moments")
        for moment in moments:
            print("entered final for loop for making graphs for various moments")
            plt.figure()
            plt.scatter(final_angles,kind_final_moment_dictionary['data_{0}'.format(moment)], label="data")
            plt.scatter(final_angles,kind_final_moment_dictionary['model_{0}'.format(moment)], label="model")
            plt.scatter(final_angles,kind_final_moment_dictionary['d{0}'.format(moment)], label="difference")
            plt.title("{0} across angles".format(moment))
            plt.legend()
            plt.savefig("{0}/{1}_{2}_across_angles.png".format(graph_directory, label, moment))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type')
    parser.add_argument('--band')
    options = parser.parse_args()
    psf_type = options.psf_type
    band = options.band
    make_call(psf_type=psf_type, band=band)
