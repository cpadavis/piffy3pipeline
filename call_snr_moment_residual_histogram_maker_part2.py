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



def make_call():
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_values_directory = "{0}/graph_values_npy_storage".format(core_directory)
    graph_directory = "{0}/multi_exposure_graphs/{1}_snr_moment_residual_histograms_averaged_across_exposures".format(core_directory, psf_type)
    terminal_command = os.system("mkdir {0}".format(graph_directory))
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    original_exposures = glob.glob("{0}/*".format(source_directory))
    original_exposures = [original_exposure.split("/")[-1] for original_exposure in original_exposures]
    if band=="all":
	exposures = original_exposures
    else:
	exposures = []
	for original_exposure in original_exposures:
	    skip=False
	    for index in range(1,63):
		try:
		    band_test_file = "{0}/{1}/psf_cat_{1}_{2}.fits".format(source_directory, original_exposure, index)
		    hdu = fits.open(band_test_file)
		    break
		except:
		    if index==62:
			skip = True
		    else:
		        pass
	    if skip==True:
		continue
	    try:
		band_test_file = "{0}/{1}/exp_psf_cat_{1}.fits".format(source_directory, original_exposure)
		hdu_c = fits.open(band_test_file)
		filter_name = hdu_c[1].data['band'][0][0]
		print(filter_name)
	    except:
		try:
    	    	    filter_name = hdu[3].data['band'][0]
		except:
		    continue
    	    if filter_name==band:
                exposures.append(original_exposure)  
	graph_directory = graph_directory + "/snr_moment_residual_histograms_just_for_filter_{0}".format(band)
	os.system("mkdir {0}".format(graph_directory))

    data_final_e0 = []
    model_final_e0 = []
    difference_final_e0 = []

    data_final_e1 = []
    model_final_e1 = []
    difference_final_e1 = []

    data_final_e2 = []
    model_final_e2 = []
    difference_final_e2 = []

    data_final_zeta1 = []
    model_final_zeta1 = []
    difference_final_zeta1 = []

    data_final_zeta2 = []
    model_final_zeta2 = []
    difference_final_zeta2 = []

    data_final_delta1 = []
    model_final_delta1 = []
    difference_final_delta1 = []

    data_final_delta2 = []
    model_final_delta2 = []
    difference_final_delta2 = []

    for exposure_i, exposure in enumerate(exposures):
        try:
    	    # for example, you could have psf_type="optatmo_const_gpvonkarman_meanified"
            data_final_e0.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[1])
            model_final_e0.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[8])        
            difference_final_e0.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[15])                

            data_final_e1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[2])
            model_final_e1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[9])        
            difference_final_e1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[16])              

            data_final_e2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[3])
            model_final_e2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[10])        
            difference_final_e2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[17])  



            data_final_zeta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[4])
            model_final_zeta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[11])        
            difference_final_zeta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[18])                

            data_final_zeta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[5])
            model_final_zeta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[12])        
            difference_final_zeta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[19])              

            data_final_delta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[6])
            model_final_delta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[13])        
            difference_final_delta1.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[20])        

            data_final_delta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[7])
            model_final_delta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[14])        
            difference_final_delta2.append(np.load("{0}/test_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[21])        
        except:
            pass

    final_snrs = np.array([5.0,15.0,25.0,35.0,45.0,55.0,65.0,75.0,85.0,95.0])
    print(data_final_e0) 
    data_final_e0 = np.nanmean(np.array(data_final_e0),axis=0)
    model_final_e0 = np.nanmean(np.array(model_final_e0),axis=0)
    difference_final_e0 = np.nanmean(np.array(difference_final_e0),axis=0)

    data_final_e1 = np.nanmean(np.array(data_final_e1),axis=0)
    model_final_e1 = np.nanmean(np.array(model_final_e1),axis=0)
    difference_final_e1 = np.nanmean(np.array(difference_final_e1),axis=0)

    data_final_e2 = np.nanmean(np.array(data_final_e2),axis=0)
    model_final_e2 = np.nanmean(np.array(model_final_e2),axis=0)
    difference_final_e2 = np.nanmean(np.array(difference_final_e2),axis=0)

    data_final_zeta1 = np.nanmean(np.array(data_final_zeta1),axis=0)
    model_final_zeta1 = np.nanmean(np.array(model_final_zeta1),axis=0)
    difference_final_zeta1 = np.nanmean(np.array(difference_final_zeta1),axis=0)

    data_final_zeta2 = np.nanmean(np.array(data_final_zeta2),axis=0)
    model_final_zeta2 = np.nanmean(np.array(model_final_zeta2),axis=0)
    difference_final_zeta2 = np.nanmean(np.array(difference_final_zeta2),axis=0)

    data_final_delta1 = np.nanmean(np.array(data_final_delta1),axis=0)
    model_final_delta1 = np.nanmean(np.array(model_final_delta1),axis=0)
    difference_final_delta1 = np.nanmean(np.array(difference_final_delta1),axis=0)

    data_final_delta2 = np.nanmean(np.array(data_final_delta2),axis=0)
    model_final_delta2 = np.nanmean(np.array(model_final_delta2),axis=0)
    difference_final_delta2 = np.nanmean(np.array(difference_final_delta2),axis=0)    

    print(final_snrs)
    print(data_final_e0)    


    plt.figure()
    plt.scatter(final_snrs,data_final_e0, label="data")
    plt.scatter(final_snrs,model_final_e0, label="model")
    plt.scatter(final_snrs,difference_final_e0, label="difference")
    plt.title("e0 across snrs")
    plt.legend()
    plt.savefig("{0}/test_e0_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_e1, label="data")
    plt.scatter(final_snrs,model_final_e1, label="model")
    plt.scatter(final_snrs,difference_final_e1, label="difference")
    plt.title("e1 across snrs")
    plt.legend()
    plt.savefig("{0}/test_e1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_e2, label="data")
    plt.scatter(final_snrs,model_final_e2, label="model")
    plt.scatter(final_snrs,difference_final_e2, label="difference")
    plt.title("e2 across snrs")
    plt.legend()
    plt.savefig("{0}/test_e2_across_snrs.png".format(graph_directory))    


    plt.figure()
    plt.scatter(final_snrs,data_final_zeta1, label="data")
    plt.scatter(final_snrs,model_final_zeta1, label="model")
    plt.scatter(final_snrs,difference_final_zeta1, label="difference")
    plt.title("zeta1 across snrs")
    plt.legend()
    plt.savefig("{0}/test_zeta1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_zeta2, label="data")
    plt.scatter(final_snrs,model_final_zeta2, label="model")
    plt.scatter(final_snrs,difference_final_zeta2, label="difference")
    plt.title("zeta2 across snrs")
    plt.legend()
    plt.savefig("{0}/test_zeta2_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_delta1, label="data")
    plt.scatter(final_snrs,model_final_delta1, label="model")
    plt.scatter(final_snrs,difference_final_delta1, label="difference")
    plt.title("delta1 across snrs")
    plt.legend()
    plt.savefig("{0}/test_delta1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_delta2, label="data")
    plt.scatter(final_snrs,model_final_delta2, label="model")
    plt.scatter(final_snrs,difference_final_delta2, label="difference")
    plt.title("delta2 across snrs")
    plt.legend()
    plt.savefig("{0}/test_delta2_across_snrs.png".format(graph_directory))  





    data_final_e0 = []
    model_final_e0 = []
    difference_final_e0 = []

    data_final_e1 = []
    model_final_e1 = []
    difference_final_e1 = []

    data_final_e2 = []
    model_final_e2 = []
    difference_final_e2 = []

    data_final_zeta1 = []
    model_final_zeta1 = []
    difference_final_zeta1 = []

    data_final_zeta2 = []
    model_final_zeta2 = []
    difference_final_zeta2 = []

    data_final_delta1 = []
    model_final_delta1 = []
    difference_final_delta1 = []

    data_final_delta2 = []
    model_final_delta2 = []
    difference_final_delta2 = []

    for exposure_i, exposure in enumerate(exposures):
        try:
            data_final_e0.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[1])
            model_final_e0.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[8])        
            difference_final_e0.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[15])                

            data_final_e1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[2])
            model_final_e1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[9])        
            difference_final_e1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[16])              

            data_final_e2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[3])
            model_final_e2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[10])        
            difference_final_e2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[17])  



            data_final_zeta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[4])
            model_final_zeta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[11])        
            difference_final_zeta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[18])                

            data_final_zeta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[5])
            model_final_zeta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[12])        
            difference_final_zeta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[19])              

            data_final_delta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[6])
            model_final_delta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[13])        
            difference_final_delta1.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[20])        

            data_final_delta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[7])
            model_final_delta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[14])        
            difference_final_delta2.append(np.load("{0}/train_{1}_information_{2}.npy".format(graph_values_directory, psf_type, exposure))[21])        
        except:
            pass

    final_snrs = np.array([5.0,15.0,25.0,35.0,45.0,55.0,65.0,75.0,85.0,95.0])
    print(data_final_e0) 
    data_final_e0 = np.nanmean(np.array(data_final_e0),axis=0)
    model_final_e0 = np.nanmean(np.array(model_final_e0),axis=0)
    difference_final_e0 = np.nanmean(np.array(difference_final_e0),axis=0)

    data_final_e1 = np.nanmean(np.array(data_final_e1),axis=0)
    model_final_e1 = np.nanmean(np.array(model_final_e1),axis=0)
    difference_final_e1 = np.nanmean(np.array(difference_final_e1),axis=0)

    data_final_e2 = np.nanmean(np.array(data_final_e2),axis=0)
    model_final_e2 = np.nanmean(np.array(model_final_e2),axis=0)
    difference_final_e2 = np.nanmean(np.array(difference_final_e2),axis=0)

    data_final_zeta1 = np.nanmean(np.array(data_final_zeta1),axis=0)
    model_final_zeta1 = np.nanmean(np.array(model_final_zeta1),axis=0)
    difference_final_zeta1 = np.nanmean(np.array(difference_final_zeta1),axis=0)

    data_final_zeta2 = np.nanmean(np.array(data_final_zeta2),axis=0)
    model_final_zeta2 = np.nanmean(np.array(model_final_zeta2),axis=0)
    difference_final_zeta2 = np.nanmean(np.array(difference_final_zeta2),axis=0)

    data_final_delta1 = np.nanmean(np.array(data_final_delta1),axis=0)
    model_final_delta1 = np.nanmean(np.array(model_final_delta1),axis=0)
    difference_final_delta1 = np.nanmean(np.array(difference_final_delta1),axis=0)

    data_final_delta2 = np.nanmean(np.array(data_final_delta2),axis=0)
    model_final_delta2 = np.nanmean(np.array(model_final_delta2),axis=0)
    difference_final_delta2 = np.nanmean(np.array(difference_final_delta2),axis=0)    

    print(final_snrs)
    print(data_final_e0)    


    plt.figure()
    plt.scatter(final_snrs,data_final_e0, label="data")
    plt.scatter(final_snrs,model_final_e0, label="model")
    plt.scatter(final_snrs,difference_final_e0, label="difference")
    plt.title("e0 across snrs")
    plt.legend()
    plt.savefig("{0}/train_e0_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_e1, label="data")
    plt.scatter(final_snrs,model_final_e1, label="model")
    plt.scatter(final_snrs,difference_final_e1, label="difference")
    plt.title("e1 across snrs")
    plt.legend()
    plt.savefig("{0}/train_e1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_e2, label="data")
    plt.scatter(final_snrs,model_final_e2, label="model")
    plt.scatter(final_snrs,difference_final_e2, label="difference")
    plt.title("e2 across snrs")
    plt.legend()
    plt.savefig("{0}/train_e2_across_snrs.png".format(graph_directory))    


    plt.figure()
    plt.scatter(final_snrs,data_final_zeta1, label="data")
    plt.scatter(final_snrs,model_final_zeta1, label="model")
    plt.scatter(final_snrs,difference_final_zeta1, label="difference")
    plt.title("zeta1 across snrs")
    plt.legend()
    plt.savefig("{0}/train_zeta1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_zeta2, label="data")
    plt.scatter(final_snrs,model_final_zeta2, label="model")
    plt.scatter(final_snrs,difference_final_zeta2, label="difference")
    plt.title("zeta2 across snrs")
    plt.legend()
    plt.savefig("{0}/train_zeta2_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_delta1, label="data")
    plt.scatter(final_snrs,model_final_delta1, label="model")
    plt.scatter(final_snrs,difference_final_delta1, label="difference")
    plt.title("delta1 across snrs")
    plt.legend()
    plt.savefig("{0}/train_delta1_across_snrs.png".format(graph_directory))

    plt.figure()
    plt.scatter(final_snrs,data_final_delta2, label="data")
    plt.scatter(final_snrs,model_final_delta2, label="model")
    plt.scatter(final_snrs,difference_final_delta2, label="difference")
    plt.title("delta2 across snrs")
    plt.legend()
    plt.savefig("{0}/train_delta2_across_snrs.png".format(graph_directory))  


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type')
    parser.add_argument('--band')
    options = parser.parse_args()
    psf_type = options.psf_type
    band = options.band
    kwargs = vars(options)
    del kwargs['psf_type']
    del kwargs['band']
    make_call()
