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



def make_graph():
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    graph_directory = "{0}/multi_exposure_graphs/{1}_aos_vs_piff_plots_averaged_across_exposures".format(core_directory, psf_type)
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
            if filter_name in band:
                exposures.append(original_exposure)  
        graph_directory = graph_directory + "/aos_vs_piff_plots_just_for_filter_{0}".format(band)
        os.system("mkdir {0}".format(graph_directory))


    aos = pd.read_csv(csv)
    expid_list = np.array(aos['expid'].tolist()) 
    aos_z4d_list = np.array(aos['z4d'].tolist())
    aos_z5d_list = np.array(aos['z5d'].tolist())
    aos_z5x_list = np.array(aos['z5x'].tolist())
    aos_z5y_list = np.array(aos['z5y'].tolist())
    aos_z6d_list = np.array(aos['z6d'].tolist())
    aos_z6x_list = np.array(aos['z6x'].tolist())
    aos_z6y_list = np.array(aos['z6y'].tolist())
    aos_z7d_list = np.array(aos['z7d'].tolist())
    aos_z7x_list = np.array(aos['z7x'].tolist())
    aos_z7y_list = np.array(aos['z7y'].tolist())
    aos_z8d_list = np.array(aos['z8d'].tolist())
    aos_z8x_list = np.array(aos['z8x'].tolist())
    aos_z8y_list = np.array(aos['z8y'].tolist())
    aos_z9d_list = np.array(aos['z9d'].tolist())
    aos_z10d_list = np.array(aos['z10d'].tolist())
    aos_z11d_list = np.array(aos['z11d'].tolist())
    try:
        aos_z14d_list = np.array(aos['z14d'].tolist())
        aos_z15d_list = np.array(aos['z15d'].tolist())
    except:
        pass
    aos_r0_list = np.array(aos['rzero '].tolist())

    piff_z4d_list = []
    piff_z5d_list = []
    piff_z5x_list = []
    piff_z5y_list = []
    piff_z6d_list = []
    piff_z6x_list = []
    piff_z6y_list = []
    piff_z7d_list = []
    piff_z7x_list = []
    piff_z7y_list = []
    piff_z8d_list = []
    piff_z8x_list = []
    piff_z8y_list = []
    piff_z9d_list = []
    piff_z10d_list = []
    piff_z11d_list = []
    try:
        piff_z14d_list = []
        piff_z15d_list = []
    except:
        pass
    piff_opt_size_list = []


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
        piff_z4d_list.append(optatmo_psf_kwargs['zPupil004_zFocal001'])
        piff_z5d_list.append(optatmo_psf_kwargs['zPupil005_zFocal001'])
        piff_z5x_list.append(optatmo_psf_kwargs['zPupil005_zFocal002'])
        piff_z5y_list.append(optatmo_psf_kwargs['zPupil005_zFocal003'])
        piff_z6d_list.append(optatmo_psf_kwargs['zPupil006_zFocal001'])
        piff_z6x_list.append(optatmo_psf_kwargs['zPupil006_zFocal002'])
        piff_z6y_list.append(optatmo_psf_kwargs['zPupil006_zFocal003'])
        piff_z7d_list.append(optatmo_psf_kwargs['zPupil007_zFocal001'])
        piff_z7x_list.append(optatmo_psf_kwargs['zPupil007_zFocal002'])
        piff_z7y_list.append(optatmo_psf_kwargs['zPupil007_zFocal003'])
        piff_z8d_list.append(optatmo_psf_kwargs['zPupil008_zFocal001'])
        piff_z8x_list.append(optatmo_psf_kwargs['zPupil008_zFocal002'])
        piff_z8y_list.append(optatmo_psf_kwargs['zPupil008_zFocal003'])
        piff_z9d_list.append(optatmo_psf_kwargs['zPupil009_zFocal001'])
        piff_z10d_list.append(optatmo_psf_kwargs['zPupil010_zFocal001'])
        piff_z11d_list.append(optatmo_psf_kwargs['zPupil011_zFocal001'])
        try:
            piff_z14d_list.append(optatmo_psf_kwargs['zPupil014_zFocal001'])
            piff_z15d_list.append(optatmo_psf_kwargs['zPupil015_zFocal001'])
        except:
            pass
        piff_opt_size_list.append(optatmo_psf_kwargs['size'])

    piff_z4d_list = np.array(piff_z4d_list)
    piff_z5d_list = np.array(piff_z5d_list)
    piff_z5x_list = np.array(piff_z5x_list)
    piff_z5y_list = np.array(piff_z5y_list)
    piff_z6d_list = np.array(piff_z6d_list)
    piff_z6x_list = np.array(piff_z6x_list)
    piff_z6y_list = np.array(piff_z6y_list)
    piff_z7d_list = np.array(piff_z7d_list)
    piff_z7x_list = np.array(piff_z7x_list)
    piff_z7y_list = np.array(piff_z7y_list)
    piff_z8d_list = np.array(piff_z8d_list)
    piff_z8x_list = np.array(piff_z8x_list)
    piff_z8y_list = np.array(piff_z8y_list)
    piff_z9d_list = np.array(piff_z9d_list)
    piff_z10d_list = np.array(piff_z10d_list)
    piff_z11d_list = np.array(piff_z11d_list)
    try:
        piff_z14d_list = np.array(piff_z14d_list)
        piff_z15d_list = np.array(piff_z15d_list)
    except:
        pass
    piff_opt_size_list = np.array(piff_opt_size_list)

    aos_z4d_list = np.delete(aos_z4d_list, delete_list)   
    aos_z5d_list = np.delete(aos_z5d_list, delete_list)   
    aos_z5x_list = np.delete(aos_z5x_list, delete_list)*129.0   
    aos_z5y_list = np.delete(aos_z5y_list, delete_list)*129.0   
    aos_z6d_list = np.delete(aos_z6d_list, delete_list)   
    aos_z6x_list = np.delete(aos_z6x_list, delete_list)*129.0 
    aos_z6y_list = np.delete(aos_z6y_list, delete_list)*129.0   
    aos_z7d_list = np.delete(aos_z7d_list, delete_list)   
    aos_z7x_list = np.delete(aos_z7x_list, delete_list)*129.0   
    aos_z7y_list = np.delete(aos_z7y_list, delete_list)*129.0   
    aos_z8d_list = np.delete(aos_z8d_list, delete_list)   
    aos_z8x_list = np.delete(aos_z8x_list, delete_list)*129.0 
    aos_z8y_list = np.delete(aos_z8y_list, delete_list)*129.0   
    aos_z9d_list = np.delete(aos_z9d_list, delete_list)   
    aos_z10d_list = np.delete(aos_z10d_list, delete_list)   
    aos_z11d_list = np.delete(aos_z11d_list, delete_list) 
    try:
        aos_z14d_list = np.delete(aos_z14d_list, delete_list)   
        aos_z15d_list = np.delete(aos_z15d_list, delete_list) 
    except:
        pass 
    aos_r0_list = np.delete(aos_r0_list, delete_list)   



    plt.figure()
    plt.scatter(aos_z4d_list, piff_z4d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z4d")
    plt.savefig("{0}/aos_vs_piff_z4d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z5d_list, piff_z5d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z5d")
    plt.savefig("{0}/aos_vs_piff_z5d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z5x_list, piff_z5x_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z5x")
    plt.savefig("{0}/aos_vs_piff_z5x_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z5y_list, piff_z5y_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z5y")
    plt.savefig("{0}/aos_vs_piff_z5y_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z6d_list, piff_z6d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z6d")
    plt.savefig("{0}/aos_vs_piff_z6d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z6x_list, piff_z6x_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z6x")
    plt.savefig("{0}/aos_vs_piff_z6x_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z6y_list, piff_z6y_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z6y")
    plt.savefig("{0}/aos_vs_piff_z6y_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z7d_list, piff_z7d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z7d")
    plt.savefig("{0}/aos_vs_piff_z7d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z7x_list, piff_z7x_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z7x")
    plt.savefig("{0}/aos_vs_piff_z7x_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z7y_list, piff_z7y_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z7y")
    plt.savefig("{0}/aos_vs_piff_z7y_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z8d_list, piff_z8d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z8d")
    plt.savefig("{0}/aos_vs_piff_z8d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z8x_list, piff_z8x_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z8x")
    plt.savefig("{0}/aos_vs_piff_z8x_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z8y_list, piff_z8y_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z8y")
    plt.savefig("{0}/aos_vs_piff_z8y_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z9d_list, piff_z9d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z9d")
    plt.savefig("{0}/aos_vs_piff_z9d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z10d_list, piff_z10d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z10d")
    plt.savefig("{0}/aos_vs_piff_z10d_{1}.png".format(graph_directory, psf_type))

    plt.figure()
    plt.scatter(aos_z11d_list, piff_z11d_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("z11d")
    plt.savefig("{0}/aos_vs_piff_z11d_{1}.png".format(graph_directory, psf_type))

    try:
        plt.figure()
        plt.scatter(aos_z14d_list, piff_z14d_list)
        plt.xlabel("aos")
        plt.ylabel("piff")
        plt.title("z14d")
        plt.savefig("{0}/aos_vs_piff_z14d_{1}.png".format(graph_directory, psf_type))

        plt.figure()
        plt.scatter(aos_z15d_list, piff_z15d_list)
        plt.xlabel("aos")
        plt.ylabel("piff")
        plt.title("z15d")
        plt.savefig("{0}/aos_vs_piff_z15d_{1}.png".format(graph_directory, psf_type))
    except:
        pass

    plt.figure()
    plt.scatter(aos_r0_list, piff_opt_size_list)
    plt.xlabel("aos")
    plt.ylabel("piff")
    plt.title("aos r0 vs piff opt size")
    plt.savefig("{0}/aos_r0_vs_piff_opt_size_{1}.png".format(graph_directory, psf_type))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--psf_type')
    parser.add_argument('--band')
    options = parser.parse_args()
    csv = options.csv
    psf_type = options.psf_type
    band = options.band
    kwargs = vars(options)
    del kwargs['csv']
    del kwargs['psf_type']
    del kwargs['band']
    make_graph()
