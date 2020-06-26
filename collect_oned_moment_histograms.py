"""
Collect fit parameters and plot their values
Calculate stats over full set of stars
Mostly just copied form stats rho stats
"""
from __future__ import print_function, division

#fix for DISPLAY variable issue
import matplotlib
matplotlib.use('Agg')


import glob
import galsim
import treecorr
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pandas as pd
import fitsio
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip

# this function graphs the stars' data, model, difference, error, and pull images in histograms.
def make_oned_hists_pdf(shapes, directory, label, piff_name):
    list_of_moments = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
    list_of_moment_arrays = []
    for moment in list_of_moments:
        data_moments = shapes["data_{0}".format(moment)].values
        model_moments = shapes["model_{0}".format(moment)].values
        difference_moments = shapes["d{0}".format(moment)].values
        error_moments = shapes["data_sigma_{0}".format(moment)].values
        pull_moments = difference_moments / error_moments
        moment_array = np.array([data_moments, model_moments, difference_moments, error_moments, pull_moments])
        list_of_moment_arrays.append(moment_array)


    fig, axss = plt.subplots(nrows=10, ncols=5, figsize=(50, 100))#, squeeze=False)
    for r, moment_array, moment in zip(list(range(0,len(list_of_moment_arrays))), list_of_moment_arrays, list_of_moments):
        
        data_moment_array = moment_array[0]
        model_moment_array = moment_array[1]  
        difference_moment_array = moment_array[2] 
        error_moment_array = moment_array[3]   
        pull_moment_array = moment_array[4]          
        
        if moment == "e0":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.003),0.003))                
            axss[r][0].set_xlim(0.36,0.48) 
        if moment == "e1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0035),0.0035))    
            axss[r][0].set_xlim(-0.07,0.07) 
        if moment == "e2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.003),0.003))    
            axss[r][0].set_xlim(-0.06,0.06)  
        if moment == "zeta1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0005),0.0005))    
            axss[r][0].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0005),0.0005))    
            axss[r][0].set_xlim(-0.005,0.015) 
        if moment == "delta1":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.00075),0.00075))    
            axss[r][0].set_xlim(-0.015,0.015)  
        if moment == "delta2":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.00075),0.00075))    
            axss[r][0].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.01),0.01))    
            axss[r][0].set_xlim(4.1,4.5) 
        if moment == "orth6":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.0875),0.0875))    
            axss[r][0].set_xlim(13.0,16.5) 
        if moment == "orth8":
            axss[r][0].hist(data_moment_array, bins=np.arange(np.min(data_moment_array),np.max(data_moment_array+0.875),0.875))    
            axss[r][0].set_xlim(55.0,90.0) 
        axss[r][0].set_title("data {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.003),0.003))    
            axss[r][1].set_xlim(0.36,0.48) 
        if moment == "e1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0035),0.0035))    
            axss[r][1].set_xlim(-0.07,0.07) 
        if moment == "e2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.003),0.003))    
            axss[r][1].set_xlim(-0.06,0.06)  
        if moment == "zeta1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0005),0.0005))    
            axss[r][1].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0005),0.0005))    
            axss[r][1].set_xlim(-0.005,0.015) 
        if moment == "delta1":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.00075),0.00075))    
            axss[r][1].set_xlim(-0.015,0.015)  
        if moment == "delta2":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.00075),0.00075))    
            axss[r][1].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.01),0.01))    
            axss[r][1].set_xlim(4.1,4.5) 
        if moment == "orth6":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.0875),0.0875))    
            axss[r][1].set_xlim(13.0,16.5) 
        if moment == "orth8":
            axss[r][1].hist(model_moment_array, bins=np.arange(np.min(model_moment_array),np.max(model_moment_array+0.875),0.875))    
            axss[r][1].set_xlim(55.0,90.0) 
        axss[r][1].set_title("model {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "e1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "e2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00164),0.00164))    
            axss[r][2].set_xlim(-0.04,0.04) 
        if moment == "zeta1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.0005),0.0005))    
            axss[r][2].set_xlim(-0.01,0.01) 
        if moment == "zeta2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.0005),0.0005))    
            axss[r][2].set_xlim(-0.01,0.01) 
        if moment == "delta1":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00075),0.00075))    
            axss[r][2].set_xlim(-0.015,0.015) 
        if moment == "delta2":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.00075),0.00075))    
            axss[r][2].set_xlim(-0.015,0.015) 
        if moment == "orth4":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.01),0.01))    
            axss[r][2].set_xlim(-0.2,0.2) 
        if moment == "orth6":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.075),0.075))    
            axss[r][2].set_xlim(-1.5,1.5) 
        if moment == "orth8":
            axss[r][2].hist(difference_moment_array, bins=np.arange(np.min(difference_moment_array),np.max(difference_moment_array+0.75),0.75))    
            axss[r][2].set_xlim(-15.0,15.0) 
        axss[r][2].set_title("difference {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "e1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "e2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.001/20.0),0.001/20.0))    
            axss[r][3].set_xlim(0.0,0.02) 
        if moment == "zeta1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "zeta2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01)
        if moment == "delta1":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "delta2":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0005/20.0),0.0005/20.0))    
            axss[r][3].set_xlim(0.0,0.01) 
        if moment == "orth4":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.00375/20.0),0.00375/20.0))    
            axss[r][3].set_xlim(0.0,0.075) 
        if moment == "orth6":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.0375/20.0),0.0375/20.0))    
            axss[r][3].set_xlim(0.0,0.75) 
        if moment == "orth8":
            axss[r][3].hist(error_moment_array, bins=np.arange(np.min(error_moment_array),np.max(error_moment_array+0.375/20.0),0.375/20.0))    
            axss[r][3].set_xlim(0.0,7.5) 
        axss[r][3].set_title("error {0} histogram".format(moment))
        
        if moment == "e0":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.2),0.2))    
            axss[r][4].set_xlim(-4.0,4.0) 
        if moment == "e1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "e2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "zeta1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.1),0.1))    
            axss[r][4].set_xlim(-2.0,2.0) 
        if moment == "zeta2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.1),0.1))    
            axss[r][4].set_xlim(-2.0,2.0) 
        if moment == "delta1":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "delta2":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth4":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth6":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        if moment == "orth8":
            axss[r][4].hist(pull_moment_array, bins=np.arange(np.min(pull_moment_array),np.max(pull_moment_array+0.15),0.15))    
            axss[r][4].set_xlim(-3.0,3.0) 
        axss[r][4].set_title("pull {0} histogram".format(moment))
        
    #plt.tight_layout()
    fig.savefig("{0}/{1}_stars_oned_hists_{2}.pdf".format(directory,label, piff_name))

def convert_momentshapes_to_shapes(momentshapes):
    shapes = []
    for m in momentshapes:
        e0, e1, e2 = m
        sigma = np.sqrt(np.sqrt(np.square(e0 ** 2 - e1 ** 2 - e2 ** 2) * 0.25))
        shear = galsim.Shear(e1=e1/e0, e2=e2/e0)
        g1 = shear.g1
        g2 = shear.g2
        shapes.append([sigma, g1, g2])
    shapes = np.array(shapes)
    return shapes

def load_shapes(fo):
    # loads shapes, converts columns, and mad cuts
    shapes = pd.read_hdf(fo)

    # now add shapes to, ahum, shapes
    T, g1, g2 = convert_momentshapes_to_shapes(shapes[['data_e0', 'data_e1', 'data_e2']].values).T
    T_data_sigma, g1_data_sigma, g2_data_sigma = convert_momentshapes_to_shapes(shapes[['data_sigma_e0', 'data_sigma_e1', 'data_sigma_e2']].values).T
    dT = T - T_data_sigma
    dg1 = g1 - g1_data_sigma
    dg2 = g2 - g2_data_sigma

    shapes['data_T'] = T
    shapes['data_g1'] = g1
    shapes['data_g2'] = g2
    shapes['data_sigma_T'] = T_data_sigma
    shapes['data_sigma_g1'] = g1_data_sigma
    shapes['data_sigma_g2'] = g2_data_sigma
    shapes['dT'] = dT
    shapes['dg1'] = dg1
    shapes['dg2'] = dg2


    # nacut
    shapes.replace([np.inf, -np.inf], np.nan, inplace=True)
    shapes.dropna(axis=0, inplace=True)
    # madcut on data outliers
    mad_cols = ['data_T', 'data_g1', 'data_g2']
    mad = shapes[mad_cols].sub(shapes[mad_cols].median()).abs()
    madcut = mad.lt(5 * 1.48 * shapes[mad_cols].mad() + 1e-8)  # add a lil bit for those cases where every value in a column is exactly the same
    conds = madcut.all(axis=1)
    cut_shapes = shapes[conds]
    return cut_shapes, shapes, conds

def self__plot_single(ax, rho, color, marker, offset=0.):
    # Add a single rho stat to the plot.
    meanr = rho.meanr * (1. + rho.bin_size * offset)
    xip = rho.xip
    sig = np.sqrt(rho.varxip)
    ax.plot(meanr, xip, color=color)
    ax.plot(meanr, -xip, color=color, ls=':')
    ax.errorbar(meanr[xip>0], xip[xip>0], yerr=sig[xip>0], color=color, ls='', marker=marker)
    ax.errorbar(meanr[xip<0], -xip[xip<0], yerr=sig[xip<0], color=color, ls='', marker=marker)
    return ax.errorbar(-meanr, xip, yerr=sig, color=color, marker=marker)

def sum_sq(x):
    return np.sum(np.square(x))

def run_onedhists(files, plotdict):

    # plotdict has: plot_path, key_x, key_y, bins_x
    outs = {}

    for file_indx, fo in enumerate(files):
        if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            print('doing {0} out of {1}:'.format(file_indx + 1, len(files)))
        # load up the dataframe containing the shapes as measured with hsm
        shapes, nocut_shapes, conds = load_shapes(fo)

        # iterate through plotdict to do the summaries
        for key in plotdict:
            key_x = plotdict[key]['key_x']
            key_y = plotdict[key]['key_y']
            bins_x = plotdict[key]['bins_x']

            if key_x not in shapes:
                continue
            elif key_y not in shapes:
                continue

            df = shapes[[key_x, key_y]]
            # cut out infinite
            df = df.dropna(how='any')

            cut = pd.cut(df[key_x], bins_x, labels=False)
            group = df.groupby(cut)
            size = group.size()
            sums = group.agg('sum')
            sqsums = group.agg(sum_sq)

            if key not in outs:
                outs[key] = {'size': size, 'sums': sums, 'sqsums': sqsums}
            else:
                outs[key]['size'] = outs[key]['size'] + size
                outs[key]['sums'] = outs[key]['sums'] + sums
                outs[key]['sqsums'] = outs[key]['sqsums'] + sqsums

    # convert into estimations of the mean and std
    for key in outs:
        outs[key]['means'] = outs[key]['sums'].div(outs[key]['size'], axis=0)
        outs[key]['std'] = np.sqrt(outs[key]['sqsums'].div(outs[key]['size'], axis=0) - outs[key]['means'] ** 2).div(np.sqrt(outs[key]['size']), axis=0)  # err on the mean

    # now plot the outs
    for key in outs:
        plot_path = plotdict[key]['plot_path']
        key_x = plotdict[key]['key_x']
        key_y = plotdict[key]['key_y']
        bins_x = plotdict[key]['bins_x']

        fig = Figure(figsize = (10,5))
        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)

        x = outs[key]['means'][key_x]
        y = outs[key]['means'][key_y]
        yerr = outs[key]['std'][key_y]
        conds = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
        x = x[conds]
        y = y[conds]
        yerr = yerr[conds]
        ax.errorbar(x, y, yerr=yerr)

        try:
            ax.set_xlim(x.min(), x.max())
        except ValueError:
            continue
        if 'log_x' in plotdict[key]:
            if plotdict[key]['log_x']:
                ax.set_xscale('log', nonposx='mask')

        canvas = FigureCanvasAgg(fig)
        # Do this after we've set the canvas to use Agg to avoid warning.
        fig.set_tight_layout(True)

        # save files based on what is listed
        print('saving plot to {0}'.format(plot_path))
        canvas.print_figure(plot_path, dpi=100)

def run_rho(files, plot_path, uv_coord):

    all_shapes = []
    for file_indx, fo in enumerate(files):
        if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            print('doing {0} out of {1}:'.format(file_indx + 1, len(files)))
        # load up the dataframe containing the shapes as measured with hsm
        shapes, nocut_shapes, conds = load_shapes(fo)
        #try:
             #shapes, nocut_shapes, conds = load_shapes(fo)
        #except:
        #    print("failure to generate rho statistics for a particular exposure!")
        #    print("fo: {0}".format(fo))
        #    continue
        all_shapes.append(shapes)
    print('concatenating')
    shapes = pd.concat(all_shapes, ignore_index=True)

    print('doing shapes')
    if uv_coord:
        u = shapes['u']
        v = shapes['v']
    else:
        ra = shapes['ra']
        dec = shapes['dec']

    # though we have to convert to regular shapes
    T, g1, g2 = convert_momentshapes_to_shapes(shapes[['data_e0', 'data_e1', 'data_e2']].values).T
    T_data_sigma, g1_data_sigma, g2_data_sigma = convert_momentshapes_to_shapes(shapes[['data_sigma_e0', 'data_sigma_e1', 'data_sigma_e2']].values).T

    dT = T - T_data_sigma
    dg1 = g1 - g1_data_sigma
    dg2 = g2 - g2_data_sigma

    if uv_coord:
        cat_kwargs = {'x': u, 'y': v, 'x_units': 'arcsec', 'y_units': 'arcsec'}
    else:
        cat_kwargs = {'ra': ra, 'dec': dec, 'ra_units': 'deg', 'dec_units': 'deg'}

    print('doing catalogs')
    cat_kwargs['g1'] = g1
    cat_kwargs['g2'] = g2
    cat_g = treecorr.Catalog(**cat_kwargs)

    cat_kwargs['g1'] = dg1
    cat_kwargs['g2'] = dg2
    cat_dg = treecorr.Catalog(**cat_kwargs)

    cat_kwargs['g1'] = g1 * dT / T
    cat_kwargs['g2'] = g2 * dT / T
    cat_gdTT = treecorr.Catalog(**cat_kwargs)

    # accumulate corr
    self_tckwargs = {'min_sep': 0.1, 'max_sep': 600, 'bin_size': 0.2, 'sep_units': 'arcmin'}
    self_rho1 = treecorr.GGCorrelation(self_tckwargs)
    self_rho2 = treecorr.GGCorrelation(self_tckwargs)
    self_rho3 = treecorr.GGCorrelation(self_tckwargs)
    self_rho4 = treecorr.GGCorrelation(self_tckwargs)
    self_rho5 = treecorr.GGCorrelation(self_tckwargs)

    print('accumulating')
    self_rho1.process(cat_dg)
    self_rho2.process(cat_g, cat_dg)
    self_rho3.process(cat_gdTT)
    self_rho4.process(cat_dg, cat_gdTT)
    self_rho5.process(cat_g, cat_gdTT)

    print('make figure')
    # figure
    fig = Figure(figsize = (10,5))
    # In matplotlib 2.0, this will be
    # axs = fig.subplots(ncols=2)
    axs = [ fig.add_subplot(1,2,1),
            fig.add_subplot(1,2,2) ]
    axs = np.array(axs, dtype=object)

    # Left plot is rho1,3,4
    rho1 = self__plot_single(axs[0], self_rho1, 'blue', 'o')
    rho3 = self__plot_single(axs[0], self_rho3, 'green', 's', 0.1)
    rho4 = self__plot_single(axs[0], self_rho4, 'red', '^', 0.2)

    axs[0].legend([rho1, rho3, rho4],
                 [r'$\rho_1(\theta)$', r'$\rho_3(\theta)$', r'$\rho_4(\theta)$'],
                 loc='upper right', fontsize=12)
    axs[0].set_ylim(1.e-9, 1e-3)
    axs[0].set_xlim(self_tckwargs['min_sep'], self_tckwargs['max_sep'])
    axs[0].set_xlabel(r'$\theta$ (arcmin)')
    axs[0].set_ylabel(r'$\rho(\theta)$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log', nonposy='clip')

    # Right plot is rho2,5
    rho2 = self__plot_single(axs[1], self_rho2, 'blue', 'o')
    rho5 = self__plot_single(axs[1], self_rho5, 'green', 's', 0.1)

    axs[1].legend([rho2, rho5],
                 [r'$\rho_2(\theta)$', r'$\rho_5(\theta)$'],
                 loc='upper right', fontsize=12)
    axs[1].set_ylim(1.e-7, 1e-3)
    axs[1].set_xlim(self_tckwargs['min_sep'], self_tckwargs['max_sep'])
    axs[1].set_xlabel(r'$\theta$ (arcmin)')
    axs[1].set_ylabel(r'$\rho(\theta)$')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log', nonposy='clip')

    canvas = FigureCanvasAgg(fig)
    # Do this after we've set the canvas to use Agg to avoid warning.
    fig.set_tight_layout(True)

    # save files based on what is listed
    print('saving plot to {0}'.format(plot_path))
    canvas.print_figure(plot_path, dpi=100)

def run_collect_optics(files, file_out):
    # extract fit parameters from psf file. We are assuming optics
    fits = {'expid': []}
    skip_keys = []
    for file_indx, fo in enumerate(files):
        if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            print('doing {0} out of {1}:'.format(file_indx + 1, len(files)))
        # extract solution
        arr = fitsio.read(fo, 'psf_solution')
        for key in arr.dtype.names:
            if 'min_' in key and key not in skip_keys:
                skip_keys.append(key)
            elif 'max_' in key and key not in skip_keys:
                skip_keys.append(key)
            elif 'fix_' in key and key not in skip_keys:
                skip_keys.append(key)

            if key not in skip_keys:
                if file_indx == 0:
                    fits[key] = []
                fits[key].append(arr[key][0])

        # make sure the expid column also gets an entry
        expid = int(fo.split('/')[-2])
        fits['expid'].append(expid)

        # load in the atmo portion
        try:
            arr = fitsio.read(fo, 'psf_atmo_interp_kernel')
            vals = arr['FIT_THETA'][0]
            for vi, val in enumerate(vals):
                for vj, vali in enumerate(val):
                    key = 'gp_atmo_{0}_param_{1}'.format(['size', 'g1', 'g2'][vi], vj)
                    if file_indx == 0:
                        fits[key] = []
                    fits[key].append(vali)
        except IOError:
            # not in the file
            continue

    try:
        fits = pd.DataFrame(fits)
    except Exception as e:
        for key in fits:
            print(key, len(fits[key]))
        print(type(e))
        print(e.args)
        print(e)
        raise e

    # load up Aaron's DBase fits
    ajr_fits_path = '/nfs/slac/g/ki/ki19/des/cpd/piff_test/CtioDB_db-part1.csv'
    if os.path.exists(ajr_fits_path):
        ajr = pd.read_csv(ajr_fits_path)
        fits = pd.merge(fits, ajr, on='expid', how='left')
    else:
        print('Could not find Donut fits at {0}!'.format(ajr_fits_path))

    print('saving fits to {0}'.format(file_out))
    if os.path.exists(file_out):
        os.remove(file_out)
    fits.to_hdf(file_out, 'data')

def agg_to_array(agg, key, bins_x, bins_y):
    indx_x_transform = agg.index.labels[0].values()
    indx_y_transform = agg.index.labels[1].values()

    C = np.ma.zeros((bins_x.size - 1, bins_y.size - 1))
    C.mask = np.ones((bins_x.size - 1, bins_y.size - 1))
    np.add.at(C, [indx_x_transform, indx_y_transform],
              agg[key].values)
    np.multiply.at(C.mask, [indx_x_transform, indx_y_transform], 0)
    # bloops
    C = C.T
    return C

def run_twodhists(files, file_out_base, sep=50):
    bins_u = np.arange(-3900, 3900 + sep, sep)
    bins_v = np.arange(-3500, 3500 + sep, sep)

    all_shapes = []
    for file_indx, fo in enumerate(files):
        if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            print('doing {0} out of {1}:'.format(file_indx + 1, len(files)))
        # load up the dataframe containing the shapes as measured with hsm
        shapes, nocut_shapes, conds = load_shapes(fo)
        if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            print(len(conds), np.sum(conds))

        all_shapes.append(shapes)
    print('concatenating')
    arrays = pd.concat(all_shapes, ignore_index=True)
    # assign indices based on bins_u and bins_v
    print('cut by u and v')
    indx_u = pd.cut(arrays['u'], bins_u, labels=False)
    indx_v = pd.cut(arrays['v'], bins_v, labels=False)
    print('groupbying')
    agg = arrays.groupby((indx_u, indx_v)).agg(np.median)

    # save
    print('saving agg')
    agg.to_hdf('{0}_agg.h5'.format(file_out_base), 'agg')

    # make figures
    for key in arrays:
        print('doing key {0}'.format(key))
        fig = Figure(figsize = (12, 9))
        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title(key)

        C = agg_to_array(agg, key, bins_u, bins_v)
        vmin = np.nanpercentile(C.data[~C.mask], q=2)
        vmax = np.nanpercentile(C.data[~C.mask], q=98)
        IM = ax.pcolor(bins_u, bins_v, C, vmin=vmin, vmax=vmax)
        ax.set_xlim(min(bins_u), max(bins_u))
        ax.set_ylim(min(bins_v), max(bins_v))
        fig.colorbar(IM, ax=ax)

        canvas = FigureCanvasAgg(fig)
        # Do this after we've set the canvas to use Agg to avoid warning.
        fig.set_tight_layout(True)

        # save files based on what is listed
        plot_path = '{0}_{1}.pdf'.format(file_out_base, key)
        print('saving plot to {0}'.format(plot_path))
        canvas.print_figure(plot_path, dpi=100)

    print('saving stars')
    arrays.to_hdf('{0}_stars.h5'.format(file_out_base), 'stars')

def _add_twodhists(z, indx_u, indx_v, unique_indx, C):
    for unique in unique_indx:
        ui, vi = unique

        sample = z[(indx_u == ui) & (indx_v == vi)]
        if len(sample) > 0:
            value = np.sum(sample)
            C[vi, ui] += value
            C.mask[vi, ui] = 0

def collect(directory, piff_name, out_directory, do_optatmo=False, skip_rho=False, skip_oned=False, skip_twod=False, band="all"):
    skip_rho = True
    skip_oned = True
    skip_twod = True
    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    if band=="all":
        pass
    else:
        out_directory = out_directory + "/filter_{0}".format(band)
        os.system("mkdir {0}".format(out_directory))

    # params
    if do_optatmo:
        very_original_files = sorted(glob.glob('{0}/*/{1}.piff'.format(directory, piff_name)))
        try:
            acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
            original_files = []
            for very_original_file in very_original_files:
                very_original_exposure = very_original_file.split("/")[-2][2:]
                if very_original_exposure in acceptable_exposures:
                    original_files.append(very_original_file)
        except:
            original_files = very_original_files
        if band=="all":
            files = original_files
        else:
            files = []
            for original_file in original_files:
                exposure = original_file.split("/")[-2][2:]
                filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
                if filter_name_and_skip_dictionary['skip'] == True:
                    continue
                else:
                    filter_name = filter_name_and_skip_dictionary['filter_name']                 
                if filter_name in band:
                    files.append(original_file)
        if len(files) > 0:
            print('collecting optatmo params for {0} for {1} psfs'.format(piff_name, len(files)))
            file_out = '{0}/fit_parameters_for_oned_moment_histograms_{1}.h5'.format(out_directory, piff_name)
            run_collect_optics(files, file_out)











    for label in ['test', 'train']:
        very_original_files = sorted(glob.glob('{0}//*/shapes_{1}_{2}.h5'.format(directory, label, piff_name)))
        try:
            acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
            original_files = []
            for very_original_file in very_original_files:
                very_original_exposure = very_original_file.split("/")[-2][2:]
                if very_original_exposure in acceptable_exposures:
                    original_files.append(very_original_file)
        except:
            original_files = very_original_files
        if band=="all":
            files = original_files
        else:
            files = []
            for original_file in original_files:
                exposure = original_file.split("/")[-2][2:]
                filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
                if filter_name_and_skip_dictionary['skip'] == True:
                    continue
                else:
                    filter_name = filter_name_and_skip_dictionary['filter_name']
                if filter_name in band:
                    files.append(original_file)



        print("label: {0}".format(label))
        all_shapes = []
        for file_indx, fo in enumerate(files):
            if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
                print('doing {0} out of {1}:'.format(file_indx + 1, len(files)))
                print("fo: {0}".format(fo))
            # load up the dataframe containing the shapes as measured with hsm
            shapes = pd.read_hdf(fo)
            try:
                 shapes = pd.read_hdf(fo)
            except:
                print("failure to generate oned moment residual histograms for a particular exposure!")
                print("fo: {0}".format(fo))
                continue

            #if (file_indx + 1) % int(max([len(files) * 0.05, 1])) == 0:
            #    print(len(conds), np.sum(conds))

            all_shapes.append(shapes)
        print('concatenating')
        arrays = pd.concat(all_shapes, ignore_index=True)
        make_oned_hists_pdf(arrays, out_directory, label, piff_name)











    if not skip_rho:
        # rho stats
        # for uv_coord in [False]:  # only do RA
        # for uv_coord in [True]:
        for uv_coord in [True, False]:
            # for label in ['test']:
            for label in ['test', 'train']:
                very_original_files = sorted(glob.glob('{0}//*/shapes_{1}_{2}.h5'.format(directory, label, piff_name)))
                try:
                    acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
                    original_files = []
                    for very_original_file in very_original_files:
                        very_original_exposure = very_original_file.split("/")[-2][2:]
                        if very_original_exposure in acceptable_exposures:
                            original_files.append(very_original_file)
                except:
                    original_files = very_original_files
                if band=="all":
                    files = original_files
                else:
                    files = []
                    for original_file in original_files:
                        exposure = original_file.split("/")[-2][2:]
                        filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
                        if filter_name_and_skip_dictionary['skip'] == True:
                            continue
                        else:
                            filter_name = filter_name_and_skip_dictionary['filter_name']
                        if filter_name in band:
                            files.append(original_file)
                if len(files) > 0:
                    print('computing rho stats for {0} for {1} psfs'.format(piff_name, len(files)))
                    if uv_coord:
                        file_out = '{0}/rhouv_{1}_{2}.pdf'.format(out_directory, label, piff_name)

                    else:
                        file_out = '{0}/rhora_{1}_{2}.pdf'.format(out_directory, label, piff_name)
                    run_rho(files, file_out, uv_coord)

    if not skip_twod:
        # twod hists
        for label, sep in zip(['test', 'train'], [30, 15]):
            very_original_files = sorted(glob.glob('{0}//*/shapes_{1}_{2}.h5'.format(directory, label, piff_name)))
            try:
                acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
                original_files = []
                for very_original_file in very_original_files:
                    very_original_exposure = very_original_file.split("/")[-2][2:]
                    if very_original_exposure in acceptable_exposures:
                        original_files.append(very_original_file)
            except:
                original_files = very_original_files
            if band=="all":
                files = original_files
            else:
                files = []
                for original_file in original_files:
                    exposure = original_file.split("/")[-2][2:]      
                    filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
                    if filter_name_and_skip_dictionary['skip'] == True:
                        continue
                    else:
                        filter_name = filter_name_and_skip_dictionary['filter_name']                               
                    if filter_name in band:
                        files.append(original_file)
            if len(files) > 0:
                print('computing twod stats for {0} for {1} psfs'.format(piff_name, len(files)))
                file_out_base = '{0}/twodhists_{1}_{2}'.format(out_directory, label, piff_name)
                run_twodhists(files, file_out_base, sep=sep)

    plotdict = {}
    for shape_key in ['data_e0', 'de0', 'data_e1', 'de1', 'data_e2', 'de2',
                      'data_zeta1', 'dzeta1', 'data_zeta2', 'dzeta2',
                      'data_delta1', 'ddelta1', 'data_delta2', 'ddelta2',
                      'data_orth4', 'dorth4',
                      'data_orth6', 'dorth6',
                      'data_orth8', 'dorth8',
                      'atmo_size', 'atmo_g1', 'atmo_g2']:
        plotdict[shape_key] = {'key_x': 'data_flux', 'key_y': shape_key,
                               'bins_x': np.logspace(3, 7, 501), 'log_x': True}

    if not skip_oned:
        for label in ['test', 'train']:
            very_original_files = sorted(glob.glob('{0}//*/shapes_{1}_{2}.h5'.format(directory, label, piff_name)))
            try:
                acceptable_exposures = np.load("{0}/acceptable_exposures.npy".format(core_directory))
                original_files = []
                for very_original_file in very_original_files:
                    very_original_exposure = very_original_file.split("/")[-2][2:]
                    if very_original_exposure in acceptable_exposures:
                        original_files.append(very_original_file)
            except:
                original_files = very_original_files
            if band=="all":
                files = original_files
            else:
                files = []
                for original_file in original_files:
                    exposure = original_file.split("/")[-2][2:]
                    filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
                    if filter_name_and_skip_dictionary['skip'] == True:
                        continue
                    else:
                        filter_name = filter_name_and_skip_dictionary['filter_name']
                    if filter_name in band:
                        files.append(original_file)
            if len(files) > 0:
                # make plotdict
                for key in plotdict:
                    plotdict[key]['plot_path'] = '{0}/onedhists_{1}_{2}_{3}.pdf'.format(out_directory, key, label, piff_name)

                print('computing onedhists stats for {0} for {1} psfs'.format(piff_name, len(files)))
                run_onedhists(files, plotdict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', action='store', dest='directory',
                        help='where to look for psf files')
    parser.add_argument('--out_directory', action='store', dest='out_directory',
                        help='where to save files')
    parser.add_argument('--piff_name', action='store', dest='piff_name',
                        help='what psf file to look for')
    parser.add_argument('--do_optatmo', action='store_true', dest='do_optatmo',
                        help='Load up and save optatmo parameters.')
    parser.add_argument('--skip_rho', action='store_true', dest='skip_rho')
    parser.add_argument('--skip_oned', action='store_true', dest='skip_oned')
    parser.add_argument('--skip_twod', action='store_true', dest='skip_twod')
    parser.add_argument('--band')
    options = parser.parse_args()
    kwargs = vars(options)
    collect(**kwargs)