"""
Fits each training star to the full atmosphere model, possibly plus interp if given. Save the resultant parameters, errors, and shapes.

Then makes plots of the rho stats, parameter distributions, and star residuals
"""
from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

import fitsio
import pandas as pd
from scipy.interpolate import interp1d

import lmfit
import piff
from fit_psf import load_star_images, measure_star_shape, plot_2dhist_shapes

# a convenient function for making a plot drawing stars
def get_radial_profile(star, star_drawn):
    # make a 1d profile of the drawn and image
    I, w, u, v = star.data.getDataVector()
    mI, mw, mu, mv = star_drawn.data.getDataVector()

    dcenter = star.fit.center
    u0 = star.data.properties['u'] + dcenter[0]
    v0 = star.data.properties['v'] + dcenter[1]
    u0 = dcenter[0]
    v0 = dcenter[1]
    r = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)  # shared between
    dI = I - mI

    rbins = np.linspace(0, 5, 126)
    df = pd.DataFrame({'r': r, 'dI': dI, 'image': I, 'image_model': mI})
    cut = pd.cut(df['r'], bins=rbins, labels=False)

    agg = df.groupby(cut).agg(np.median)

    return agg

# this gets the radial profile for a set of stars
def collect_radial_profiles(stars, stars_drawn):
    rs = []
    dIs = []
    for star, star_drawn in zip(stars, stars_drawn):
        # make a 1d profile of the drawn and image
        I, w, u, v = star.data.getDataVector()
        mI, mw, mu, mv = star_drawn.data.getDataVector()

        dcenter = star.fit.center
        u0 = star.data.properties['u'] + dcenter[0]
        v0 = star.data.properties['v'] + dcenter[1]
        u0 = dcenter[0]
        v0 = dcenter[1]
        r = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)  # shared between I and mI
        dI = (I - mI) / star.fit.flux
        rs += r.tolist()
        dIs += dI.tolist()
    r = np.array(rs)
    dI = np.array(dIs)

    rbins = np.linspace(0, 5, 501)
    df = pd.DataFrame({'r': r, 'dI': dI})
    cut = pd.cut(df['r'], bins=rbins, labels=False)

    agg = df.groupby(cut).agg(np.median)

    return agg

def draw_stars(star, star_drawn, fig, axs):
    image = star.image.array
    drawn = star_drawn.image.array

    vmin = np.percentile([image, drawn], 2)
    vmax = np.percentile([image, drawn], 98)
    dvmax = np.percentile(np.abs(image - drawn), 95)
    dvmin = -dvmax

    ax = axs[0]
    IM = ax.imshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(IM, ax=ax)
    ax.set_title('Star')

    ax = axs[1]
    IM = ax.imshow(drawn, vmin=vmin, vmax=vmax)
    fig.colorbar(IM, ax=ax)
    ax.set_title('PSF at (u,v) = ({0:+.2e}, {1:+.2e})'.format(star.data.properties['u'], star.data.properties['v']))

    ax = axs[2]
    IM = ax.imshow(image - drawn, vmin=dvmin, vmax=dvmax, cmap=plt.cm.RdBu_r)
    fig.colorbar(IM, ax=ax)

    weight = star.data.getImage()[1].array
    chi2 = np.sum(np.square((np.sqrt(weight) * (image - drawn)).flatten()))
    ax.set_title('Star - PSF. Chi2/dof = {0:.2f}'.format(chi2 * 1. / 625))

    agg = get_radial_profile(star, star_drawn)
    center = agg['r']
    hist = agg['dI']

    ax = axs[3]
    ax.plot(center, hist, 'k-')
    ax.plot(center, hist * 0, 'k--')
    ax.set_title('Averaged residual radial profile')


def zernike(directory, config_file_name, piff_name):
    config = piff.read_config('{0}/{1}.yaml'.format(directory, config_file_name))
    logger = piff.setup_logger(verbose=3)

    # load base optics psf
    out_path = '{0}/{1}.piff'.format(directory, piff_name)
    psf = piff.read(out_path)

    # load images for train stars
    psf.stars = load_star_images(psf.stars, config, logger=logger)
    stars = psf.stars

    params = psf.getParamsList(stars)

    # draw model stars
    model_stars = psf.drawStarList(stars)

    # fit radial piece
    radial_agg = collect_radial_profiles(stars, model_stars)
    interpfunc = interp1d(radial_agg['r'].values, radial_agg['dI'].values)
    radial_agg.to_hdf('{0}/radial_{1}_{2}.h5'.format(directory, 'train', piff_name), 'data')
    fig = Figure(figsize = (10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(radial_agg['r'], radial_agg['dI'])
    ax.set_xlabel('r')
    ax.set_ylabel('Residual radial image')
    canvas = FigureCanvasAgg(fig)
    # Do this after we've set the canvas to use Agg to avoid warning.
    fig.set_tight_layout(True)
    plot_path = '{0}/radial_{1}_{2}.pdf'.format(directory, 'train', piff_name)
    logger.info('saving plot to {0}'.format(plot_path))
    canvas.print_figure(plot_path, dpi=100)

    logger.info('Fitting {0} stars'.format(len(stars)))
    model_fitted_stars = []
    for star_i, star in zip(range(len(stars)), stars):
        if (star_i + 1) % int(max([len(stars) * 0.05, 1])) == 0:
            logger.info('doing {0} out of {1}:'.format(star_i + 1, len(stars)))
        try:
            if (star_i + 1) % int(max([len(stars) * 0.05, 1])) == 0:
                model_fitted_star, results = psf.fit_model(star, params=params[star_i], vary_shape=True, vary_optics=True, mode='pixel', logger=logger)
            else:
                model_fitted_star, results = psf.fit_model(star, params=params[star_i], vary_shape=True, vary_optics=True, mode='pixel')
            model_fitted_stars.append(model_fitted_star)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning('{0}'.format(str(e)))
            logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
    stars = model_fitted_stars
    logger.info('Drawing final model stars')
    drawn_stars = [psf.drawProfile(star, psf.getProfile(star.fit.params), star.fit.params, copy_image=True, use_fit=True) for star in stars]


    logger.info('Measuring star shapes')
    shapes = measure_star_shape(stars, drawn_stars, logger=logger)

    logger.info('Adding fitted params and params_var')
    shape_keys = ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']
    shape_plot_keys = []
    for key in shape_keys:
        shape_plot_keys.append(['data_' + key, 'model_' + key, 'd' + key])
    # TODO: this should be a part of the PSF
    param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2']
    if psf.atmosphere_model == 'vonkarman':
        param_keys += ['atmo_L0']
    param_keys += ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
    logger.info('Extracting training fit parameters')
    params = np.array([star.fit.params for star in stars])
    params_var = np.array([star.fit.params_var for star in stars])
    for i in range(params.shape[1]):
        shapes['{0}_fit'.format(param_keys[i])] = params[:, i]
        shapes['{0}_var'.format(param_keys[i])] = params_var[:, i]

    shapes['chisq'] = np.array([star.fit.chisq for star in stars])
    shapes['dof'] = np.array([star.fit.dof for star in stars])

    logger.info('saving shapes')
    shapes.to_hdf('{0}/zernikeshapes_{1}_{2}_zernike.h5'.format(directory, 'train', piff_name), 'data')

    logger.info('saving stars')
    fits_path = '{0}/zernikestars_{1}_{2}_zernike.fits'.format(directory, 'train', piff_name)
    with fitsio.FITS(fits_path, 'rw', clobber=True) as f:
        piff.Star.write(stars, f, extname='zernike_stars')

    logger.info('making 2d plots')
    # plot shapes
    fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
    # save
    fig.savefig('{0}/zernike_shapes_{1}_{2}.pdf'.format(directory, 'train', piff_name))
    # plot params
    plot_keys = []
    plot_keys_i = []
    for i in range(params.shape[1]):
        plot_keys_i.append(param_keys[i])
        if len(plot_keys_i) == 3:
            plot_keys.append(plot_keys_i)
            plot_keys_i = []
    if len(plot_keys_i) > 0:
        plot_keys_i += [plot_keys_i[0]] * (3 - len(plot_keys_i))
        plot_keys.append(plot_keys_i)
    fig, axs = plot_2dhist_shapes(shapes, [[key + '_fit' for key in kp] for kp in plot_keys], diff_mode=False)
    fig.savefig('{0}/zernike_fit_params_{1}_{2}.pdf'.format(directory, 'train', piff_name))

    nstars = min([20, len(stars)])
    indices = np.random.choice(len(stars), nstars, replace=False)
    logger.info('saving {0} star images'.format(nstars))
    fig = Figure(figsize = (4 * 4, 3 * nstars))
    for i, indx in enumerate(indices):
        axs = [ fig.add_subplot(nstars, 4, i * 4 + j + 1) for j in range(4)]
        # select a star
        star = stars[indx]
        # draw the model star
        params = star.fit.params
        prof = psf.getProfile(params)
        star_drawn = psf.drawProfile(star, prof, params, use_fit=True, copy_image=True)
        # make plot
        draw_stars(star, star_drawn, fig, axs)

    canvas = FigureCanvasAgg(fig)
    # Do this after we've set the canvas to use Agg to avoid warning.
    fig.set_tight_layout(True)

    # save files based on what is listed
    plot_path = '{0}/zernike_stars_{1}_{2}.pdf'.format(directory, 'train', piff_name)
    logger.info('saving plot to {0}'.format(plot_path))
    canvas.print_figure(plot_path, dpi=100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', action='store', dest='directory',
                        help='where to look for psf files')
    parser.add_argument('--piff_name', action='store', dest='piff_name',
                        help='what psf file to look for')
    parser.add_argument('--config_file_name', action='store', dest='config_file_name',
                        help='name of the config file (without .yaml)')

    options = parser.parse_args()
    kwargs = vars(options)

    zernike(**kwargs)
