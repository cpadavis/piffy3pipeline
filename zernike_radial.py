"""
Fits each training star to the full atmosphere model, possibly plus interp if given. Save the resultant parameters, errors, and shapes.

Then makes plots of the rho stats, parameter distributions, and star residuals for 50 stars.

TODO: save chi2 and dof of the final fits as well
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

# pretty much optatmopsf.drawProfile but with the correction added
def drawProfile(self, star, prof, params, use_fit=True, copy_image=True, interpfunc=None):
    """Generate PSF image for a given star and profile

    :param star:        Star instance holding information needed for
                        interpolation as well as an image/WCS into which
                        PSF will be rendered.
    :param profile:     A galsim profile
    :param params:      Params associated with profile to put in the star.
    :param use_fit:     Bool [default: True] shift the profile by a star's
                        fitted center and multiply by its fitted flux
    :param self:        PSF instance
    :param interpfunc:  The interpolation function

    :returns:           Star instance with its image filled with rendered
                        PSF
    """
    # use flux and center properties
    if use_fit:
        prof = prof.shift(star.fit.center) * star.fit.flux
    image, weight, image_pos = star.data.getImage()
    if copy_image:
        image_model = image.copy()
    else:
        image_model = image
    prof.drawImage(image_model, method='auto', offset=(star.image_pos-image_model.true_center))

    properties = star.data.properties.copy()
    for key in ['x', 'y', 'u', 'v']:
        # Get rid of keys that constructor doesn't want to see:
        properties.pop(key, None)
    data = piff.StarData(image=image_model,
                    image_pos=star.data.image_pos,
                    weight=star.data.weight,
                    pointing=star.data.pointing,
                    field_pos=star.data.field_pos,
                    values_are_sb=star.data.values_are_sb,
                    orig_weight=star.data.orig_weight,
                    properties=properties)
    fit = piff.StarFit(params,
                  flux=star.fit.flux,
                  center=star.fit.center)
    new_s = piff.Star(data, fit)

    # apply radial correction
    if interpfunc is not None:
        new_im = apply_correction(new_s, interpfunc)
        new_s = piff.Star(new_s.data.setData(new_im.flatten(), include_zero_weight=True), new_s.fit)

    return new_s

def _fit_model_residual_with_radial(lmparams, star, self, interpfunc):
    """Residual function for fitting individual profile parameters

    :param lmparams:    lmfit Parameters object
    :param star:        A Star instance.
    :param self:        PSF instance
    :param interpfunc:  The interpolation function

    :returns chi:       Chi of observed pixels to model pixels
    """

    all_params = lmparams.valuesdict().values()
    flux, du, dv = all_params[:3]
    params = all_params[3:]


    prof = self.getProfile(params)

    image, weight, image_pos = star.data.getImage()

    # use for getting drawprofile
    star.fit.flux = flux
    star.fit.center = (du, dv)
    star_drawn = drawProfile(self, star, prof, params, use_fit=True, interpfunc=interpfunc)
    image_model = star_drawn.image

    chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
    return chi

# pretty much straight out of optatmopsf.fit_model but with the residual function changed
def fit_with_radial(self, star, interpfunc, vary_shape=True, vary_optics=True):

    params = star.fit.params

    flux = star.fit.flux
    if flux == 1.:
        # a pretty reasonable first guess is to just take the sum of the pixels
        flux = star.image.array.sum()
    du, dv = star.fit.center
    lmparams = lmfit.Parameters()
    # Order of params is important!
    lmparams.add('flux', value=flux, vary=True, min=0.0)
    lmparams.add('du', value=du, vary=True, min=-1, max=1)
    lmparams.add('dv', value=dv, vary=True, min=-1, max=1)
    # we must also cut the min and max based on opt_params to avoid things
    # like large ellipticities or small sizes
    min_size = self.optatmo_psf_kwargs['min_size']
    max_size = self.optatmo_psf_kwargs['max_size']
    max_g = self.optatmo_psf_kwargs['max_g1']
    # getParams puts in atmosphere terms

    fit_size = params[0]
    fit_g1 = params[1]
    fit_g2 = params[2]
    opt_size = params[3]
    opt_g1 = params[4]
    opt_g2 = params[5]
    lmparams.add('atmo_size', value=fit_size, vary=vary_shape, min=min_size - opt_size, max=max_size - opt_size)
    lmparams.add('atmo_g1', value=fit_g1,   vary=vary_shape, min=-max_g - opt_g1, max=max_g - opt_g1)
    lmparams.add('atmo_g2', value=fit_g2,   vary=vary_shape, min=-max_g - opt_g2, max=max_g - opt_g2)
    # add other params to the params model
    # we do NOT vary the optics size, g1, g2
    lmparams.add('optics_size', value=opt_size, vary=False)
    lmparams.add('optics_g1', value=opt_g1, vary=False)
    lmparams.add('optics_g2', value=opt_g2, vary=False)
    for i, pi in enumerate(params[6:]):
        # we do allow zernikes to vary
        lmparams.add('optics_zernike_{0}'.format(i + 4), value=pi, vary=vary_optics, min=-5, max=5)

    results = lmfit.minimize(_fit_model_residual_with_radial, lmparams, args=(star, self, interpfunc),
                             epsfcn=1e-5, method='leastsq')

    # subtract 3 for the flux, du, dv
    fit_params = np.zeros(len(results.params) - 3)
    params_var = np.zeros(len(fit_params))
    for i, key in enumerate(results.params):
        indx = i - 3
        if key in ['flux', 'du', 'dv']:
            continue
        param = results.params[key]
        fit_params[indx] = param.value
        if hasattr(param, 'stderr'):
            params_var[indx] = param.stderr ** 2

    flux = results.params['flux'].value
    du = results.params['du'].value
    dv = results.params['dv'].value
    center = (du, dv)
    chisq = results.chisqr
    dof = results.nfree

    fit = piff.StarFit(fit_params, params_var=params_var, flux=flux, center=center, chisq=chisq, dof=dof)
    star_fit = piff.Star(star.data.copy(), fit)

    return star_fit, results

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

# this function modifies the image from a star based on the interp_function
def apply_correction(star_drawn, interp_function, flux=None, center=None):
    mI, mw, mu, mv = star_drawn.data.getDataVector(include_zero_weight=True)
    if center is not None:
        dcenter = center
    else:
        dcenter = star_drawn.fit.center

    if flux is None:
        flux = star_drawn.fit.flux

    u0 = dcenter[0]
    v0 = dcenter[1]
    mr = np.sqrt((mu - u0) ** 2 + (mv - v0) ** 2)  # shared between

    # for each r, find nearest rin, add dI.
    # Ignore points that fall outside the fitted range (usually the central pixel or else edge pixels)
    rconds = (mr < interp_function.__dict__['x'].max()) * (mr > interp_function.__dict__['x'].min())
    dmI = interp_function(mr[rconds]) * flux

    mI = mI.copy()
    mI[rconds] += dmI

    # reshape I
    mI = mI.reshape(25, 25)

    return mI

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


def zernike(directory, config_file_name, piff_name, do_interp):
    config = piff.read_config('{0}/{1}.yaml'.format(directory, config_file_name))
    logger = piff.setup_logger(verbose=3)

    # load base optics psf
    out_path = '{0}/{1}.piff'.format(directory, piff_name)
    psf = piff.read(out_path)

    # load images for train stars
    psf.stars = load_star_images(psf.stars, config, logger=logger)
    stars = psf.stars

    params = psf.getParamsList(stars)

    # if do_interp, draw star models and fit radial profile
    if do_interp:
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

        # do the fits of the stars
        logger.info('Fitting {0} stars'.format(len(stars)))
        model_fitted_stars = []
        for star_i, star in zip(range(len(stars)), stars):
            if (star_i + 1) % int(max([len(stars) * 0.05, 1])) == 0:
                logger.info('doing {0} out of {1}:'.format(star_i + 1, len(stars)))
            try:
                model_fitted_star, results = fit_with_radial(psf, star, interpfunc, vary_shape=True, vary_optics=True)
                model_fitted_stars.append(model_fitted_star)
                if (star_i + 1) % int(max([len(stars) * 0.05, 1])) == 0:
                    logger.debug(lmfit.fit_report(results, min_correl=0.5))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.warning('{0}'.format(str(e)))
                logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
        stars = model_fitted_stars
        logger.info('Drawing final model stars')
        drawn_stars = [drawProfile(psf, star, psf.getProfile(star.fit.params), star.fit.params, use_fit=True, copy_image=True, interpfunc=interpfunc) for star in stars]
    else:
        # else just do regular zernike fit
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
    param_keys = ['atmo_size', 'atmo_g1', 'atmo_g2'] + ['optics_size', 'optics_g1', 'optics_g2'] + ['z{0:02d}'.format(zi) for zi in range(4, 45)]
    logger.info('Extracting training fit parameters')
    params = np.array([star.fit.params for star in stars])
    params_var = np.array([star.fit.params_var for star in stars])
    for i in range(params.shape[1]):
        shapes['{0}_fit'.format(param_keys[i])] = params[:, i]
        shapes['{0}_var'.format(param_keys[i])] = params_var[:, i]

    shapes['chisq'] = np.array([star.fit.chisq for star in stars])
    shapes['dof'] = np.array([star.fit.dof for star in stars])

    logger.info('saving shapes')
    shapes.to_hdf('{0}/zernikeshapes_{1}_{2}_zernike{3}.h5'.format(directory, 'train', piff_name, ['_reg', '_interp'][do_interp]), 'data')

    logger.info('saving stars')
    fits_path = '{0}/zernikestars_{1}_{2}_zernike{3}.fits'.format(directory, 'train', piff_name, ['_reg', '_interp'][do_interp])
    with fitsio.FITS(fits_path, 'rw', clobber=True) as f:
        piff.Star.write(stars, f, extname='zernike_stars')

    logger.info('making 2d plots')
    # plot shapes
    fig, axs = plot_2dhist_shapes(shapes, shape_plot_keys, diff_mode=True)
    # save
    fig.savefig('{0}/zernike{3}_shapes_{1}_{2}.pdf'.format(directory, 'train', piff_name, ['_reg', '_interp'][do_interp]))
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
    fig.savefig('{0}/zernike{3}_fit_params_{1}_{2}.pdf'.format(directory, 'train', piff_name, ['_reg', '_interp'][do_interp]))

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
        if do_interp:
            star_drawn = drawProfile(psf, star, psf.getProfile(star.fit.params), star.fit.params, use_fit=True, copy_image=True, interpfunc=interpfunc)
        else:
            star_drawn = psf.drawProfile(star, prof, params, use_fit=True, copy_image=True)
        # make plot
        draw_stars(star, star_drawn, fig, axs)

    canvas = FigureCanvasAgg(fig)
    # Do this after we've set the canvas to use Agg to avoid warning.
    fig.set_tight_layout(True)

    # save files based on what is listed
    plot_path = '{0}/zernike{3}_stars_{1}_{2}.pdf'.format(directory, 'train', piff_name, ['_reg', '_interp'][do_interp])
    logger.info('saving plot to {0}'.format(plot_path))
    canvas.print_figure(plot_path, dpi=100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', action='store', dest='directory',
                        help='where to look for psf files')
    parser.add_argument('--piff_name', action='store', dest='piff_name',
                        help='what psf file to look for')
    parser.add_argument('--do_interp', action='store_true', dest='do_interp',
                        help='do the interpolate')
    parser.add_argument('--config_file_name', action='store', dest='config_file_name',
                        help='name of the config file (without .yaml)')

    options = parser.parse_args()
    kwargs = vars(options)

    zernike(**kwargs)
