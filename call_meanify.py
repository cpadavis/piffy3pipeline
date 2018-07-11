"""
Collect PSF files, run

ex for using the average file:
        gp = piff.GPInterp2pcf(kernel="0.009 * RBF(300.*0.26)",
                               optimize=fit_hyp, white_noise=1e-5, average_fits='output/average.fits')
        gp.initialize(stars_training)
        gp.solve(stars_training)

"""

from __future__ import print_function, division

import os
import glob

from sklearn.neighbors import KNeighborsRegressor
import fitsio
import piff
import pandas as pd

from fit_psf import plot_2dhist_shapes

def meanify_config(files, average_file, meanify_params):
    config = {'output': {'file_name': files},
              'hyper': {'file_name': average_file}}
    for key in ['output', 'hyper']:
        if key in meanify_params:
            config[key].update(meanify_params[key])

    return config

def call_meanify(run_config_path, overwrite, n):

    logger = piff.setup_logger(verbose=3, log_file='meanify.log')

    run_config = piff.read_config(run_config_path)
    psf_files = run_config['psf_optics_files']
    directory = run_config['directory']  # where we save files
    meanify_params = run_config['meanify']

    for psf_file in psf_files:
        logger.info('PSF File {0}'.format(psf_file))
        psf_name = psf_file.split('.yaml')[0].split('/')[-1]
        average_file = '{0}/meanify_{1}.fits'.format(directory, psf_name)

        if os.path.exists(average_file):
            if overwrite:
                os.remove(average_file)
            else:
                logger.warning('skipping {0} because its average file already exists and we are not overwriting!'.format(psf_name))
                continue

        # glob over expids
        logger.info('Globbing')
        files = sorted(glob.glob('{0}/*/{1}.piff'.format(directory, psf_name)))
        if n > 0:
            files = files[:n]
        logger.info('Meanifying')

        config = meanify_config(files, average_file, meanify_params)
        piff.meanify(config, logger=logger)

        # # make plots of meanify
        # logger.info('Making plots')

        # # load X0 y0
        # n_neighbors=4
        # average = fitsio.read(average_file)
        # X0 = average['COORDS0'][0]
        # y0 = average['PARAMS0'][0]

        # neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        # neigh.fit(X0, y0)
        # y = neigh.predict(X0)
        # keys = [['atmo_size', 'atmo_g1', 'atmo_g2']]
        # shapes = {'u': X0[:, 0], 'v': X0[:, 1],
        #           'atmo_size': y[:, 0],
        #           'atmo_g1': y[:, 1], 'atmo_g2': y[:, 2]}
        # keys_i = []
        # for i in range(3, len(y[0])):
        #     key = 'param_{0}'.format(i)
        #     shapes[key] = y[:, i]
        #     keys_i.append(key)
        #     if len(keys_i) == 3:
        #         keys.append(keys_i)
        #         keys_i = []
        # if len(keys_i) > 0:
        #     keys_i += [keys_i[0]] * (3 - len(keys_i))
        #     keys.append(keys_i)

        # shapes = pd.DataFrame(shapes)

        # fig, axs = plot_2dhist_shapes(shapes, keys, gridsize=500)
        # fig.savefig('{0}/meanify_params_{1}.pdf'.format(directory, psf_name))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', dest='overwrite',
                        help='Overwrite existing output')
    parser.add_argument('-n', action='store', dest='n', type=int, default=0, help='Number of fits to meanify')
    parser.add_argument(action='store', dest='run_config_path',
                        help='Run config to load up and do')

    options = parser.parse_args()
    kwargs = vars(options)

    call_meanify(**kwargs)
