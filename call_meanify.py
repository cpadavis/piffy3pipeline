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
import numpy as np
from astropy.io import fits

from fit_psf import plot_2dhist_shapes

def meanify_config(files, average_file, meanify_params):
    config = {'output': {'file_name': files},
              'hyper': {'file_name': average_file}}
    for key in ['output', 'hyper']:
        if key in meanify_params:
            config[key].update(meanify_params[key])

    return config

def call_meanify(run_config_path, overwrite, n):

    core_directory = os.path.realpath(__file__)
    program_name = core_directory.split("/")[-1]
    core_directory = core_directory.split("/{0}".format(program_name))[0]
    source_directory = np.load("{0}/source_directory_name.npy".format(core_directory))[0]

    logger = piff.setup_logger(verbose=3, log_file='meanify.log')

    run_config = piff.read_config(run_config_path)
    psf_files = run_config['psf_optics_files']
    directory = run_config['directory']  # where we save files
    meanify_params = run_config['meanify']

    for psf_file in psf_files:
        logger.info('PSF File {0}'.format(psf_file))
        psf_name = psf_file.split('.yaml')[0].split('/')[-1]
        average_file = '{0}/meanify_{1}_{2}.fits'.format(directory, psf_name, band)

        if os.path.exists(average_file):
            if overwrite:
                os.remove(average_file)
            else:
                logger.warning('skipping {0} because its average file already exists and we are not overwriting!'.format(psf_name))
                continue

        # glob over expids
        logger.info('Globbing')
        if band=="all":
            files = sorted(glob.glob('{0}/*/{1}.piff'.format(directory, psf_name)))
        else:
            original_files = sorted(glob.glob('{0}/*/{1}.piff'.format(directory, psf_name)))
            files = []
            for original_file in original_files:
                exposure = original_file.split("/")[-2][2:]
                skip=False
                for index in range(1,63):
                    try:
                        band_test_file = "{0}/{1}/psf_cat_{1}_{2}.fits".format(source_directory, exposure, index)
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
                    band_test_file = "{0}/{1}/exp_psf_cat_{1}.fits".format(source_directory, exposure)
                    hdu_c = fits.open(band_test_file)
                    filter_name = hdu_c[1].data['band'][0][0]
                    print(filter_name)
                except:
                    try:
                        filter_name = hdu[3].data['band'][0]
                    except:
                        continue
                if filter_name in band:
                    files.append(original_file)
        if n > 0:
            files = files[:n]
        logger.info('Meanifying')

        config = meanify_config(files, average_file, meanify_params)
        piff.meanify(config, logger=logger)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', dest='overwrite',
                        help='Overwrite existing output')
    parser.add_argument('-n', action='store', dest='n', type=int, default=0, help='Number of fits to meanify')
    parser.add_argument(action='store', dest='run_config_path',
                        help='Run config to load up and do')
    parser.add_argument('--band')
    options = parser.parse_args()
    band = options.band
    kwargs = vars(options)
    del kwargs['band']
    call_meanify(**kwargs)
