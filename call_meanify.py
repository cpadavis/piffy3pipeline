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
import piff

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
