# OK this should go actually into the fit_psf.py, but let's do this here for now

import numpy as np

import os
import glob

import piff
from fit_psf import load_star_images

def do_residuals(f, verbose=True, number_plot=0):
    c = f.replace('.piff', '.yaml')
    piff_name = f.split('/')[-1].split('.piff')[0]
    label = 'fitted'
    config = piff.read_config(c)
    if verbose:
        print('psf')
    psf = piff.read(f)

    # initialize output
    directory = '/'.join(c.split('/')[:-1])
    config['output']['dir'] = directory
    output = piff.Output.process(config['output'])

    # select nstars based on stars piece of config
    stat = output.stats_list[3]  # TODO: this bit is hardcoded
    if number_plot != 0:
        stat.number_plot = number_plot
    stat.indices = np.random.choice(len(psf.stars), stat.number_plot, replace=False)

    # pass params into output
    stat.stars = []
    for i, index in enumerate(stat.indices):
        star = psf.stars[index]
        stat.stars.append(star)
    # load their images
    if verbose:
        print('loading star images')
    stat.stars = load_star_images(stat.stars, config)

    if verbose:
        print('loading model')
    stat.models = []
    for star in stat.stars:

        # draw model star
        params = star.fit.params
        prof = psf.getProfile(params)
        model = psf.drawProfile(star, prof, params, copy_image=True)
        stat.models.append(model)

    if verbose:
        print('writing')

    file_name = '{0}/{1}_{2}_{3}'.format(directory, label, piff_name, os.path.split(stat.file_name)[1])
    stat.write(file_name=file_name)
    return file_name, stat.stars, stat.models

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(action='store', dest='run_config_path',
                        default='config.yaml',
                        help='Run config to load up and do')
    parser.add_argument('--index', action='store', dest='index', type=int, default=-1)
    options = parser.parse_args()
    kwargs = vars(options)

    # load config
    run_config = piff.read_config(run_config_path)
    psf_files = run_config['psf_optics_files'] + run_config['psf_other_files']
    directory = run_config['directory']  # where we save files

    files = []
    for psf_file in psf_files:
        piff_name = psf_file.split('.yaml')[0].split('/')[-1]
        files_i = glob.glob('{0}/*/{1}.piff'.format(directory, piff_name))
        files += files_i
    # # create list of files via glob
    # vkfiles = glob.glob('/u/ki/cpd/ki19/piff_test/y3pipeline/00*/psf_optatmovk.piff')
    # kofiles = glob.glob('/u/ki/cpd/ki19/piff_test/y3pipeline/00*/psf_optatmo.piff')
    # files = vkfiles + kofiles

    index = kwargs['index'] - 1
    if index == -2:
        print('Total number of files: {0}'.format(len(files)))
    else:
        niter = 10
        for f in files[niter * index: niter * (index + 1)]:
            print(f)
            fname, stars, models = do_residuals(f, verbose=False, number_plot=20)
