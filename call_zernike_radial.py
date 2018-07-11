"""
Calls the zernike.py command for individual optatmo PSF files (lets us see if starting position matters for the resultant fits). Also calls with and without the radial interpolation function.
"""

from __future__ import print_function, division

import glob
import subprocess
import os

import piff

def call_zernike(run_config_path, bsub, check, call):

    run_config = piff.read_config(run_config_path)
    directory = run_config['directory']  # directory such that you can go glob(directory/*/psf.piff) and find piff files

    # load up psf names
    # rho stats and atmo params
    piff_names = []
    psf_file_paths = []
    for psf_file in run_config['psf_optics_files']:
        config = piff.read_config(psf_file)
        base_piff_name = config['output']['file_name'].split('.piff')[0]
        interps = config['interps']
        # stick in the interps
        for interp_key in interps.keys():
            config_interp = interps[interp_key]
            piff_name = '{0}_{1}'.format(base_piff_name, interp_key)
            piff_names.append(piff_name)
            psf_file_paths.append(psf_file)

            if 'GPInterp' in config_interp['type']:
                piff_name = '{0}_{1}_meanified'.format(base_piff_name, interp_key)
                piff_names.append(piff_name)
                psf_file_paths.append(psf_file)

        piff_names.append('{0}_noatmo'.format(base_piff_name))
        psf_file_paths.append(psf_file)

    # go through the names and call
    for psf_file_path, piff_name in zip(psf_file_paths, piff_names):
        psf_name = psf_file.split('.yaml')[0].split('/')[-1]

        time = 2000
        memory = 2

        # go through expids
        files = sorted(glob.glob('{0}/*/{1}.piff'.format(directory, psf_name)))
        expids = [int(val.split('/')[-2]) for val in files]

        for expid in expids:
            for do_interp in [False, True]:
                job_directory = '{0}/{1:08d}'.format(directory, expid)
                job_name = 'zernike__expid_{0:08d}__{1}{2}'.format(expid, piff_name, ['', '_dointerp'][do_interp])

                # create command
                command = [
                           'python',
                           'zernike_radial.py',
                           '--directory', job_directory,
                           '--piff_name', piff_name,
                           '--config_file_name', psf_name,
                           ]
                if do_interp:
                    command += ['--do_interp']

                # call command
                skip_iter = False
                if check and bsub and call:
                    jobcheck = subprocess.check_output(['bjobs', '-wJ', job_name])
                    if job_name in jobcheck:
                        print('skipping {0} because it is already running'.format(job_name))
                        skip_iter = True
                if skip_iter:
                    continue

                if bsub:
                    logfile = '{0}/bsub_zernike_{1}{2}.log'.format(job_directory, piff_name, ['', '_dointerp'][do_interp])
                    # check file exists, make it
                    if os.path.exists(logfile) and call:
                        os.remove(logfile)

                    bsub_command = ['bsub',
                                    '-J', job_name,
                                    '-o', logfile]
                    if time > 0:
                        bsub_command += ['-W', str(time)]
                    if memory > 1:
                        bsub_command += ['-n', str(memory),]
                                         # '-R', '"span[ptile={0}]"'.format(memory)]

                    command = bsub_command + command

                print(' '.join(command))
                if call:
                    print(job_name)
                    subprocess.call(command)
        raise Exception('Temporarily stopping the zernike after first ap')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsub', action='store_true', dest='bsub',
                        help='Use bsub')
    parser.add_argument('--check', action='store_true', dest='check',
                        help='When using bsub, do check if job submitted')
    parser.add_argument('--call', action='store_true', dest='call',
                        help='Do bsub or call.')
    parser.add_argument(action='store', dest='run_config_path',
                        default='config.yaml',
                        help='Run config to load up and do')

    options = parser.parse_args()
    kwargs = vars(options)

    call_zernike(**kwargs)
