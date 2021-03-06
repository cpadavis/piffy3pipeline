"""
Do the collecting
"""

from __future__ import print_function, division

import subprocess
import os

import piff

def call_collect(run_config_path, bsub, check, call, skip_rho, skip_oned, skip_twod, skip_params):

    run_config = piff.read_config(run_config_path)
    directory = run_config['directory']  # directory such that you can go glob(directory/*/psf.piff) and find piff files

    # load up psf names
    # rho stats and atmo params
    piff_names = []
    psf_file_paths = []
    do_optatmos = []
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
            do_optatmos.append(True)

            if 'GPInterp' in config_interp['type']:
                piff_name = '{0}_{1}_meanified'.format(base_piff_name, interp_key)
                piff_names.append(piff_name)
                psf_file_paths.append(psf_file)
                do_optatmos.append(True)

        piff_names.append('{0}_noatmo'.format(base_piff_name))
        psf_file_paths.append(psf_file)
        do_optatmos.append(False)
    for psf_file in run_config['psf_other_files']:
        config = piff.read_config(psf_file)
        piff_name = config['output']['file_name'].split('.piff')[0]
        psf_file_paths.append(psf_file)
        piff_names.append(piff_name)
        do_optatmos.append(False)

    # go through the names and call
    for psf_file_path, piff_name, do_optatmo in zip(psf_file_paths, piff_names, do_optatmos):

        time = 600
        memory = 3

        # code to make job_directory and name
        out_directory = '{0}/plots/{1}'.format(directory, piff_name)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        job_name = 'collect_{0}'.format(piff_name)

        # create command
        command = [
                   'python',
                   'collect.py',
                   '--directory', directory,
                   '--out_directory', out_directory,
                   '--piff_name', piff_name,
                   ]
        if do_optatmo and not skip_params:
            command += ['--do_optatmo']
        if skip_rho:
            command += ['--skip_rho']
        if skip_oned:
            command += ['--skip_oned']
        if skip_twod:
            command += ['--skip_twod']
        if not skip_twod:
            # the twod bit is lonnnnng
            time = 2000

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
            logfile = '{0}/bsub_collect_{1}.log'.format(out_directory, piff_name)
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
    parser.add_argument('--skip_rho', action='store_true', dest='skip_rho')
    parser.add_argument('--skip_oned', action='store_true', dest='skip_oned')
    parser.add_argument('--skip_twod', action='store_true', dest='skip_twod')
    parser.add_argument('--skip_params', action='store_true', dest='skip_params')

    options = parser.parse_args()
    kwargs = vars(options)

    call_collect(**kwargs)
