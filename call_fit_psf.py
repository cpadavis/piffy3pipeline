"""
Produce the config files, directories. Calls fit_psf.py
"""

from __future__ import print_function, division

import subprocess
import os
import glob
import yaml

import piff

def save_config(config, file_name):
    """Take a configuration dictionary and save to file

    :param config:  Dictionary of configuration
    :param file_name: file we are saving to
    """
    with open(file_name, 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

def call_fit_psf(run_config_path, bsub, check, call, print_log, overwrite, meanify, nmax):

    run_config = piff.read_config(run_config_path)
    if meanify:
        psf_files = run_config['psf_optics_files']
    else:
        psf_files = run_config['psf_optics_files'] + run_config['psf_other_files']
    psf_dir = run_config['psf_dir']  # where we load the psf files
    directory = run_config['directory']  # where we save files

    # get list of expids
    expids = [int(val.split('/')[-2]) for val in sorted(glob.glob('{0}/*/input.yaml'.format(psf_dir)))]

    nrun = 0
    for expid in expids:
        for psf_file in psf_files:
            psf_name = psf_file.split('.yaml')[0].split('/')[-1]
            if meanify:
                meanify_file_path = '{0}/meanify_{1}.fits'.format(directory, psf_name)
                if not os.path.exists(meanify_file_path):
                    continue

            config = piff.read_config(psf_file)
            time = config.pop('time', 30)
            memory = config.pop('memory', 1)

            config['input']['image_file_name'] = '{0}/{1}/*.fits.fz'.format(psf_dir, expid)
            # load up wcs info from expid
            try:
                expid_config = piff.read_config('{0}/{1}/input.yaml'.format(psf_dir, expid))
            except IOError:
                # input doesn't exist or is malformed
                print('skipping {0} because input.yaml cannot load'.format(expid))
                continue

            config['input']['wcs'] = expid_config['input']['wcs']
            # I messed up expnum when saving these
            config['input']['wcs']['exp'] = expid
            # and I also messed up the ccd splitting
            config['input']['wcs']['ccdnum']['str'] = "image_file_name.split('_')[-1].split('.fits')[0]"

            # code to make job_directory and name
            job_directory = '{0}/{1:08d}'.format(directory, expid)
            job_name = 'fit_psf__expid_{0:08d}__{1}'.format(expid, psf_name)

            # with updated everying, save config file
            if not os.path.exists(job_directory):
                os.makedirs(job_directory)
            save_config(config, job_directory + '/{0}.yaml'.format(psf_name))

            # create command
            command = [
                       'python',
                       'fit_psf.py',
                       '--directory', job_directory,
                       '--config_file_name', psf_name,
                       ]
            if print_log:
                command = command + ['--print_log']
            if meanify:
                command = command + ['--meanify_file_path', meanify_file_path]

            # call command
            skip_iter = False
            if check and bsub and call:
                jobcheck = subprocess.check_output(['bjobs', '-wJ', job_name])
                if job_name in jobcheck:
                    print('skipping {0} because it is already running'.format(job_name))
                    skip_iter = True
            if skip_iter:
                continue

            # check if output exists
            if meanify:
                if not os.path.exists('{0}/{1}.piff'.format(job_directory, psf_name)):
                    continue
            else:
                if overwrite:
                    file_names = glob.glob('{0}/{1}*'.format(job_directory, psf_name))
                    for file_name in file_names:
                        if '{0}.yaml'.format(psf_name) in file_name:
                            continue
                        elif os.path.exists(file_name):
                            os.remove(file_name)
                else:
                    if os.path.exists('{0}/{1}.piff'.format(job_directory, psf_name)):
                        continue

            if bsub:
                logfile = '{0}/bsub_fit_psf_{1}.log'.format(job_directory, psf_name)
                # check file exists, make it
                if os.path.exists(logfile) and call and overwrite:
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
            if bsub and call:
                print(job_name)
                subprocess.call(command)
            elif call:
                # do in python directoy so that if things fail we fail
                print(job_name)
                # call manually if no bsub
                subprocess.call(command)

        nrun += 1
        if nmax > 0 and nrun >= nmax:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsub', action='store_true', dest='bsub',
                        help='Use bsub')
    parser.add_argument('--check', action='store_true', dest='check',
                        help='When using bsub, do check if job submitted')
    parser.add_argument('--call', action='store_true', dest='call',
                        help='Do bsub or call.')
    parser.add_argument('--print_log', action='store_true', dest='print_log',
                        help='print logging instead of save')
    parser.add_argument('--overwrite', action='store_true', dest='overwrite',
                        help='Overwrite existing output')
    parser.add_argument('--meanify', action='store_true', dest='meanify',
                        help='look for meanify outputs and do those')
    parser.add_argument('-n', action='store', dest='nmax', type=int, default=0, help='Number of fits to run')
    parser.add_argument(action='store', dest='run_config_path',
                        default='config.yaml',
                        help='Run config to load up and do')

    options = parser.parse_args()
    kwargs = vars(options)

    call_fit_psf(**kwargs)
