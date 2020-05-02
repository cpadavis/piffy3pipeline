"""
Produce the config files, directories. Calls fit_psf.py
"""

from __future__ import print_function, division

import subprocess
import os
import glob
import yaml
from astropy.io import fits
import numpy as np

import piff
from call_angular_moment_residual_plot_maker_part2 import find_filter_name_or_skip

def save_config(config, file_name):
    """Take a configuration dictionary and save to file

    :param config:  Dictionary of configuration
    :param file_name: file we are saving to
    """
    with open(file_name, 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

def call_fit_psf(run_config_path, bsub, check, call, print_log, overwrite, meanify, nmax, fit_interp_only, band, bands_meanified_separately, opt_only, no_opt, no_interp, no_final_graphs, final_graphs_only):

    if opt_only and (no_opt or fit_interp_only or meanify or final_graphs_only):
        raise ValueError('You cannot both do only the optical fit and skip the optical fit.')
    if no_interp and (fit_interp_only or meanify or final_graphs_only):
        raise ValueError('You cannot both do only the atmo interp or part of it (like the final graphs) and skip the atmo interp.')
    if no_final_graphs and final_graphs_only:
        raise ValueError('You cannot both do only the final graphs and skip the final graphs.')
    run_config = piff.read_config(run_config_path)
    if True:
        psf_files = run_config['psf_optics_files']
    else:
        psf_files = run_config['psf_optics_files'] + run_config['psf_other_files']
    psf_dir = run_config['psf_dir']  # where we load the psf files
    source_directory = psf_dir
    #directory = run_config['directory']
    current_directory = os.path.realpath(__file__)
    program_name = current_directory.split("/")[-1]
    current_directory = current_directory.split("/{0}".format(program_name))[0]
    directory = current_directory # where we save files

    # get list of expids
    expids = [int(val.split('/')[-2]) for val in sorted(glob.glob('{0}/*/input.yaml'.format(psf_dir)))]

    nrun = 0
    for expid in expids:
        exposure = str(expid)
        if band=="all":
            filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
            if filter_name_and_skip_dictionary['skip'] == True:
                continue
            else:
                filter_name = filter_name_and_skip_dictionary['filter_name']                    
        else:
            filter_name_and_skip_dictionary = find_filter_name_or_skip(source_directory=source_directory, exposure=exposure)
            if filter_name_and_skip_dictionary['skip'] == True:
                continue
            else:
                filter_name = filter_name_and_skip_dictionary['filter_name']                
            if filter_name in band:
                pass
            else:
                continue
        for psf_file in psf_files:
            psf_name = psf_file.split('.yaml')[0].split('/')[-1]
            if meanify:
                if not bands_meanified_separately:
                    meanify_file_path = '{0}/meanify_{1}_{2}.fits'.format(directory, psf_name, band)
                else:
                    meanify_file_path = '{0}/meanify_{1}_{2}.fits'.format(directory, psf_name, filter_name)                    
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
            filter_lam = {"u": 387.6, "g": 484.2, "r": 643.9, "i": 782.1, "z": 917.2, "Y": 987.8}
            if config['psf']['type'] == 'OptAtmo':
           	# look up band information
                # modify config band information
                config['psf']['optical_psf_kwargs']['lam'] = filter_lam[filter_name]

            if print_log:
                config['verbose'] = 3

            # code to make job_directory and name
            job_directory = '{0}/{1:08d}'.format(directory, expid)
            job_name = 'fit_psf__expid_{0:08d}__{1}'.format(expid, psf_name)

            # with updated everying, save config file
            if not os.path.exists(job_directory):
                os.makedirs(job_directory)
            save_config(config, job_directory + '/{0}.yaml'.format(psf_name))

            np.save("{0}/source_directory_name.npy".format(directory), np.array([source_directory]))
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
            if fit_interp_only:
                command = command + ['--fit_interp_only']
            if opt_only:
                command = command + ['--opt_only']
            if no_opt:
                command = command + ['--no_opt']
            if no_interp:
                command = command + ['--no_interp']
            if no_final_graphs:
                command = command + ['--no_final_graphs']
            if final_graphs_only:
                command = command + ['--final_graphs_only']

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
                if overwrite and not fit_interp_only:
                    file_names = glob.glob('{0}/{1}*'.format(job_directory, psf_name))
                    for file_name in file_names:
                        if '{0}.yaml'.format(psf_name) in file_name:
                            continue
                        elif os.path.exists(file_name):
                            os.remove(file_name)
                else:
                    if os.path.exists('{0}/{1}.piff'.format(job_directory, psf_name)):
                        if fit_interp_only:
                            pass
                        else:
                            continue
                    else:
                        if fit_interp_only:
                            # need that piff file to exist in order to fit it
                            continue
                        else:
                            pass

            if bsub:
                logfile = '{0}/bsub_fit_psf_{1}.log'.format(job_directory, psf_name)
                # check file exists, make it
                if os.path.exists(logfile) and call and overwrite:
                    os.remove(logfile)

                #time = 50
                #memory = 2
                bsub_command = ['bsub',
                                '-J', job_name,
                                '-R', 'rhel60',                                
                                '-o', logfile]
                if time > 0:
                    bsub_command += ['-W', str(time)]
                #if memory > 1:
                #    bsub_command += ['-n', str(memory),]
                #                     # '-R', '"span[ptile={0}]"'.format(memory)]
                if memory > 1:
                    bsub_command += ['-n', str(memory),
                                      #'-R', '"span[ptile={0}]"'.format(memory), 'rhel60']
                                      '-R', 'span[hosts=1]']

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
    parser.add_argument('--fit_interp_only', action='store_true', dest='fit_interp_only',
                        help='start fit after (previously done) individual atmo star fit')
    parser.add_argument('-n', action='store', dest='nmax', type=int, default=0, help='Number of fits to run')
    parser.add_argument(action='store', dest='run_config_path',
                        default='config.yaml',
                        help='Run config to load up and do')
    parser.add_argument('--band')
    parser.add_argument('--bands_meanified_separately', action='store_true', dest='bands_meanified_separately',
                        help='one can either use a meanify file for multiple bands together or multiple meanify files ,one for each band; default is the former unless this flag is used')
    parser.add_argument('--opt_only', action='store_true', dest='opt_only',
                        help='only do the optical fit')
    parser.add_argument('--no_opt', action='store_true', dest='no_opt',
                        help='start fit after (previously done) optical fit')
    parser.add_argument('--no_interp', action='store_true', dest='no_interp',
                        help='stop fit right before the atmo interpolation')
    parser.add_argument('--no_final_graphs', action='store_true', dest='no_final_graphs',
                        help='stop fit right before the final graphs are made')
    parser.add_argument('--final_graphs_only', action='store_true', dest='final_graphs_only',
                        help='(having previously done everything else), do just the final graphs')
    options = parser.parse_args()
    kwargs = vars(options)
    print("hello")
    call_fit_psf(**kwargs)
