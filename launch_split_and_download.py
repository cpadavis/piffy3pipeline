import numpy as np 
import time
import os
import glob

Parallel = True
Failure_mode = True

def get_njobs():

    jobs_running = os.popen('bjobs | grep RUN | wc').read()
    char_run = ''
    for c in jobs_running:
        if c != ' ':
            char_run += c
        if c == ' ' and len(char_run) !=0:
            break

    jobs_pend = os.popen('bjobs | grep PEND | wc').read()
    char_pend = ''
    for c in jobs_pend:
        if c != ' ':
            char_pend += c
        if c == ' ' and len(char_pend) !=0:
            break 
    njobs = int(char_pend) + int(char_run)
    
    return njobs

def check_success_exposure(rep, expid):

    exp = rep+expid
    tt = glob.glob(exp+'/*Tmp*')
    AA = glob.glob(exp+'/D*')
    nccd = len(glob.glob(exp+'/psf_im_*_*.fits.fz'))
    fail =  (len(tt) != 0 or len(AA) != 0)
    fail = (fail or nccd==0)
    return fail 

def clean_repo_exposure(rep, expid):
    exp = rep+expid
    os.system('rm '+exp+'/*Tmp*')
    os.system('rm '+exp+'/D*')    

N = 5766

exp_repo = glob.glob('/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exp_info_y3a1-v29_grizY/*')
exp_name = []
for exp in exp_repo:
    if exp not in ['/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exp_info_y3a1-v29_grizY/name_des_file.list',
                   '/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exp_info_y3a1-v29_grizY/exp_info_y3a1-v29_grizY.tarz']:
            
        exp_name.append(exp[-6:])

exp_name = np.array(exp_name)
exp_name.sort()

if Parallel:
    NMAX = 10
    n_fail = 0
    for i in range(N):
        print i+1, '/', n_fail, '/', N
        failure = check_success_exposure('/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exposures_v29_grizY/',exp_name[i])
        if failure and Failure_mode:
            clean_repo_exposure('/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exposures_v29_grizY/',exp_name[i])
            output_job_scr = '/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/y3_v29_dl/y3_v29_%i_test.log'%(i)
            command = 'python split_and_download_y3_piff_summary.py -i %i'%(i)
            os.system('bsub -W 0:50 -o %s "%s"'%((output_job_scr, command)))
            n_fail += 1
            time.sleep(5)

        njobs_run_pend = get_njobs()
        print 'JOB RUNNING/PENDING: ', njobs_run_pend


        if njobs_run_pend>=NMAX:
            while njobs_run_pend>=NMAX:
                print 'wait, I have %i jobs in computation'%(njobs_run_pend)
                time.sleep(60)
                njobs_run_pend = get_njobs()
        print ''

else:
    n_fail = 0
    for i in range(N):
        failure = check_success_exposure('/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exposures_v29_grizY/',exp_name[i])
        if failure or i==15:
            clean_repo_exposure('/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/exposures_v29_grizY/',exp_name[i])
            command = 'python split_and_download_y3_piff_summary.py -i %i'%(i)
            os.system(command)
            n_fail += 1
            if n_fail==1:
                break
        print i, n_fail, N 
