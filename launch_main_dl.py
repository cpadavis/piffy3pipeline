import os

output_job_scr = '/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff/y3_v29_dl/MAIN.log'
command = 'python launch_split_and_download.py'
os.system('bsub -W 50:00 -o %s "%s"'%((output_job_scr, command)))
