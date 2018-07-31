#!/bin/bash
#BSUB -W 45
#BSUB -o y3_v29_dl/y3_v29_%I.log
# there were a LOT of exposures (like 30k) but I only do some
#BSUB -J "y3_v29[640-1170]%20"


python split_and_download_y3_piff_summary.py -i $LSB_JOBINDEX
