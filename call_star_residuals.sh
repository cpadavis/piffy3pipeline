#!/bin/bash
#BSUB -W 20
#BSUB -o logs/star_residuals_%I.log
#BSUB -J "star_residual[1-216]%100"
#BSUB -n 1

python star_residuals.py --index $LSB_JOBINDEX config.yaml
