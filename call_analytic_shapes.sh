#!/bin/bash
#BSUB -W 600
#BSUB -o logs_analytic_shapes/%I.log
#BSUB -J "analytic_shapes[500-653]%800"


python analytic_shapes.py -i $LSB_JOBINDEX
