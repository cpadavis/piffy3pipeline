# the y3 piff optatmo pipeline

This directory has the set of scripts and configs to produce the y3 psf runs.

- call_fit_psf.py
    - produces the directories and config files used for piffify

- fit_psf.py
    - runs the fits. Splits stars and saves test stars.

- call_meanify.py

- call_collect.py

- collect.py
    - collects optics fit parameters and accumulates rho stats of various fits


# example run:

    python call_fit_psf.py --bsub --call config.yaml

    # wait a while while each PSF is fitted

    # if you want to do meanify fits:
        python call_meanify.py config.yaml
        python call_fit_psf.py --meanify --bsub --call config.yaml

    python call_collect.py --bsub --call config.yaml

You also can collect fits to the zernike parameters of individual stars via call_zernike.py

# TODO

    - The code is getting tripped up on whether I mean to use the name of a file, or its name and extension, or the full path. I should be able to run this code anywhere, not just in the y3pipeline.
    - now that I have a successful run under my belt, I need to check the memory and tiem usages to put something more sane in
