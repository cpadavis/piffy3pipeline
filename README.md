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

# another example run:

    you'll make some config pfl_config.yaml

    # test that your config works -- will run only one expid, and will print logger output to screen
    python call_fit_psf.py --call --print_log -n 1 pfl_config.yaml

    # submit it all to the batch farm >:)
    python call_fit_psf.py --call --bsub pfl_config.yaml

    # drink coffee, probably eat a meal and sleep

    # meanify the psfs
    python call_meanify.py pfl_config.yaml

    # if you mess with your meanify params, overwrite this way:
    python call_meanify.py --overwrite pfl_config.yaml

    # rerun the meanify fit_psf
    python call_fit_psf.py --call --bsub --meanify pfl_config.yaml

    # you didn'jt like your interp settings, so rerun the interps
    python call_fit_psf.py --call --bsub --meanify --fit_interp_only --overwrite pfl_config.yaml

    # collect and summarize the PSFs
    python call_collect.py --call --bsub pfl_config.yaml


# TODO

    - now that I have a successful run under my belt, I need to check the memory and time use to put something more sane in
