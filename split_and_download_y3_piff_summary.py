"""
Using summary PSF catalog:
    - split into individual CCD catalog files for collation
    - match to gaia?
    - download ccd image
    - create config file with wcs info
"""

from __future__ import print_function, division
import os
import glob
import yaml
import time
import numpy as np
from scipy.stats import mode

import fitsio

def main(i=-1):
    try:
        #piff_dir = '/nfs/slac/g/ki/ki19/des/cpd/y3_piff'
        piff_dir = '/nfs/slac/g/ki/ki19/deuce/AEGIS/leget/piff'
        piff_dir_astro = '/nfs/slac/g/ki/ki19/des/cpd/y3_piff'
        out_dir = piff_dir + '/exposures_v29_grizY'
        cat_dir = piff_dir + '/exp_info_y3a1-v29_grizY'
        # cat_dir = piff_dir + '/exp_info_y3a1-v23'
        possible_exps = sorted(glob.glob(cat_dir + '/*'))
        print(len(possible_exps))
        expnums = []
        for d in possible_exps:
            try:
                di = int(d.split('/')[-1])
                expnums.append(di)
            except ValueError:
                # not a d
                print('Skipping {0}'.format(d))
        expnums = np.array(expnums)
        exposures_ccd = fitsio.read('{0}/exposures-ccds-Y3A1_COADD.fits'.format(piff_dir))
        astro_zones = fitsio.read('{0}/astro/which_zone.fits'.format(piff_dir_astro))

        if i >= 0:
            expnums = [expnums[i]]
        for expnum in expnums:
            exposures_ccd_expnum = exposures_ccd[exposures_ccd['expnum'] == expnum]

            # ccds
            ccds = exposures_ccd_expnum['ccdnum']

            # paths
            paths = exposures_ccd_expnum['path']

            # zonenum. Decide which zone for the exposure based on which zone occures most often.
            zonenum = mode(astro_zones[astro_zones['expnum'] == expnum]['zone'])[0][0]

            cat_file = '{0}/{1}/exp_psf_cat_{1}.fits'.format(cat_dir, expnum)

            do_exposure(out_dir, cat_file, exposures_ccd_expnum, expnum, ccds, paths, zonenum)
        success = True
    except:
        success = False
    exp = '{0}/{1}'.format(out_dir, expnum) 
    temp_file_1 = glob.glob(exp+'/*Tmp*')
    temp_file_2 = glob.glob(exp+'/D*')
    nccd = len(glob.glob(exp+'/psf_im_*_*.fits.fz'))
 
    success &= (len(temp_file_1) == 0)
    success &= (len(temp_file_2) == 0)

    if not success:
        os.system('rm '+exp+'/*Tmp*')
        os.system('rm '+exp+'/D*')

    success &= (nccd != 0)

    return success

def do_exposure(outdir, cat_file, exposures_ccd_expnum, expnum, ccds, paths, zonenum):
    # output directory
    sdir = '{0}/{1}'.format(outdir, expnum)
    if not os.path.exists(sdir):
        print('Creating directory {0}'.format(sdir))
        os.makedirs(sdir)

    # TODO: split catalog file
    print('splitting {0}'.format(expnum))
    catalog = fitsio.read(cat_file, ext=1)
    for ccd in np.unique(catalog['ccdnum']):
        ccd_path = '{0}/psf_cat_{1}_{2}.fits'.format(sdir, expnum, ccd)
        cat = catalog[catalog['ccdnum'] == ccd]
        exposures_ccd_expnum_i = exposures_ccd_expnum[exposures_ccd_expnum['ccdnum'] == ccd]
        # OK to keep this in the same format as the previous things, we save this twice as ext 1 and 2, and then ext 3 will be the exposures recarr but just the particular ccd
        fitsio.write(ccd_path, cat, 'all_obj')
        fitsio.write(ccd_path, cat, 'stars')
        fitsio.write(ccd_path, exposures_ccd_expnum_i, 'info')

    # download images
    for ccd, path in zip(ccds, paths):
        # check if cat file exists
        ccd_path = '{0}/psf_cat_{1}_{2}.fits'.format(sdir, expnum, ccd)
        if os.path.exists(ccd_path):
            download_single_ccd(path, sdir, expnum, ccd)
        else:
            print('skipping {0}/{1} because we could not find {2}'.format(expnum, ccd, ccd_path))

    # make yaml
    indict = {
        'image_file_name': 'psf_im_{0}_*.fits.fz'.format(expnum),
        'image_hdu': 0,
        'badpix_hdu': 1,
        'weight_hdu': 2,

        'cat_file_name':
            {
                'simage_file_name': '@input.image_file_name',
                'str': 'image_file_name.replace("psf_im", "psf_cat").replace(".fits.fz", ".fits")',
                'type': 'Eval'
            },
        'chipnum':
            {
                'simage_file_name': '@input.image_file_name',
                'str': "image_file_name.split('_')[3].split('.fits')[0]",
                'type': 'Eval',
            },
        'cat_hdu': 2,

        'x_col': 'x',
        'y_col': 'y',
        'sky_col': 'sky',  # note that in the Y3 cats these are all 0!
        # 'ra': 'TELRA',  # not ra?
        # 'dec': 'TELDEC',  # not dec?
        'ra': 'ra',  # not ra?
        'dec': 'dec',  # not dec?
        'gain': 1.0,
        'use_col': 'use',

        'max_snr': 100,
        'min_snr': 20,

        'stamp_size': 25,

        'wcs': {
            'type': 'Pixmappy',
            'dir': '/nfs/slac/g/ki/ki19/des/cpd/y3_piff/astro/',
            # file_name also needs to change to be for EACH ccd. astro_zones specifies via detpos, but we want ccdnum
            'file_name': 'zone{0:03d}.astro'.format(zonenum),
            'exp': '{0}'.format(expnum),
            'ccdnum':
                {
                    'simage_file_name': '@input.image_file_name',
                    'str': "image_file_name.split('_')[3].split('.fits')[0]",
                    'type': 'Eval',
                },
            },
    }

    yaml_dict = {'input': indict}
    # save yaml_dict someplace
    with open('{0}/input.yaml'.format(sdir), 'w') as f:
        f.write(yaml.dump(yaml_dict, default_flow_style=False))

# TODO:
def download_single_ccd(path, sdir, expnum, ccd):
    url_base = 'https://cpd:%s@desar2.cosmology.illinois.edu/DESFiles/desarchive/'%ps()

    """
    # need;
    image_file_name
    expnum
    zonenum

    # no idea what they are yet
    path
In [10]: arr['path'][0].rsplit('/', 3)
Out[10]:
    ['OPS/finalcut/Y2A1/Y3-2379/20160108/D00509375/p01',
     'red',
      'immask',
       'D00509375_z_c39_r2379p01_immasked.fits.fz']


    can load the exp_psf_cat_509375.fits blahblah file, go for the path array, which will have the path at desdm
    """

    # outname
    outname = '{0}/psf_im_{1}_{2}.fits'.format(sdir, expnum, ccd)
    if os.path.exists(outname):
        print('skipping {0} because it exists!'.format(outname))
        return

    # Download the files we need:
    base_path, _, _, image_file_name = path.rsplit('/',3)
    # strip any spaces on end
    image_file_name = image_file_name.strip()
    root, ext = image_file_name.rsplit('_',1)
    print('root, ext = |%s| |%s|'%(root,ext))
    image_file = wget(url_base, base_path + '/red/immask/', sdir, root + '_' + ext)
    print('image_file = ',image_file)
    print(time.ctime())

    bkg_file = wget(url_base, base_path + '/red/bkg/', sdir, root + '_bkg.fits.fz')
    print('bkg_file = ',bkg_file)
    print(time.ctime())

    # Unpack the image file if necessary
    # change the image file name so that it makes life easier
    unpack_image_file = unpack_file(image_file, outname)
    print('unpacked to ',unpack_image_file)
    print(time.ctime())

    # Subtract off the background right from the start
    with fitsio.FITS(unpack_image_file, 'rw') as f:
        bkg = fitsio.read(bkg_file)
        #print('after read bkg')
        img = f[0].read()
        img -= bkg
        #print('after subtract bkg')
        f[0].write(img)
        #print('after write new img')
    print('subtracted off background image')
    print(time.ctime())
    pack_file(unpack_image_file)  # saves to unpack_image_file + .fz
    print('packing image')
    print(time.ctime())

    # delete non subtracted files
    for file_to_remove in [sdir + '/' + root + '_' + ext, sdir + '/' + root + '_bkg.fits.fz', unpack_image_file]:
        os.remove(file_to_remove)

def wget(url_base, path, wdir, file):
    url = url_base + path + file
    full_file = os.path.join(wdir,file)

    if not os.path.isfile(full_file):
        print('Downloading ',full_file)
        # Sometimes this fails with an "http protocol error, bad status line".
        # Maybe from too many requests at once or something.  So we retry up to 5 times.
        nattempts = 5
        cmd = 'wget %s -O %s'%(url, full_file)
        for attempt in range(1,nattempts+1):
            print('%s  (attempt %d)'%(cmd, attempt))
            print('For PF: ', cmd)
            run_with_timeout(cmd, 300)
            if os.path.exists(full_file):
                break
    return full_file

def run_with_timeout(cmd, timeout_sec):
    # cf. https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
    import subprocess, shlex
    from threading import Timer

    proc = subprocess.Popen(shlex.split(cmd))
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()

def obfuscate(s):
    mask = 'I Have a Dream'
    lmask = len(mask)
    nmask = [ord(c) for c in mask]
    return ''.join([chr(ord(c) ^ nmask[i % lmask]) for i, c in enumerate(s)])

def ps():
    return obfuscate('~\x10+\t\x1f\x15S\x07T3')[:-3]

def unpack_file(file_name, img_file):
    """Create the unpacked file in the work directory if necessary.

    If the unpacked file already exists, then a link is made.
    Otherwise funpack is run, outputting the result into the work directory.
    """
    print('unpack ',file_name)
    print('to ',img_file)

    # img_file = os.path.splitext(file_name)[0]
    if os.path.lexists(img_file):
        print('   %s exists already.  Removing.'%img_file)
        os.remove(img_file)
    print('   unpacking fz file')
    cmd = 'funpack -O {outf} {inf}'.format(outf=img_file, inf=file_name)
    print(cmd)
    for itry in range(5):
        print('For PF: ', cmd)
        run_with_timeout(cmd, 120)
        if os.path.lexists(img_file): break
        print('%s was not properly made.  Retrying.'%img_file)
        time.sleep(10)

    if not os.path.lexists(img_file):
        print('Unable to create %s.  Skip this file.'%img_file)
        return None

    return img_file

def pack_file(file_name):
    """Create the unpacked file in the work directory if necessary.

    If the unpacked file already exists, then a link is made.
    Otherwise funpack is run, outputting the result into the work directory.
    """
    print('pack ',file_name)

    img_file = file_name + '.fz'
    # img_file = os.path.splitext(file_name)[0]
    if os.path.lexists(img_file):
        print('   %s exists already.  Removing.'%img_file)
        os.remove(img_file)
    print('   packing fz file')
    cmd = 'fpack {inf}'.format(inf=file_name)
    print(cmd)
    for itry in range(5):
        print('For PF: ', cmd)
        run_with_timeout(cmd, 120)
        if os.path.lexists(img_file): break
        print('%s was not properly made.  Retrying.'%img_file)
        time.sleep(10)

    if not os.path.lexists(img_file):
        print('Unable to create %s.  Skip this file.'%img_file)
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='i',
                        type=int, default=-1,
                        help='Which job')
    args = parser.parse_args()
    
    success = False 
    ntry=1
    while not success and ntry<5:
        success = main(args.i)
        ntry += 1

