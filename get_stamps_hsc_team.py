#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
from glob import glob, iglob
from subprocess import call

import numpy as np

import urllib.request
import getpass
from urllib.error import HTTPError
from http.client import IncompleteRead


from astropy.io import ascii, fits
# from lsdlib import read_input
# from array_utils import lowercase_table_keywords

#users = {"banados":"mypwd"}
users = {"lauramr":"+Qqg4W0/BiFVVx0snDsH2C+nFtniJnXJZvRN0gmA"}

EXAMPLES = """

This is a wrapper of the following:

Get a single file

A link (link) will appear to the right of the Quarry button when the button is enabled. You can copy & paste the link if you want to use a command line tool to download it.

You can successively get other files by changing the suffix of the url. We recommend that you create a url list like:

https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-G&tract=&rerun=
https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-R&tract=&rerun=
https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-I&tract=&rerun=
https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-Z&tract=&rerun=
https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-Y&tract=&rerun=

dr2
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=34.339958333&dec=-2.147944444&sw=2amin&sh=2amin&type=coadd&image=on&filter=HSC-I&tract=&rerun=pdr2_wide
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=34.339958333&dec=-2.147944444&sw=2amin&sh=2amin&type=coadd&image=on&filter=HSC-Y&tract=&rerun=pdr2_wide

https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=242.470958333333&dec=53.4725&sw=2amin&sh=2amin&type=coadd&image=on&filter=HSC-Y&tract=&rerun=pdr2_wide

https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-G&tract=&rerun=pdr2_wide
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-R&tract=&rerun=pdr2_wide
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-I&tract=&rerun=pdr2_wide
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-Z&tract=&rerun=pdr2_wide
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-Y&tract=&rerun=pdr2_wide

DR3
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?ra=-24&dec=0&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-Y&tract=&rerun=pdr3_wide


Note:
You can convert the flux to magnitude using the following equation;

m_obs = -2.5 * { log10( Flux valu in catalog ) - log10( FLUXMAG0 ) }

# FLUXMAG0 is shown in the header of $home/hsc/rerun/[rerin]/deepCoadd/[filter]/[tract]/[patch].fits
# HSC pipeline uses the following value to all filters;
#     FLUXMAG0 = 63095734448.0194 ~ 27 mag
http://hsc.mtk.nao.ac.jp/pipedoc_e/e_usage/multiband.html

EXAMPLES

get_stamps_hsc.py -i ydrops_test.csv  --bands Y

get_stamps_hsc.py -i decals_drops.csv --bands Y -u banados --size 1 --size_units arcmin

get_stamps_hsc.py -i qso.txt --bands Y -u banados --size 1 --size_units arcmin --data_release pdr2_wide --imtype coadd


get_stamps_hsc.py -i qso.txt --bands Y -u banados --size 1 --size_units arcmin --data_release pdr3_wide --imtype coadd

        """


def authorize_url(url, user):
    """
    Necessary when the url are password protected
    """
    auth_user = user
    if user in users.keys():
        auth_passwd=users[user]
        print("Using saved password for user: ", user)
    else:
        auth_passwd=getpass.getpass('Password: ')
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, url, auth_user, auth_passwd)
    authhandler = urllib.request.HTTPBasicAuthHandler(passman)
    opener = urllib.request.build_opener(authhandler)
    urllib.request.install_opener(opener)


#function that build the url from the coordinate
def build_url(ra, dec, size, size_units, imtype, bands, dr):
    basedr = dict(dr1="https://hsc-release.mtk.nao.ac.jp/das_quarry/cgi-bin/quarryImage?",
                  pdr2_wide="https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?",
                   pdr2_dud="https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?",
                   pdr3_wide="https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?",
                    pdr3_dud="https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?")
    base = basedr[dr]
    ra = "ra={:0}&".format(ra)
    dec = "dec={:0}&".format(dec)
    sw = "sw={0:s}&".format(size+size_units)
    sh = "sh={0:s}&".format(size+size_units)
    imgtype = 'type={0:s}&image=on&'.format(imtype)
    bands = "filter=HSC-{0:s}&".format(bands)

    url = base + ra + dec + sw + sh + imgtype + bands

    if dr == 'dr1':
        url+='tract=&rerun='
    else:
        url+='tract=&rerun='+ dr

    return url



def get_name(d):
    '''
    get name column if it is called name or ps1_name
    '''
    try:
        n = d['ps1_name']
    except KeyError:
        n = d['name']

    return n



def lowercase_table_keywords(data):
    """
    Receives a Table or fits file with column names.
    They are all lowercased and the table is returned

    """

    for old_col, new_col in zip(data.colnames, np.char.lower(data.colnames)):
        data[old_col].name = 'tmpcol'
        data['tmpcol'].name = new_col

    return data


def read_input(filename, ext=0):
    '''
    Read either text, csv of fits inputs for several PS1 codes.
    If the file is a fits file, by default reads the extension 0.
    Otherwise it must be specified.
    '''
    if filename.endswith('fits'):
        data=fits.getdata(filename, ext=ext)
        data=Table(data)
    elif filename.endswith('csv'):
        data=ascii.read(filename, delimiter=',')

    else:
        data=ascii.read(filename)

    return data


#function that take the arguments in input
def parse_arguments():

    parser = argparse.ArgumentParser(
        description='''

	Get a list with coordinates and ps1 name and extract the decals image,
	as a ps1_name_decals.fits file.

        ''',
	formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('-i','--input', required=True, type=str,
                            help='File containing\
                            the target coordinates and ps1 name\
		            in columns called\
                            ra, dec. and ps1_name or name' )

    parser.add_argument('--size', required=False, type=str,
                        default='2',
                            help='The (horizontal) size of the image to download. See size_units' )

    parser.add_argument('--size_units', required=False, type=str,
                        default='amin', choices=['amin', 'arcmin', 'asec', 'arcsec'],
                            help='The units of the image size ' )

    parser.add_argument('--bands', required=False, type=str,
                        default="Y", choices=["G", "R", "I", "Z", "Y"],
                            help='Filter' )

    parser.add_argument('--imtype', required=False, type=str,
                        default='coadd', choices=['coadd', 'warp'],
                            help='File type ' )

    parser.add_argument('--data_release', required=False, type=str,
                        default="pdr3_wide", choices=["dr1","pdr2_wide",
                         "pdr2_dud", "pdr3_wide", "pdr3_dud"],
                            help='Data release [dr wide or ultra-deep,dr1-wide]' )


    parser.add_argument('-e','--ext', required=False, default=None, type=int,
                            help='If a fits file, the extension to read. None otherwise' )

    parser.add_argument('-u', '--user', default='banados', type=str,
                        help='HSC account username')


    return parser.parse_args()

if __name__ == '__main__':
    args=parse_arguments()

    if args.ext is not None:
        data = read_input(args.input, ext=args.ext)
    else:
        data = read_input(args.input)

    data = lowercase_table_keywords(data)
    ra = data['ra']
    dec = data['dec']
    #ps1name = data['ps1_name']
    ps1name = get_name(data)
    i = 0.
    j = 0

    for ra, dec, ps1name_i in zip(ra, dec, ps1name):
        ps1name_i = ps1name_i.strip()
        print("*" * 30)
        print("Getting HSC image for ", ps1name_i)
        print("bands: ", args.bands)
        i = i+1.
        #build_url(ra, dec, size, size_units, imtype, bands, dr):
        url = build_url(ra, dec, size=args.size, size_units=args.size_units, imtype=args.imtype, bands=args.bands, dr=args.data_release)
        print("=" * 30 )
        print("Obtaining image from")
        print(url)

        authorize_url(url, args.user)


        try:
            datafile = urllib.request.urlopen(url)
            file = datafile.read()
            #name_fits = '{0}_HSC_{1}_{2}_{3}.fits'.format(ps1name_i,
             #           args.data_release, args.imtype, args.bands)
            name_fits = 'cutout_{0}_{1}_{2}_HSC-{3}.fits'.format(i,
                        ra, dec, args.bands)

            output = open(name_fits,'wb')
            output.write(file)
            output.close()
            print(name_fits, " created")
        except (IncompleteRead, HTTPError) as err:
            print(err)
            msg = "No HSC coverage for " + ps1name_i + " in bands: " + args.bands
            print(msg)
            j+=1



    print("Objects without images in HSC in some bands: ", j, " of ", i)
