#!/usr/bin/env python


import os
# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import math
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import layers
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import os, glob
from itertools import compress
from tqdm import tqdm
import time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
import pandas as pd

def cropping(img_band, header_img, filename, npixels = 60):
    ra = float(filename[filename.find('ra')+3:filename.find('dec')-1])
    dec = float(filename[filename.find('dec')+4:filename.find('HSC')-1])
    position = SkyCoord(ra, dec, unit = 'deg', frame='icrs')
    wcs = WCS(header_img)
    size = u.Quantity(img_band.shape, u.pixel)
    cutout = Cutout2D(img_band, position, (npixels,npixels), wcs=wcs)
    return cutout.data

def normalization_perImag(im_5):
    norm_img_t = (im_5-np.min(im_5))/(np.max(im_5)-np.min(im_5))
    return norm_img_t

def total_norm(all_im_5, np_norm = 10):
    npixel = all_im_5.shape[1]
    n_in = int((npixel-np_norm)/2)
    i_max = np.max(all_im_5[:, n_in:n_in+np_norm, n_in:n_in+np_norm,2])
    i_min = np.min(all_im_5[:, n_in:n_in+np_norm, n_in:n_in+np_norm,2])
    print('Maximum flux: {} and mininum flux {} at i band.'.format(i_max, i_min))
    norm_all_im_5 = all_im_5/i_max

    return norm_all_im_5

def bad_pixels(image, name):
    if np.isnan(image).any() == True:
        print('Nan values in: ', name)
        image[np.isnan(image)] = 0
    else:
        pass
    return image
    
def load_tensors(path, npixels = 60, np_norm = 10, substring = None):
    names = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        # Substring for labeled dataset
        if substring != None and filename.find(substring) != -1:
            names.append(filename[:-6]) 
        elif substring != None and filename.find(substring) == -1:
            print('Substring doesnt exist. Please try a different one!')
        # None substring for unlabeled train set
        elif substring == None and filename.find('QSOs') == -1 and \
            filename.find('Gals') == -1 and filename.find('Stars') == -1:
            names.append(filename[:-6])
            
    unique_names = np.unique(names)
    print('Creating a tensor with {} sources. '.format(len(unique_names)))
    print('Masking bad pixels by changing nan values to zero.')
    print('Cropping each image from size {} to {} arcsecs.'.format(4.66666666666311E-05*84*3600/2, 4.66666666666311E-05*npixels*3600/2))
    print('Normalizing tensor by the maximum of the i-flux distribution in {} arcsec around the source. '.format(4.66666666666311E-05*np_norm*3600/2))

    all_im_5 = np.zeros((len(unique_names),npixels, npixels, 5))
    start_time = time.time()
    for j in tqdm(range(len(unique_names))):
        bands = ['G', 'R', 'I', 'Z', 'Y']
        for b in range(len(bands)):
            fits_band = get_pkg_data_filename(path + unique_names[j] + bands[b] + '.fits')
            img_band = fits.getdata(fits_band, ext = 1)
            header_img = fits.getheader(fits_band, ext = 1)
            img_band_bp = bad_pixels(img_band, unique_names[j] + bands[b] + '.fits')
            img_band_crop = cropping(img_band_bp, header_img,  unique_names[j])
            all_im_5[j,:,:,b] = img_band_crop
    norm_all_im_5 = total_norm(all_im_5, np_norm)
    if substring == None:
        np.save('train_set.npy', norm_all_im_5)
    else:
        return norm_all_im_5


def add_labeled_data(path, class_names, npixels = 60, np_norm = 10):
    labels = []
    for i in range(len(class_names)):
        lab_set = load_tensors(path, npixels, np_norm, class_names[i])
        n_set = len(lab_set)
        labels = labels + [i]*n_set
        if i == 0:
            labeled_set = lab_set
        else:
            labeled_set = np.concatenate((labeled_set, lab_set), axis = 0)
    labels = np.array(labels)
    np.save('labeled_set.npy', labeled_set)
    np.save('labels.npy', labels)

print('Creating and saving tensor of images.......................')
load_tensors('HSC/', 60, 10) 
print('Creating and saving tensors of labeled images and labels .......................')
add_labeled_data('HSC/', ['QSOs', 'Stars', 'Gals'], 60, 10)

