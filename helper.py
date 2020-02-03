"""
Image processing helper functions

@author: Nanyan "Rosalie" Zhu and Chen "Raphael" Liu
"""

import numpy as np
import skimage.exposure as exposure
from skimage.util.shape import view_as_windows
import itertools
import warnings

warnings.filterwarnings('ignore')

def rescaling(img, lower_fraction, upper_fraction, bins = 65536, out_range = (0, 2**16 - 1)):
    '''
    lower_fraction and upper_fraction defines the rescaling range. This algorithm will map the intensity range you selected to the out_range you defined and the dtype of your original data will be kept the same.

    lower_fraction: float, [0, 1], the lower boundary of your selected intensity range. For instance, if you use 0.2, then the lowest 20% of all intensities will be mapped to the minimum of the out_range.

    upper_fraction: float, [0, 1], the upper boundary of your selected intensity range. For instance, if you use 0.9, then the highest 10% of all intensities will be mapped to the maximum of the out_range.

    bins: define bin number of the mapping histogram.

    out_range: the desired intensity range of output image.
    '''

    dtype = img.dtype.type
    
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    
    if lower_fraction == None:
        min_threshold = bins[0]
    else:
        min_idx = np.where(img_cdf < lower_fraction)[0][-1]
        min_threshold = bins[min_idx]
    if upper_fraction == None:
        max_threshold = bins[-1]
    else:
        max_idx = np.where(img_cdf > upper_fraction)[0][0]
        max_threshold = bins[max_idx]
    
    better_contrast = exposure.rescale_intensity(img, in_range = (min_threshold, max_threshold), out_range = out_range)
    
    return np.array(better_contrast, dtype = dtype)

def extract_patches_given_center(img, center_x, center_y, patch_x, patch_y):
    '''
    Extract a patch_x by patch_y patch from the image surrounding the center pixel at location (center_x, center_y).
    Assumes mirror padding, i.e., use the pixels from the other extremity of the image if the selected center is at the edge or corner of the image.
    '''
    
    dtype = img.dtype.type
    
    img_dimension = img.shape
    
    try:
        assert patch_x % 2 == 1 and patch_y % 2 == 1
    except:
        print('Please make sure the patch length and width are odd numbers.')

    patch_delta_x = int((patch_x - 1) / 2)
    patch_delta_y = int((patch_y - 1) / 2)

    try:
        assert patch_x <= img_dimension[0] and patch_y <= img_dimension[1]
    except:
        print('Do you really want a patch that is larger than the image? We are not smart enough to do that for you.')
        
    large_3x3_duplicate_img = np.hstack((np.vstack((img, img, img)), np.vstack((img, img, img)), np.vstack((img, img, img))))
    padded_img = large_3x3_duplicate_img[img_dimension[0] - patch_delta_x : 2 * img_dimension[0] + patch_delta_x, \
                                         img_dimension[1] - patch_delta_y : 2 * img_dimension[1] + patch_delta_y]
    
    patch = view_as_windows(padded_img, (patch_x, patch_y))[center_x, center_y]
    
    return patch

def extract_all_patches(img, patch_x, patch_y):
    '''
    Extract all patch_x by patch_y patches from the image surrounding the each pixel in the image.
    Assumes mirror padding, i.e., use the pixels from the other extremity of the image if the selected center is at the edge or corner of the image.
    '''
    
    img_dimension = img.shape
    
    try:
        assert patch_x % 2 == 1 and patch_y % 2 == 1
    except:
        print('Please make sure the patch length and width are odd numbers.')

    patch_delta_x = int((patch_x - 1) / 2)
    patch_delta_y = int((patch_y - 1) / 2)

    try:
        assert patch_x <= img_dimension[0] and patch_y <= img_dimension[1]
    except:
        print('Do you really want a patch that is larger than the image? We are not smart enough to do that for you.')
        
    large_3x3_duplicate_img = np.hstack((np.vstack((img, img, img)), np.vstack((img, img, img)), np.vstack((img, img, img))))
    padded_img = large_3x3_duplicate_img[img_dimension[0] - patch_delta_x : 2 * img_dimension[0] + patch_delta_x, \
                                         img_dimension[1] - patch_delta_y : 2 * img_dimension[1] + patch_delta_y]
    
    all_patches = view_as_windows(padded_img.copy(), (patch_x, patch_y))
    
    return all_patches

# These are from online sources: https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python and https://stackoverflow.com/questions/38054593/zip-longest-without-fillvalue.

def grouper(n, iterable):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_discard_gen(*args)

def zip_discard_gen(*iterables, sentinel=object()):
    return (tuple([entry for entry in iterable if entry is not sentinel])
            for iterable in itertools.zip_longest(*iterables, fillvalue=sentinel))