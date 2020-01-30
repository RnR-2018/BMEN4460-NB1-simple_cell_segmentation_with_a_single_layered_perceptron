"""
Image processing helper functions

@author: Nanyan "Rosalie" Zhu and Chen "Raphael" Liu
"""

import numpy as np
import skimage.exposure as exposure

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