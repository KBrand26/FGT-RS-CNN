from distutils.command.build import build
import warnings
import numpy as np
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import pyvo as vo
import requests
from matplotlib import pyplot as plt
import pandas as pd
from skimage.morphology import dilation, square
from scipy.ndimage import binary_fill_holes
from skimage.transform import rotate
import os
import warnings

def norm_image(img):
    """Perform normalization of a given image

    Args:
        img (ndarray): The image that should be processed

    Returns:
        ndarray: The processed image
    """
    bot = np.min(img)
    top = np.max(img)
    norm = (img - bot)/(top - bot)
    return norm

def construct_true_aux_mix():
    """
    Constructs auxiliary feature vectors that include the true auxiliary labels where possible.
    
    Returns:
        ndarray: Returns the 'true' auxiliary feature vectors.
    """
    if not os.path.exists('../../data/galaxy_X.npy'):
        warnings.warn('Galaxy data not found in data directory. Please ensure that you generate this data first.', RuntimeWarning)
        return
    
    X = np.load('../../data/galaxy_X.npy')
    
    if not os.path.exists('../../aux_data/features.txt'):
        warnings.warn('Manually extracted labels not found in aux_data directory. Please ensure that you generate this data first.', RuntimeWarning)
        return
    
    manual_df = pd.read_csv('../../aux_data/features.txt')
    y_man_aux = []
    disagree = 0
    
    for i in range(len(X)):
        if i % 100 == 0:
            print(f'{i} images processed...')
        elem = manual_df[manual_df['ID'] == i]
        bent = elem['Curved'].values[0]
        manual_cores = elem['Cores'].values[0]
        img = X[i]
        fr_ratio, cores, core_frac, pb = process_input(img, i)
        if (bent != pb):
            disagree += 1
        feats = [bent, fr_ratio, manual_cores, core_frac]
        
        y_man_aux.append(feats)
    
    y_man_aux = np.array(y_man_aux)
    
    if not os.path.exists('../../data/'):
        os.makedirs('../../data/')
    
    np.save('../../data/galaxy_y_manual_aux.npy', y_man_aux)
    return y_man_aux 

def process_input(img, i):
    '''
    Processes the given image and extracts the necessary features.
    
    Parameters
    ----------
    img : ndarray
        The image from which to extract the features.
    i : int
        The index corresponding to the image.
        
    Returns
    -------
    tuple
        A tuple containing the FR ratio, core count, ratio of core size to galaxy and whether a galaxy has a curved shape
    '''
    # Preprocess image
    img = norm_image(img)        
    threshed, basic = thresh_image(img, testing = False)
    
    # Extract cores and other important information regarding the cores
    cores, inter_dist, core_pixels, total_pixels = detect_cores(img, threshed)
    core_frac = core_pixels/total_pixels
    
    if not basic:
        # If basic thresholding was not used, additional processing is necessary to account for the presence of noise
        threshed = prep_bent_class(img, threshed)
        threshed = remove_small_components(threshed, thresh=10)
    rotated = rotate_axes(threshed)
    
    gal_size = max(rotated[0, :]) - min(rotated[0, :])
    fr_ratio = np.round(inter_dist/gal_size, 2)
    
    v_size = calc_vertical_size(rotated)
    h_size = calc_horizontal_size(rotated)
    if v_size == 0:
        print(f'Zero vertical size detected for image {i}')
        return cores, 0, h_size, v_size, False, threshed, rotated
    ratio = h_size/v_size
    pb = potential_bent(v_size, h_size)
    
    return fr_ratio, cores, core_frac, pb

def thresh_image(img):
    '''
    Thresholds image to remove noise and background pixels.
    
    Parameters:
    -----------
    img : ndarray
        The image that needs to be thresholded.
        
    Returns:
    --------
    tuple
        Tuple containing the thresholded image, as well as a flag that indicates whether basic thresholding was used. 
        
    '''
    # Generate histogram of image
    bins = np.linspace(0, 1, 256)
    binned = np.digitize(img, bins)
    
    # Determine which thresholding approach to use 
    elig, s, std = eligible_for_basic_thresh(img, binned)
    if elig:
        # Perform basic thresholding
        threshed = basic_thresh(img, s)
        threshed = remove_small_components(np.copy(threshed), 5)
        return threshed, True
    else:
        # There is a lot of noise in the image. Apply more sophisticated thresholding.
        
        val_dist = find_num_vals(img)
        
        # Extract quantiles
        q98 = np.quantile(val_dist, 0.985)
        q96 = np.quantile(val_dist, 0.96)
        
        # Pixels along the border of the galaxy will have a wider variety of pixel values
        # from galaxy pixels, noise pixels and background pixels. Thus we first extract pixels that
        # have a large number of pixel values in their neighbourhood.
        first = np.where(val_dist >= q98, 1, 0)
        # Remove artefacts
        first = remove_small_components(np.copy(first), 10)
        
        # Use dilations to `grow' the extracted pixels to include connected pixels that still have a large range of values
        # in their neighbourhood.
        exten = np.where(val_dist >= q96, 1, 0)
        se = square(3)
        
        prev = np.copy(first)
        cur = np.copy(first)
        cur = np.multiply(dilation(cur, se), exten)
        while (cur - prev).any():
            prev = np.copy(cur)
            cur = np.multiply(dilation(cur, se), exten)
            
        # Remove artefacts
        cleaned = remove_small_components(np.copy(cur))
        # Fill holes that are surrounded by extracted pixels. This ensures that the galaxy pixels in the center
        # of the galaxy are also extracted.
        threshed = binary_fill_holes(cleaned)
        
        return threshed, False

def find_num_vals(img):
    '''
    This function creates a new matrix where each element represents the number of unique pixel values in the
    9x9 neighbourhood of the corresponding pixel in the given image.
    
    Parameters:
    -----------
    img : ndarray
        The image to use to create the new matrix.
        
    Returns:
    --------
    ndarray
        The matrix representing the unique value counts in the neighbourhoods of the given image.
    '''
    vals_dist = np.zeros_like(img)
    size = 9
    step = size//2
    for r in range(step, img.shape[0]-step):
        for c in range(step, img.shape[1]-step):
            # Extract neighbourhood
            neigh = img[r-step:r+step+1, c-step:c+step+1]
            
            # Create local neighbourhood histogram
            bins = np.linspace(0, 1, 256)
            binned = np.digitize(neigh, bins)
            
            # Count how many bins are represented in the neighbourhood
            vals = len(np.unique(binned))
            vals_dist[r, c] = vals
    return vals_dist

def eligible_for_basic_thresh(img, bins):
    '''
    This function investigates the histogram corresponding to an image to determine which thresholding approach to apply
    
    Parameters:
    -----------
    img : ndarray
        The image that is being tresholded.
    bins : ndarray
        The histogram that corresponds to the given image.
        
    Returns:
    --------
    tuple
        Tuple indicating whether basic thresholding should be used, the number of bins with more than 100 pixels and
        the standard deviation of the image.
    '''
    # Count how many bins have more than 100 pixels
    counts = np.array([len(img[bins == i]) for i in range(256)])
    s = np.sum(counts > 100)
    
    std = np.std(img)
    return s < 17 or std < 0.035, s, std

def basic_thresh(image, s):
    '''
    Applies the basic thresholding algorithm to the given image.
    
    Parameters:
    -----------
    image : ndarray
        The image that needs to be thresholded.
    s : int
        Number of histogram bins that had more than 100 pixels.
    
    Returns:
    --------
    ndarray
        The thresholded image.
    '''
    
    if s <= 14:
        # Noise is almost non-existant, use a static threshold
        return np.where(image > 0.1, 1, 0)
    # Identify relevant quantiles
    q985 = np.quantile(image, 0.985)
    q98 = np.quantile(image, 0.98)
    
    # Threshold at high quantile to ensure only galaxy pixels are extracted
    first = np.where(image > q985, 1, 0)
    
    # Remove any small thresholding artefacts from background
    first = remove_small_components(np.copy(first), 10)
    
    # `Grow` the extracted pixels with dilations to include connected pixels that are above the secondary threshold
    exten = np.where(image >= q98, 1, 0)
    se = square(3)
    
    prev = np.copy(first)
    cur = np.copy(first)
    # Dilation adds surrounding pixels, multiplication removes pixels that are not larger than 98th quantile.
    cur = np.multiply(dilation(cur, se), exten)
    while (cur - prev).any():
        prev = np.copy(cur)
        cur = np.multiply(dilation(cur, se), exten)

    return cur

def remove_small_components(img, thresh=110):
    '''
    This function finds connected components in the given image and removes any
    components that are smaller than the given threshold.
    
    Parameters:
    -----------
    img : ndarray
        The image from which to remove the small components
        
    thresh : int
        The threshold to use when determining whether a component is too small.
        
    Returns:
    --------
    ndarray
        The image after all of the small components were removed
    '''
    total = np.sum(img)
    if total < thresh:
        # If the total count of pixels in the image is smaller than the threshold size, no thresholding is necessary.
        return img
    
    new_img = np.zeros_like(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] == 1:
                comp, size = find_connected_component(img, r, c)
                img -= comp
                if size > thresh:
                    new_img += comp
    return new_img

def find_connected_component(img, r, c):
    '''
    Finds all the pixels that belong to a component in an image, given the coordinates of a pixel
    that is known to belong to that components
    
    Parameters:
    -----------
    img : ndarray
        The image within which to look for the component
    r : int
        The row corresponding to the pixel from the component.
    c : int
        The column corresponding to the pixel from the component.
        
    Returns:
    --------
    tuple
        A tuple that contains the extracted component, as well as its size.
    '''
    comp = np.zeros_like(img)
    # Starting pixel for the connected component
    comp[r, c] = 1.0

    se = square(3)

    # Repeatedly `grow' the connected component to include surrounding pixels
    prev_comp = np.copy(comp)
    # Remove background and noise pixels by multiplying with the thresholded image.
    comp = np.multiply(dilation(comp, se), img)

    while (comp - prev_comp).any():
        prev_comp = np.copy(comp)
        comp = np.multiply(dilation(comp, se), img)

    return comp, np.sum(comp)