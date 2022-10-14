import warnings
import numpy as np
import glob
import argparse
import pandas as pd
from skimage.morphology import dilation, square
from scipy.ndimage import binary_fill_holes
from skimage.transform import rotate
import os
from astropy.io import fits
import warnings
from train_models import probe_dir
from sklearn.model_selection import train_test_split

def load_galaxy_data():
    """
    Loads the galaxy data and prepares the necessary npy files for use in model training and evaluation.
    """
    if not os.path.exists('FITS/'):
        warnings.warn('Please download the FITS data from Zenodo first', RuntimeWarning)
        return
    
    rootdir = 'FITS/'
    subdirs = ['BENT_NAT', 'BENT_WAT', 'COMP', 'FRI', 'FRII']
    label_dict = {'BENT_NAT': [1, 0, 0, 0], 'BENT_WAT': [1, 0, 0, 0], 'COMP': [0, 1, 0, 0], 'FRI': [0, 0, 1, 0], 'FRII': [0, 0, 0, 1]}
    
    X = []
    X_aug = []
    y = []
    y_aug = []
    y_aux  = []
    for sub in subdirs:
        # Loop through directories
        label = label_dict[sub]
        files = glob.glob(rootdir + sub + '/*.fits')
        for file in files:
            # Read files, focus on center of image
            if file == 'FITS/BENT_WAT/BENT_WAT_106.fits':
                # File is corrupt
                continue
            img = fits.open(file)[0].data
            focus = img[75:225, 75:225]
            focus = (focus - 0)/1. # Data is read in as big endian which is incompatible with skimage. This calculation should not alter data, but fixes buffer type.
            
            # Extract features from sample
            fr_ratio, cores, core_frac, pb = process_input(focus, -1)
            bent = 1 if pb else 0
            feats = [bent, fr_ratio, cores, core_frac]
            
            X.append(focus)
            y.append(label)
            y_aux.append(feats)
    # Save arrays
    X = np.array(X)
    y = np.array(y)
    y_aux = np.array(y_aux)
    
    probe_dir('data/')
    
    np.save('data/galaxy_X.npy', X)
    np.save('data/galaxy_y.npy', y)
    np.save('data/galaxy_y_aux.npy', y_aux)

def augment_galaxy_data(img, r=60):
    """
    This function is used to create additional, rotated copies of the given image.

    Args:
        img (ndarray): The image that has to be augmented.
        r (int, optional): The interval between the angles used to rotate the image. Defaults to 60.

    Returns:
        list: A list containing the augmented samples after rotating the given image.
    """    
    augmented = []
    for deg in range(0,360,r):
        augmented.append(rotate(img.copy(),deg))
        
    return augmented

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
    Constructs auxiliary feature vectors that include the manually extracted auxiliary labels where possible.
    """
    if not os.path.exists('data/galaxy_X.npy'):
        warnings.warn('Galaxy data not found in data directory. Please ensure that you generate this data first.', RuntimeWarning)
        return
    
    X = np.load('data/galaxy_X.npy')
    
    if not os.path.exists('aux_data/features.txt'):
        warnings.warn('Manually extracted labels not found in aux_data directory. Please ensure that you generate this data first.', RuntimeWarning)
        return
    
    manual_df = pd.read_csv('aux_data/features.txt')
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
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    
    np.save('data/galaxy_y_manual_aux.npy', y_man_aux)

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
    threshed, basic = thresh_image(img)
    
    # Extract cores and other important information regarding the cores
    cores, inter_dist, core_pixels, total_pixels = detect_cores(img, threshed)
    core_frac = core_pixels/total_pixels
    
    if not basic:
        # If basic thresholding was not used, additional processing is necessary to account for the presence of noise
        threshed = prep_bent_class(img, threshed)
        threshed = remove_small_components(threshed, thresh=10)
   
    # Standardize the rotation of the thresholded pixels.
    rotated = rotate_axes(threshed)
    
    # Calculating the distance between the minimum and maximum coordinate along the first principal component.
    gal_size = max(rotated[0, :]) - min(rotated[0, :])

    # Calculate FR Ratio
    fr_ratio = np.round(inter_dist/gal_size, 2)
   
    # Calculate the size of the galaxy along the first and second principal component.
    v_size = calc_vertical_size(rotated)
    h_size = calc_horizontal_size(rotated)
    
    # If the vertical size is zero we don't need to do any further processing.
    if v_size == 0:
        print(f'Zero vertical size detected for image {i}')
        return cores, 0, h_size, v_size, False, threshed, rotated

    # Calculate the ratio between the size of the galaxy along the first and second principal component
    ratio = h_size/v_size

    # Determine whether the galaxy might have a bend in it
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

def detect_cores(img, threshed):
    '''
    This function is used to identify the cores in a given image.

    Parameters:
    -----------
    img : ndarray
        The image within which to look for cores.
    threshed : ndarray
        The thresholded image that corresponds to img.

    Returns:
    --------
    tuple
        A tuple that contains the number of cores, the distance between the cores,
        the number of pixels in the cores and the total number of pixels in the galaxy. 
    '''
    # Replace the extracted galaxy pixels with their original values
    processed = np.where(threshed, img, 0)

    # Find the standard deviation of the non-zero pixels.
    std = galaxy_deviation(processed)

    # Extract the cores from the image
    hard_thresh = segment_cores(processed, std)

    # Count the number of pixels in the cores
    core_pixels = np.sum(hard_thresh)

    # Count the number of pixels in the galaxy
    bin_total = processed > 0
    total_pixels = np.sum(bin_total)

    # Count the cores and calculate the distance between them
    cores, inter_dist = process_cores(hard_thresh, img)
    
    return cores, inter_dist, core_pixels, total_pixels

def galaxy_deviation(img):
    '''
    This function calculates the deviation of the pixels that have been identified as
    belonging to the galaxy.

    Parameters:
    -----------
    img : ndarray
        The thresholded image for which the standard deviation should be calculated.

    Returns:
    --------
    float
        The standard deviation of the thresholded pixels in the given image.
    '''
    thresh = np.copy(img)

    # Replace zeros with NaN
    tmp = np.where(thresh == 0, np.nan, thresh)

    # Calculate standard deviation without considering NaN elements
    std = np.nanstd(tmp)

    return std

def segment_cores(img, std):
    '''
    This function is used to extract the pixels that most likely correspond to cores from the given image.

    Parameters:
    -----------
    img : ndarray
        The image from which to extract the cores
    std : float
        The standard deviation of the non-zero pixels in the given image.

    Returns:
    ndarray
        A thresholded version of the given image, containing only pixels believed to belong to the cores.
    '''
    threshed = np.copy(img)

    # Replace zero pixels with NaN
    tmp = np.where(threshed == 0, np.nan, threshed)
    
    # Determine which quantile to use as the threshold to extract the noise.
    # The std gives an indication of how heterogeneous the pixel values are.
    # Wider varieties of pixel values require a lower quantile to extract all of the possible core pixels.
    if std > 0.20:
        q = np.nanquantile(tmp, 0.80)
    elif std > 0.13:
        q = np.nanquantile(tmp, 0.93)
    else:
        q = np.nanquantile(tmp, 0.98)

    return threshed > q

def process_cores(thresh, img=None):
    '''
    This function is used to count the number of cores in the thresholded image and
    to calculate the average distance between the cores.

    Parameters:
    -----------
    thresh : ndarray
        The thresholded image that only contains pixels that are likely to be part of cores.
    img : ndarray
        The image corresponding to thresh with the original pixel intensities.

    Returns:
    --------
    tuple
        A tuple containing the number of cores and the average distance between them.
    '''
    val_img = np.where(thresh, 1, 0)
    coords = []

    count = 0
    for r in range(val_img.shape[0]):
        for c in range(val_img.shape[1]):
            if val_img[r, c] == 1:
                # Expand pixel to find the corresponding core
                comp, size = find_connected_component(val_img, r, c)

                # Approximate the center of the core
                coords.append(reduce_component(comp, img))

                # Remove the core from the image so it is not found again
                val_img -= comp

                count += 1
    
    if count == 1:
        # If the count is one, just take a third of the component size as the inter-core distance
        inter_core_size = calc_comp_size(comp)
    else:
        # If there are more than 1 core, calculate the average distance between the cores.
        inter_core_size = calc_core_dist(coords)
    return count, inter_core_size

def reduce_component(comp, img):
    '''
    This function is used to reduce the given component to a single pixel.

    Parameters:
    -----------
    comp : ndarray
        An image containing the component that needs to be reduced.
    img : ndarray
        The original image from which the component was extracted.

    Returns:
    tuple
        A tuple containing the row, column and value of the pixel identified as the center of the component.
    '''
    # Replace the component pixels with their original intensities.
    masked = np.where(comp, img, 0)

    # Find the maximum value in the component.
    max_val = np.amax(masked)

    # Find the coordinates of the brightest pixel.
    r, c = np.where(masked == max_val)

    return (r[0], c[0], max_val)

def calc_comp_size(comp):
    '''
    This function is used to calculate the size of the given component. It returns a third of this value
    which represents an approximation of the distance between the two cores that might be overlapping.

    Parameters:
    -----------
    comp : ndarray
        The extracted component for which the size should be calculated.

    Returns:
    --------
    float
        A third of the size of the component.
    '''
    # Find the coordinates of the pixels in the component
    rs, cs = np.where(comp == 1)
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)

    # Identify the distance between the min and max row and min and max column. Use the biggest distance as the size of the component.
    max_dist = np.max([abs(max_r - min_r), abs(max_c - min_c)])

    return np.round((1/3)*max_dist)

def calc_core_dist(coords):
    '''
    This function is used to calculated the distance between the cores.

    Parameters:
    -----------
    coords: list
        A list containing the coords and value of the center pixels in each identified core.
    
    Returns:
    --------
    float
        The average distance between the cores.
    '''
    total = 0
    for i in range(len(coords)):
        cur = coords[i]
        min_dist = 9999999999999
        first = True
        for j in range(len(coords)):
            # Calculate the distance between core i and each other core in the list
            if j == i:
                continue
            neigh = coords[j]
            dist = calc_euc(cur, neigh)
            if first:
                min_dist = dist
                first = False
            elif dist < min_dist:
                min_dist = dist

        # Keep track of the distances between all of the cores
        total += min_dist
    # Return the average distance
    return np.round(total/len(coords))

def calc_euc(coord1, coord2):
    '''
    This function calculate the euclidean distance between two coordinates.

    Parameters:
    -----------
    coord1 : tuple
        The tuple containing the first coordinate
    coord2 : tuple
        The tuple containing the second coordinate

    Returns:
    --------
    float
        The Euclidean distance between the two coordinates.
    '''
    c1 = np.array([coord1[0], coord1[1]])
    c2 = np.array([coord2[0], coord2[1]])
    return np.linalg.norm(c1 - c2)

def prep_bent_class(img, threshed):
    '''
    This function is used to finetune thresholding results in the presence of significant noise.
    This is necessary to ensure that the bent feature can be extracted accurately.

    Parameters:
    -----------
    img : ndarray
        The original image.
    threshed : ndarray
        The corresponding thresholded image that needs to be cleaned.

    Returns:
    --------
    ndarray
        Thresholded image after finetuning.
    '''
    # Replace pixels with their original intensities.
    processed = np.where(threshed, img, 0)

    # Determine how many standard deviations each pixel is from the mean 
    zscores = calc_zscores(processed)

    # Keep pixels that have a zscore larger than the median zscore
    return thresh_median(zscores, binary=True)

def calc_zscores(threshed_img):
    '''
    This function is used to calculate the number of standard deviations that each pixel is from the
    mean pixel intensity.

    Parameters:
    -----------
    threshed_img : ndarray
        The thresholded image for which to calculate z-scores.

    Returns:
    --------
    ndarray
        Array containing zscores that correspond to each pixel in the given image.
    '''
    thresh = np.copy(threshed_img)
    # Replace zero values with NaN
    tmp = np.where(thresh == 0, np.nan, thresh)

    # Calculate standard deviation of pixels extracted during thresholding.
    std = np.nanstd(tmp)

    # Calculate Z-scores for given image.
    zscores = (thresh - np.nanmean(tmp))/std         
    return zscores

def thresh_median(img, binary=False):
    '''
    This function thresholds the given image and only keeps pixel intensities that lie above the median intensity

    Parameters:
    -----------
    img : ndarray
        The image that should be thresholded
    binary : boolean
        A flag that indicates whether the output should be a binary image.

    Returns:
    --------
    ndarray
        The image after extracting pixels larger than themedian
    '''
    min_val = img.min()
    tmp = np.where(img == min_val, np.nan, img)
    q50 = np.nanquantile(tmp, 0.5)
    if binary:
        return np.where(img < q50, 0, 1)
    else:
        return np.where(img < q50, min_val, img)

def rotate_axes(img):
    '''
    This function is used to standardize the rotation of the galaxy pixels in the given image.

    Parameters:
    -----------
    img : ndarray
        The image for which rotation needs to be standardized.

    Returns:
    --------
    ndarray
        An array of the new pixel coordinates after standardizing rotation
    '''
    # Extract the coordinates of the galaxy pixels
    coords = [[], []]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                coords[0].append(j)
                coords[1].append(i)

    coords = np.array(coords)

    # Center the image at the mean coordinate (necessary before PCA)
    mean = np.mean(coords,axis=1)[:,np.newaxis]
    coords = (coords - mean)

    # Calculate the principal directions using the SVD
    u, s, vh = np.linalg.svd(coords,full_matrices=False)

    # U contains the principal components in the columns, premultiplying by this matrix
    # rotates the coordinates to align with these principal components
    rotated = u.T.dot(coords)

    return rotated

def calc_vertical_size(coords):
    '''
    This function is used to calculate the distance between the minimum and maximum coordinate
    along the second principal component. This can be seen as calculating the size of the galaxy along the vertical axis.

    Parameters:
    -----------
    coords : ndarray
        An array of galaxy pixel coordinates after standardising the rotation of the galaxy.

    Returns:
    --------
    float
        The distance between the minimum and maximum coordinate along the second principal component.
        
    '''
    start_row = coords[1, :].min()
    end_row = coords[1, :].max()

    return end_row - start_row

def calc_horizontal_size(coords):
    '''
    This function is used to calculate the distance between the minimum and maximum coordinate
    along the first principal component. This can be seen as calculating the size of the galaxy along the horizontal axis.

    Parameters:
    -----------
    coords : ndarray
        An array of galaxy pixel coordinates after standardising the rotation of the galaxy.

    Returns:
    --------
    float
        The distance between the minimum and maximum coordinate along the first principal component.
        
    '''
    start_col = coords[0, :].min()
    end_col = coords[0, :].max()

    return end_col - start_col

def potential_bent(v_size, h_size):
    '''
    This function determines whether the galaxy contains a curve.

    Parameters:
    -----------
    v_size : float
        The size of the galaxy along the second principal component
    h_size : float
        The size of the galaxy along the first principal component

    Returns:
    --------
    boolean
        A boolean that indicates whether the galaxy might contain a curve.
    '''
    ratio = h_size/v_size

    return 0.5 < ratio < 6.7 and 14 < v_size < 54

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for experiments")
    parser.add_argument('-d', "--data", action="store_true",
                        help="""Extract the various images and class labels from the dataset and split them into
                                train, validation and test sets.""")
    args = parser.parse_args()
    config = vars(args)
    
    if config['data']:
        load_galaxy_data()
        X = np.load('data/galaxy_X.npy')
        y = np.load('data/galaxy_y.npy')
        
        X_tmp_train, X_test1, y_tmp_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=42, shuffle=True, stratify=y)
        X_train1, X_val1, y_train, y_val = train_test_split(X_tmp_train, y_tmp_train, test_size=0.11, train_size=0.89, random_state=42, shuffle=True, stratify=y_tmp_train)
        
        np.save('data/galaxy_X_train1.npy', X_train1)
        np.save('data/galaxy_y_train.npy', y_train)
        np.save('data/galaxy_X_val1.npy', X_val1)
        np.save('data/galaxy_y_val.npy', y_val)
        np.save('data/galaxy_X_test1.npy', X_test1)
        np.save('data/galaxy_y_test.npy', y_test)
        
        construct_true_aux_mix()
    
    