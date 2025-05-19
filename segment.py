#© 2025, Northwestern University. All rights reserved.

import sys
import argparse
import skimage
import cv2 as cv
import scipy
import matplotlib
import queue

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc

from skimage.filters import unsharp_mask
from skimage.morphology import convex_hull_image
from skimage.transform import resize

from datetime import datetime
from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 12})

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=False)

def invert(arr):
    inverse = (~(arr.astype('bool'))*1)
    return inverse

def crop_to_box(arr, min_x, max_x, min_y, max_y):
    crop = (arr[min_y:max_y,min_x:max_x]).astype('uint8')
    return crop

def cap_box_size(arr, percent):
    max_box_size = np.ceil(min(arr.shape)*percent)
    return max_box_size

def get_low_slope_pts(diagonal, max_rise, search_cutoff):
    low_slope_pts = [diagonal[0]]
    for i in range(1,len(diagonal)):
        if (abs(diagonal[i-1]-diagonal[i]) < max_rise) and (diagonal[i] < search_cutoff):
            low_slope_pts.append(diagonal[i])
    return low_slope_pts

def diagonal_mask_background(image_resized):
    diagonal = np.diag(image_resized).astype('int')
    rng = np.ptp(diagonal) 
    search_cutoff = np.ceil(rng*.1) 
    max_rise = np.ceil(rng * 0.01)
    ullr = get_low_slope_pts(diagonal, max_rise, search_cutoff)
    ullr_cutoff = max(ullr)
    cutoff = np.ceil(ullr_cutoff+(0.1*rng)) 
    foreground_mask = np.where(image_resized > cutoff, 1, 0)
    return foreground_mask, rng

def make_background_eq_0(arr): 
    if (arr[0][0]==0):
        return arr
    else:
        return invert(arr)

def k_means(arr, k):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    Z = arr.reshape((-1,1)) #1D
    Z = np.float32(Z)
    _,label,_ = cv.kmeans(Z,k,None,criteria, 10, flags)
    label_reshape = label.reshape(arr.shape)
    uint_img = np.array(label_reshape).astype('uint8')
    return uint_img

def make_box(array, row, col, max_size):
    bound1 = array.shape[0]-col
    bound2 = array.shape[1]-row
    max_box_len = int(min(bound1,bound2,max_size))
    fix = array[col:col+max_box_len,row:row+max_box_len]
    if (fix==1).all():
        return max_box_len
    else:
        for i in range(1, max_box_len):
            box = fix[:i,:i]
            if not (box==1).all():
                return i-1
    return i

def modify_region(array, row, col, box_size):
    array[row:row+box_size,col:col+box_size] = 0
    return array

def find_boundaries(region):
    dim1,dim2 = np.nonzero(region)
    coords = np.stack((dim2, dim1), axis=-1)    
    min_x = np.min(coords[:,0]) #min x
    max_x = np.max(coords[:,0]) #max x
    min_y = np.min(coords[:,1]) #min y
    max_y = np.max(coords[:,1]) #max y
    return min_x, max_x, min_y, max_y

def alg(arr):    
    boxes = []
    nonzero_cols, nonzero_rows = np.nonzero(np.transpose(arr))
    coords = np.vstack([nonzero_cols,nonzero_rows]).T
    max_box_size = cap_box_size(arr, 0.2)
    q = queue.Queue(maxsize = coords.shape[0])
    q.queue = queue.deque(coords)
    while not q.empty():
        elem = q.get()
        if not arr[elem[1], elem[0]] == 0: # row, col - check that there's still a one at this location
            box_size = make_box(arr, elem[0], elem[1], max_box_size) # col, row
            boxes.append([elem[0], elem[1], box_size]) # col, row
            arr = modify_region(arr, elem[1], elem[0], box_size) # row, col
    return boxes

def process(arr, comps, bins):
    image_resized = resize(arr, (128, 128),anti_aliasing=True, preserve_range=True).astype('uint8')
    sharp_init = (unsharp_mask(image_resized, radius=5, amount=5)*255).astype('uint8')
    blur = cv.GaussianBlur(sharp_init,(15,15),0)
    adapt_g = cv.adaptiveThreshold(blur,1,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
    adapt_g_foreground = make_background_eq_0(adapt_g) 
    min_x, max_x, min_y, max_y = find_boundaries(adapt_g_foreground)
    crop = crop_to_box(image_resized, min_x, max_x, min_y, max_y)
    foreground_mask, rng = diagonal_mask_background(crop)
    chull = convex_hull_image(foreground_mask, offset_coordinates = True)
    foreground_background_offset = int(np.ceil(rng*0.2))
    filtered_image = np.where(chull==False, 0, crop+foreground_background_offset) 
    sharp_2 = (unsharp_mask(filtered_image, radius=3, amount=3)*255).astype('uint8')
    blur_2 = cv.medianBlur(sharp_2,3)
    # preparing to segment
    blur_flat = blur_2.flatten()
    nonzero = blur_flat[blur_flat != 0] 
    hist, bin_edges = np.histogram(nonzero, bins)
    bin_edges = bin_edges[1:]
    peaks, _ = find_peaks(np.append(hist,0), prominence=np.max(hist)*0.05)
    num_peaks = int(peaks.shape[0])
    if (num_peaks < 2) or (abs(num_peaks-comps) >= 2): 
        K = comps + 1 
    else:
        K = num_peaks + 1 
    segmented = k_means(blur_2,K)
    return min_x, max_x, min_y, max_y, K, segmented, crop 
    
def make_plots(arr, boxes, ret, crop, min_x, max_x, min_y, max_y):
    savedir = os.getcwd()
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    axes[0].imshow(arr,cmap='gray')
    axes[0].set_title('Original')

    axes[1].imshow(arr,cmap='gray')
    rect = []
    for i in range(boxes.shape[0]):
        rect = patches.Rectangle((ret[i,0], ret[i,1]), ret[i,2], \
                                    ret[i,2], linewidth=1, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title('Output')
    
    axes[2].imshow(crop,cmap='gray',extent=[min_x,max_x,max_y,min_y])
    rect = []
    for i in range(boxes.shape[0]):
        rect = patches.Rectangle((boxes[i,0], boxes[i,1]), boxes[i,2], boxes[i,2], linewidth=1, edgecolor='r', facecolor='none')
        axes[2].add_patch(rect)
    axes[2].set_title('Detail') 

    plt.tight_layout()
    nowstr = datetime.now().strftime('%H-%M-%S_%Y-%m-%d') #nb odd format for easier searching
    savename = savedir + '/' + nowstr + '.png'
    print("Saving to", (savedir + '/' + nowstr + '.png'))
    plt.savefig(savename)
    plt.close('all')

def run_arr(arr, comps):
    bins = 20
    min_x, max_x, min_y, max_y, K, segmented, crop = process(arr, comps, bins)
    appended_boxes = []
    for i in range(K):
        this_region = (segmented==i)*1
        if not ((this_region)[0][0]==1): # this step allows us to skip the background region
            reg_boxes = alg(this_region)
            appended_boxes.append(reg_boxes)
        boxes = np.array([x for xs in appended_boxes for x in xs])
    #convert to original dimensions
    boxes[:,0] = boxes[:,0] + min_x
    boxes[:,1] = boxes[:,1] + min_y
    fac = arr.shape[0]/128
    ret = boxes*fac
    make_plots(arr, boxes, ret, crop, min_x, max_x, min_y, max_y)
    return

def main():
    x, y = np.mgrid[0:128, 0:128]
    #designed to mimic experimental images
    c_arr = (-(x-64)**2 - (y-64)**2)+150
    test = np.where(c_arr>0, c_arr, 0).astype('uint8')
    kernel = np.ones((5,5),np.float32)/25
    test_arr = cv.filter2D(test,-1,kernel).astype('uint8')
    #run code 
    run_arr(test_arr, comps = 1)

if __name__=="__main__": 
    main()
