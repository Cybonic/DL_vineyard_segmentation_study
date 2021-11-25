
# https://geoscripting-wur.github.io/PythonRaster/
# https://rasterio.readthedocs.io/en/latest/topics/overviews.html
# https://gis.stackexchange.com/questions/5701/differences-between-dem-dsm-and-dtm/5704#5704
# https://geoscripting-wur.github.io/PythonRaster/

# https://carpentries-incubator.github.io/geospatial-python/aio/index.html

# =====================================================================
# Date: 02/09/2021
# Author: Tiago B.
#
# Split a raster into subimages
# 

import os
import math
import numpy as np
import matplotlib.image as saver
import progressbar
import rioxarray
#import utils.utils as utils
from tqdm import tqdm
from PIL import Image
import numpy as np
import tifffile
import argparse



def save_sub_tiff(input,target_height,target_width,dest_path):
    '''
    Split "input" into subarrays with the size "target_height" and "target_width". 
    Save these subarrays to "dest_path"

    @param: input (Numpy) Array with dim [height,width,bands] 
    @param: target_height (int) height of the subarray
    @param: target_width (int) width of the subarray
    @return: number of images generated
    '''
   
    bands = input.shape[2]
    height= input.shape[0]
    width = input.shape[1]

    if bands>height or bands>width:
        print("[ERROR] band dim > than height or > than width")
        return(0)

    print("H:%d W:%d B:%d"%(height,width,bands))
    
    h_array = list(range(0,height,target_height))
    w_array = list(range(0,width,target_width))

    h_array_len = len(h_array)
    w_array_len = len(w_array)

    for i in tqdm(range(0,h_array_len-1)):
    # reset width counter 
        for j in range(0,w_array_len-1):
            # Sub-image name + absolute path 
            img_path = os.path.join(dest_path,"%05d_%05d.tiff"%(h_array[i],w_array[j]))
            # crop sub-image
            sub_array = input[h_array[i]:h_array[i+1],w_array[j]:w_array[j+1],:]
            # Save image
            tifffile.imsave(img_path, sub_array)

    n_images_saved = h_array_len*w_array_len
    return(n_images_saved)


def tiff2numpy(tiff):
    return(np.array(tiff.values))

def show_raster_info(raster):
    print("Geotiff information: ")
    print("CRS      {}".format(raster.rio.crs))
    print("NODATA   {}".format(raster.rio.nodata))
    print("Bounds   {}".format(raster.rio.bounds()))
    print("Width    {}".format(raster.rio.width))
    print("Height   {}".format(raster.rio.height))


def main_save_sub_tiff(source_file,dest_dir,t_hight,t_width):

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    # open file
    raster = rioxarray.open_rasterio(source_file)
    # Plot additional information regarding geotiff
    show_raster_info(raster)
    # Convert to Numpy 
    array = tiff2numpy(raster)
    array = np.transpose(array, (1, 2, 0))

    save_sub_tiff(array,t_hight,t_width,dest_dir)


def TEST_SAVE_SUB_IM():

    width = 1000
    hight = 1000
    bands = 5

    sub_width = 100
    sub_hight = 100

    array = np.random.randint(5, size=(bands,hight,width),dtype=np.uint8)

    array = np.transpose(array, (1, 2, 0))

    dest_path = "TESTS"
    
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    save_sub_tiff(array,sub_hight,sub_width,dest_path)


if __name__=='__main__':
    
    # TEST_SAVE_SUB_IM()
    parser = argparse.ArgumentParser(description='Split and save sub tiff images')
    parser.add_argument('--source_file',
                        default = '/home/tiago/greenai/dataset/QtaBaixo27Jul/x7/OrthoRGBQtaBaixoJul27.tif',
                        help='')
    parser.add_argument('--dest_dir',
                        default = "/home/tiago/greenai/dataset/QtaBaixo27Jul/x7/sub_imgs",
                        help='')
    parser.add_argument('--height',
                        default = 240,
                        help='target height of sub images')
    parser.add_argument('--width',
                        default = 240,
                        help='target width of sub images')              
    args = parser.parse_args()

    source_file = args.source_file
    dest_dir    = args.dest_dir
    t_height    = args.height
    t_width     = args.width

    print("="*50)
    print("Source file: %s"%(source_file))
    print("Dest. directory: %s"%(dest_dir))
    print("Target height: %s"%(t_height))
    print("Target width: %s"%(t_width))
    print("="*50)


    main_save_sub_tiff(source_file,dest_dir,t_height,t_width)
    

    










