'''

https://gis.stackexchange.com/questions/57005/python-gdal-write-new-raster-using-projection-from-old


'''

import yaml
import os
import rasterio
import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from tqdm import tqdm 


NO_DATA = 9999

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


def load_tif(File):
    if not os.path.isfile(File):
        raise NameError('File does not exist')
    img = gdal.Open(File)
    return(img)


def get_rgb_bans(multispectral):
    
    # since there are 3 bands
    # we store in 3 different variables
    r = multispectral.GetRasterBand(1).ReadAsArray() # Red channel
    g = multispectral.GetRasterBand(2).ReadAsArray() # Green channel
    b = multispectral.GetRasterBand(3).ReadAsArray() # Blue channel

    img = np.dstack((r, g, b))

    return(img)



def align_rasters(match_ds,src):
    # Other approach https://stackoverflow.com/questions/54959995/match-raster-cell-size-to-another-raster/55123011

    outFile = "Temp\out.tif"

    # Source
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    src_bands = src.RasterCount
    # We want a section of source that matches this:
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst = gdal.GetDriverByName('Gtiff').Create(outFile, wide, high, src_bands, gdal.gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdal.gdalconst.GRA_NearestNeighbour)
    # os.remove(outFile)

    return(dst)

def tif2array(input_file):
        """
        Read GeoTiff and convert to numpy.ndarray

        :param input_file: absolute path to input GeoTiff file
        :return : (np.array) image for each bands

        Source:
            - https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8
        """
        # logging.info(">>>> Converting geographic format to array...")

        dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
        datatype = dataset.GetRasterBand(1).DataType
        ndv = dataset.GetRasterBand(1).GetNoDataValue()

        image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount), dtype=float)

        for b in range(dataset.RasterCount):
            band = dataset.GetRasterBand(b + 1)
            image[:, :, b] = band.ReadAsArray()

        image[image == ndv] = NO_DATA

        return image, datatype, dataset

def array2raster(new_rasterf_fn, target_geo_data_file, array, dtype = gdal.gdalconst.GDT_Float32):
    """
    Save GTiff file from numpy.array

    :param new_rasterf_fn: the output image filename
    :param dataset: the original tif file, with spatial metadata
    :param array: image in numpy.array
    :param dtype: Byte or Float32.

    Source:
        - https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8
        - https://gdal.org/development/rfc/rfc58_removing_dataset_nodata_value.html
    """
    #logging.info(">>>> Converting array to geographic format...")
    target_geo_raster = gdal.Open(target_geo_data_file, gdal.GA_ReadOnly)

    cols = array.shape[1]
    rows = array.shape[0]
    origin_x, pixel_width, b, origin_y, d, pixel_height = target_geo_raster.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')

    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

   
    out_raster = driver.Create(new_rasterf_fn, cols, rows, band_num, dtype)
    out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))

    

    for b in range(band_num):
        outband = out_raster.GetRasterBand(b + 1)
        outband.SetNoDataValue(NO_DATA)

        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:, :, b])
        
    
    prj = target_geo_raster.GetProjection()
    out_raster_srs = osr.SpatialReference(wkt=prj)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())
    outband.FlushCache()
    return(out_raster)