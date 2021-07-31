import yaml
import os
# from osgeo import gdal
# import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def color_mapper(img,value):
    ms_rgb_img = np.array((np.abs(img)**(value)),dtype=np.float64)
    # ms_rgb_img = np.array((img**(1/4))*255,dtype=np.int32) 
    return(ms_rgb_img)
    
def load_subimg(file):
    array = np.load(file + '.npy')
    return(array)

def load_config(config_file = 'data_config.yaml'):
    if not os.path.isfile(config_file):
        raise NameError('File Does not Exist')
    conf_data = yaml.load(open(config_file), Loader=yaml.FullLoader)

    return(conf_data)

def save_config(session,param):
    with open(session, 'w') as file:
        documents = yaml.dump(param, file)

def load_ortho_raster(File):
    if not os.path.isfile(File):
        raise NameError('File does not exist')
    img = gdal.Open(File)
    return(img)

def get_rgb_bans(multispectral):
    # Get and convert  RGB bands to numpy tensor gdal [nxmx6] -> numpy [nxmx3]
    # 
    # get_rgb_bans(multispectral):
    # Input: 
    #  multispectral raster image (gdal) 
    # Output:
    #  
    # since there are 3 bands
    # we store in 3 different variables
    r = multispectral.GetRasterBand(1).ReadAsArray() # Red channel
    g = multispectral.GetRasterBand(2).ReadAsArray() # Green channel
    b = multispectral.GetRasterBand(3).ReadAsArray() # Blue channel

    img = np.dstack((r, g, b))

    return(img)

def align_rasters(match_ds,src):
    # Other approach https://stackoverflow.com/questions/54959995/match-raster-cell-size-to-another-raster/55123011

    if not os.path.isdir("Temp"):
        os.mkdir("Temp")
        print("[INF] \"Temp\" folder created!")
        
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
    # os.remove('Temp')
    return(dst)


