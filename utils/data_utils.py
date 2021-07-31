import yaml
import os
# from osgeo import gdal
# import rasterio
import numpy as np
import matplotlib.image as mpimg
import cv2

def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    if min is not None and max is not None:
        norm = (im - min) / (max-min)
    else:
        cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm[norm<0.0] = 0.0
    norm[norm>1.0] = 1.0
    return norm

def standardization(im):
    for i in range(im.shape[2]):
        im[:,:,i] = (im[:,:,i] - np.mean(im[:,:,i])) / np.std(im[:,:,i])
    return(im)

def load_config(config_file = 'data_config.yaml'):
        if not os.path.isfile(config_file):
            raise NameError('File Does not Exist')
        conf_data = yaml.load(open(config_file), Loader=yaml.FullLoader)

        return(conf_data)

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
  

class green_ai():
    def __init__(self, config_file):
        config_file = config_file
        self.config = load_config(config_file)
    
    def get_path(self,dataset,sequence):
        sequence_path = self.config[dataset][sequence]
        dataset_path = os.path.join(self.config['root'], sequence_path)
        return(dataset_path)

    def load_multispectral(self,multispectralPath):
        multispectralFile = self.get_multispectral_path(multispectralPath)
        img = load_ortho_raster(multispectralFile)
        return(img)
    
    def load_hd_rgb(self,hd_rgbPath):
        hd_rgbFile = self.get_hd_rgb_path(hd_rgbPath)
        img = load_ortho_raster(hd_rgbFile)
        return(img)
    
    def load_dsm(self,dsmPath):
        dsmFile = self.get_dsm_path(dsmPath)
        if not os.path.isfile(dsmFile):
            raise NameError('File does not exist')
        dsm_img = load_ortho_raster(dsmFile)
        # dsm_img =  rasterio.open(dsmFile,driver="GTiff")
        return(dsm_img)

    def load_mask(self,path):
        File = self.get_mask_path(path)
        if not os.path.isfile(File):
            raise NameError('File does not exist')
        
        img = mpimg.imread(File)
        # raster = load_ortho_raster(File)
        return(img)
        # dsm_img =  rasterio.open(dsmFile,driver="GTiff")
    
    def get_dsm_path(self,path):
        return(os.path.join(path,'orthomosaic','DSMX7.tif'))

    def get_hd_rgb_path(self,path):
        return(os.path.join(path,'orthomosaic','OrtoX7.tif'))
    
    def get_multispectral_path(self,path):
        return(os.path.join(path,'orthomosaic','OrtoAltum.tif'))
    def get_mask_path(self,path):
        return(os.path.join(path,'orthomosaic','mask.png'))