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

def get_files(dir):
    '''
    return files in a directory
    @param dir (string) target direcotry
    @retrun (list): list of files  
    '''
    if not os.path.isdir(dir):
        return(list([]))

    files = os.listdir(dir)
    #if not end:
    new_files = [f.split('.')[0] for f in files]
    # Runs only when DEBUG FLAG == TRUE
    t = files[0].split('.')[1]
    return({'root':dir,'files':new_files,'file_type':t})


