
'''
Author: Tiago Barros
Created: 06/09/2021

Function: 
Implementation of the othoseg pipline. The orthoseg pipline performs semantic segmentation on orthomosaics,
outputting a masks with the same size. 


'''

import os
import rioxarray
import numpy as np

class orthoseg():
    def __init__(self,temp_folder = 'temp',sub_image_size=240):
        # Load file 
        # Split orthomosaic into sub-images
        self.sub_image_size = sub_image_size # image size of the sub images 
        self.temp_folder    = temp_folder #  path to the temp folder
        # segmentation
    
    def load_ortho(self, path_to_file):
        '''
        Load orthomosaic to memory. 
        INPUT: 
            path_to_file: absolute path to file with tif 
        OUTPUT:
            numpy array representing the orthomosaic

        ''' 

        if not os.path.isfile(path_to_file): 
            print("[ERROR] File does not exist: " + path_to_file)
            return(-1)
        
        raster = rioxarray.open_rasterio(path_to_file)
        array = np.array(raster.values)

        return(raster)
    
    def preprocessing(self,raster):
        '''
        Applying preprocessing to the loaded orthomosaic 

        INPUT: 
            numpy array containing the orthomosaic

        OUTPUT: 
            numpy array containing the orthomosaic after the preprocessing  
        '''
        
        return(raster)
    
    def ortho_splitting(self,raster):
        '''
        Splitting the orthomosaic into sub_images which are saved in a temp folder
        '''




        
        

