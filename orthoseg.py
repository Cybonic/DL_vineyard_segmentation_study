
'''
Author: Tiago Barros
Created: 06/09/2021

Function: 
Implementation of the othoseg pipline. The orthoseg pipline performs semantic segmentation on orthomosaics,
outputting a masks with the same size. 


'''

import os
from urllib import parse
import rioxarray
import numpy as np
import progressbar
import matplotlib.image as mpimg
import torch 
from networks import unet_bn as unet
from networks import segnet
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import utils.tif_utils as tif
from osgeo import gdal

def _rebuild_ortho_mask(file_masks,sub_mask_dir):
    raster_array = np.array([])
    raster_line  = np.array([])
    line = 0
    bar = progressbar.ProgressBar(max_value=len(file_masks)) 
    for i,file in enumerate(file_masks):
        # file name parser
        parsed_file = file.split('.')[0].split('_')
        init_height = int(parsed_file[0])
        init_width  = int(parsed_file[1])

        mask_path = os.path.join(sub_mask_dir,file)
        mask = np.asarray(Image.open(mask_path))

        if(init_height>line):
            if raster_array.size == 0:
                raster_array = raster_line.copy() 
            else:
                raster_array = np.vstack((raster_array,raster_line))
            raster_line = np.array([])
            line = init_height
        
        if line == init_height:
            if raster_line.size == 0:
                raster_line = mask
            else: 
                raster_line = np.hstack((raster_line,mask))
        bar.update(i)
    return(raster_array)



def img2label(img,labels={255:1}):

   mask =  np.zeros(img.shape[0:2])
   img = img[:,:,0]
   for pixel_label,label in labels.items():

        mask[img==pixel_label] = label


def label2pixel(array):
    img_mask = np.array(array*255,dtype=np.uint8)
    img = np.stack((img_mask,img_mask,img_mask)).transpose(1,2,0)
    return(img)


def logit2label(array,thres):
    '''
    Convetion from a cxmxn logits torch array - where c represents the number of labels - to a 1xmxn label torch array 
    Parameters 
      - array: c x m x n torch array of logits
    Returns
      - 1 x m x n  array of labels (int) 
    '''
    new_array = np.zeros(array.shape,dtype=np.int32)
    norm_array = torch.nn.Sigmoid()(array).cpu().detach().numpy()
    bin_class = norm_array >= thres
    new_array[bin_class] = 1
    return(new_array)

class orthoseg():
    def __init__(self,  temp_folder     = 'temp',
                        sub_image_size  = 240, 
                        device          = 'cuda', 
                        thresh          = 0.5,
                        file_format     = 'BMP'
                        ):
        
       
        # Load file 
        self.format = file_format
        # Split orthomosaic into sub-images
        self.sub_image_size = sub_image_size # image size of the sub images 
        self.temp_folder    = temp_folder #  path to the temp folder
        self.sub_img_dir    = os.path.join(temp_folder,'sub_img')
        self.sub_mask_dir   = os.path.join(temp_folder,'sub_masks')
        self.sub_img_list   = [] # array with the sub_image names 

        # segmentation model
        self.device = device 
        self.model  = segnet.SegNet(num_classes=1,  n_init_features=3) # UNet has no dropout
        
        self.thresh =  thresh # segmentation Treshold 
        # Device configuration
        # device = 'cpu'
        if torch.cuda.is_available() and device == 'cuda': 
            device = 'cuda:0'
            torch.cuda.empty_cache()

        self.model.to(device)

    
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
        #array = np.array(raster.values)

        self.width = raster.rio.width 
        self.height = raster.rio.height 

        return(raster)
    
    def preprocessing(self,raster):
        '''
        Applying preprocessing to the loaded orthomosaic 

        INPUT: 
            raster containing the orthomosaic

        OUTPUT: 
            numpy array containing the orthomosaic after the preprocessing  
        '''
        
        array = np.array(raster.values)

        return(array)
    
    def ortho_splitting(self,array):
        '''
        Splitting the orthomosaic into sub_images which are saved in a temp folder

        INPUT: 
            numpy array containing orthomosaic
        OUTPUT:
            list with all subimage file names

        '''

                # delete tmp file if it exist
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
            print("[WAN] Directory deleted: %s"%(self.temp_folder))

        #if not os.path.isdir(self.sub_img_dir):
            # create a new directory to save sub_images
        os.makedirs(self.sub_img_dir)
        print("[WRN] New directory created: " + self.sub_img_dir)

        #if not os.path.isdir(self.sub_mask_dir):
        os.makedirs(self.sub_mask_dir)
        print("[WRN] New directory created: " + self.sub_mask_dir)

        sub_img_list = [] 

        target_height = self.sub_image_size
        target_width  = self.sub_image_size

        width = self.width
        height = self.height
        
        array = array.transpose(1,2,0)

        max_itr = height
        bar = progressbar.ProgressBar(max_value=max_itr)  
        h_itr= 0
        w_itr= 0
        while(h_itr < height):
            # reset width counter 
            bar.update(h_itr) 
            w_itr = 0
            while(w_itr < width):
                # Sub-image name + absolute path 
                #sub_img_path = os.path.join(self.sub_img,"%05d_%05d.png"%(h_itr,w_itr))
                sub_img_name = "%05d_%05d.%s"%(h_itr,w_itr,self.format)
                sub_img_list.append(sub_img_name)
                # crop sub-image
                sub_array = array[h_itr:h_itr+target_height,w_itr:w_itr+target_width,:]
                # Save image
                sub_img = Image.fromarray(sub_array)
                sub_img_path = os.path.join(self.sub_img_dir,sub_img_name)
                sub_img.save(sub_img_path, format=self.format)
                #mpimg.imsave(, sub_array)
                # Next width iteration
                w_itr = w_itr + target_width
            # Next height iteration
            h_itr = h_itr + target_height

        return(sub_img_list)
    
    def segmentation(self,sub_img_list):
        '''
        Semenatic segmentation of sub-images

        INPUT: 
            [list] of image files

        '''

        bar = progressbar.ProgressBar(max_value=len(sub_img_list))  

        for i,file in enumerate(sub_img_list):
            img_path = os.path.join(self.sub_img_dir,file)
            img = np.asarray(Image.open(img_path)).transpose(2,0,1)
            img = np.expand_dims(img,0)
            # img = np.array(mpimg.imread(img_path)).transpose(2,0,1)
            img_torch = torch.from_numpy(img.copy()).type(torch.FloatTensor).to(self.device)
            # Model
            pred_mask = self.model(img_torch).squeeze()
            mask = logit2label(pred_mask,self.thresh) # Input (torch) output (numpy)
            img_mask = label2pixel(mask)
            # image 
            #img_mask = np.array(mask*255,dtype=np.uint8)
            mask_img  = Image.fromarray(img_mask)
            mask_path = os.path.join(self.sub_mask_dir,file)
            mask_img.save(mask_path, format=self.format)
            # mpimg.imsave(os.path.join(self.sub_mask_dir,file),img_mask)
            bar.update(i)
            # Save mask with the same name to temp folder


    def rebuild(self,mask_raster_file,raster_file,file_masks):
        '''
        Rebuild ortho mask from the submasks 

        INPUT: 
            [list] of image files
        
        OUTPUT:
            ortho.tif

        '''
        print("[INF] ortho mask: " + raster_file)
        rebuilded_array = _rebuild_ortho_mask(file_masks,self.sub_mask_dir)
        
        raster = gdal.Open(raster_file, gdal.GA_ReadOnly)
        mask_raster = tif.array2raster(mask_raster_file,raster,rebuilded_array)
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
            print("[WAN] Directory deleted: %s"%(self.temp_folder))
        return(mask_raster)



    def pipeline(self,path_to_file):
        # loading orthomosaic 
        print("[INF] Loading ortho")
        orig_raster = self.load_ortho(path_to_file)
        # preprocessing
        print("[INF] Preprocessing")
        raster = self.preprocessing(orig_raster)
        # Splitting
        print("[INF] Ortho splitting")
        sub_img_list = self.ortho_splitting(raster)
        # Segmentation network
        print("[INF] Segmentation")
        self.segmentation(sub_img_list)
        # rebuild orthomask path_to_save_mask_raster,path_to_ortho,path_to_masks
        print("[INF] Rebuilding")
        # path_to_save_mask_ortho = os.path.join(path_to_file,'ortho_mask.tif')
        path_to_save_mask_ortho = 'ortho_mask.tif'
        ortho = self.rebuild(path_to_save_mask_ortho,path_to_file,sub_img_list)
        print("[INF] Finished")
        return()



# ========================================================================================

def TEST_LOAD_IMG():
    
    '''
    PNG images usually have four channels. 
    Three color channels for red, green and blue, 
    and the fourth channel is for transparency, 
    also called alpha channel.

    '''
    plt.ion()

    plt.show()
    
    img_dir = 'temp/sub_img'

    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    print(len(files))
    bar = progressbar.ProgressBar(max_value=len(files))  
    for i,file in enumerate(files):
        file_path = os.path.join(img_dir,file)
        if not os.path.isfile(file_path):
            print("File does not exist")
        img = mpimg.imread(file_path)
        
        img = img
        plt.imshow(img)
        plt.draw()
        plt.pause(0.001)
        bar.update(i)
        img_np = np.asarray(img).transpose(2,0,1)




def TEST_SEGMENTATION():
    img_dir = 'temp/sub_img'
    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    print(len(files))
    orthoseg().segmentation(files)
        


def TEST_REBUILDING():
    import platform
    pc_name = platform.node() 
    print("PC Name: " + pc_name)
    if pc_name == 'DESKTOP-SSEDT6V':
        root = "E:\\dataset"
    else:
        root = "/home/tiago/BIG/dataset"

    ortho_dir = os.path.join(root,"greenAI/drone/quintabaixo/04_05_2021/60m/x7")
    # Path to tif
    path_to_ortho = os.path.join(ortho_dir,'ortho.tif')
    # Load raster
    raster = orthoseg().load_ortho(path_to_ortho)
    # Get sub-masks' names
    img_dir = 'temp/sub_masks'
    path_to_masks = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    #path_to_save_mask_raster = os.path.join(ortho_dir,'mask_ortho.tif')
    path_to_save_mask_raster = 'mask_ortho.tif'
    orthoseg().rebuild(path_to_save_mask_raster,path_to_ortho,path_to_masks)


if __name__ == '__main__':

    #TEST_SEGMENTATION()
    TEST_REBUILDING()
    print("MAIN")