
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
import progressbar
import matplotlib.image as mpimg
import torch 
from networks import unet_bn as unet
from networks import segnet
import shutil
import imageio
import matplotlib.pyplot as plt

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
    def __init__(self,temp_folder = 'temp',sub_image_size=240, device = 'cuda', thresh = 0.5):
        
       
        # Load file 
        
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
        array = np.array(raster.values)

        self.width = raster.rio.width 
        self.height = raster.rio.height 

        return(array)
    
    def preprocessing(self,raster):
        '''
        Applying preprocessing to the loaded orthomosaic 

        INPUT: 
            numpy array containing the orthomosaic

        OUTPUT: 
            numpy array containing the orthomosaic after the preprocessing  
        '''
        
        return(raster)
    
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
                sub_img_name = "%05d_%05d.jpg"%(h_itr,w_itr)
                sub_img_list.append(sub_img_name)
                # crop sub-image
                sub_array = array[h_itr:h_itr+target_height,w_itr:w_itr+target_width,:]
                # Save image
                mpimg.imsave(os.path.join(self.sub_img_dir,sub_img_name), sub_array)
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
            img = np.array(mpimg.imread(img_path)).transpose(2,0,1)
            img = np.expand_dims(img,0)
            # img = np.array(mpimg.imread(img_path)).transpose(2,0,1)
            img_torch = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
            # Model
            pred_mask = self.model(img_torch)
            mask = logit2label(pred_mask,self.thresh) # Input (torch) output (numpy)
            # image 
            mpimg.imsave(os.path.join(self.sub_mask_dir,file),mask)
            bar.update(i)
            # Save mask with the same name to temp folder


    def rebuild(self,file_masks):
        '''
        Rebuild ortho mask from the submasks 

        INPUT: 
            [list] of image files
        
        OUTPUT:
            ortho.tif

        '''

        print("ssss")

    def pipeline(self,path_to_file):
        # loading orthomosaic 
        raster = self.load_ortho(path_to_file)
        # preprocessing
        raster = self.preprocessing(raster)
        # Splitting
        sub_img_list = self.ortho_splitting(raster)
        # Segmentation network
        self.segmentation(sub_img_list)
        # rebuild orthomask
        ortho = self.rebuild(sub_img_list)


        return()



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

    model  = segnet.SegNet(num_classes=1,  n_init_features=3).to('cuda:0') # UNet has no dropout
    img_dir = 'temp/sub_img'
    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    print(len(files))
    bar = progressbar.ProgressBar(max_value=len(files))  
    
    for i,file in enumerate(files):
        file_path = os.path.join(img_dir,file)
        if not os.path.isfile(file_path):
            print("File does not exist")
        
        img = mpimg.imread(file_path)
        img = np.asarray(img).transpose(2,0,1)
        img_torch = torch.from_numpy(img).type(torch.FloatTensor).to('cuda:0')
        pred_mask = model(img_torch)
        mask = logit2label(pred_mask,0.5) # Input (torch) output (numpy)
        mask = mask.cpu().detach().numpy()
        



if __name__ == '__main__':

    TEST_SEGMENTATION()
    print("MAIN")