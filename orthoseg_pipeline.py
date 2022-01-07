
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
import utils.tif_utils as tifutils
import tifffile
# from osgeo import gdal

band_to_indice = {'B':0,'G':1,'R':2,'RE':3,'NIR':4,'thermal':5}

def parse_name(file):
    h,w = file.split('_')
    return(int(h),int(w))




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
    def __init__(self,
                    model = None,
                    output_ortho_file = 'ortho_mask.tif',
                    temp_folder     = 'temp',
                    sub_image_size  = 240, 
                    device          = 'cuda', 
                    thresh          = 0.5,
                    bands = {'R':True,'G':True,'B':True,'NIR':False,'RE':False}
                    ):
        
       
        # Load file 
        self.format = 'tiff'
        self.bands_idx = [band_to_indice[key] for key,value in bands.items() if value == True]
        # Split orthomosaic into sub-images
        self.sub_image_size = sub_image_size # image size of the sub images 
        self.temp_folder    = temp_folder #  path to the temp folder
        self.sub_img_dir    = os.path.join(temp_folder,'sub_img')
        self.sub_mask_dir   = os.path.join(temp_folder,'sub_masks')
        self.sub_img_list   = [] # array with the sub_image names 
        self.path_to_save_ortho_mask = output_ortho_file

        self.device = device

        self.width  = -1
        self.height = -1

        if model == None:
            print("[ERROR] No Segmentation model available")
            exit(-1)

        self.model  = model
        
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
        self.width  = raster.rio.width 
        self.height = raster.rio.height 
        raster = raster.values[self.bands_idx,:,:]
        return(raster)
    
    def preprocessing(self,array):
        '''
        Applying preprocessing to the loaded orthomosaic 

        INPUT: 
            raster containing the orthomosaic

        OUTPUT: 
            numpy array containing the orthomosaic after the preprocessing  
        '''
        
        pass

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

        width  = self.width
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
                sub_img_name = "%05d_%05d"%(h_itr,w_itr)
                sub_img_list.append(sub_img_name)
                # crop sub-image
                sub_array = array[h_itr:h_itr+target_height,w_itr:w_itr+target_width,:]
                # Save image
                sub_img_path = os.path.join(self.sub_img_dir,sub_img_name+'.'+self.format)
                tifffile.imsave(sub_img_path, sub_array)
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
            img_path = os.path.join(self.sub_img_dir,file+'.'+self.format)
            img = np.asarray(rioxarray.open_rasterio(img_path))
            img = np.expand_dims(img,0)
            # img = np.array(mpimg.imread(img_path)).transpose(2,0,1)
            img_torch = torch.from_numpy(img.copy()).type(torch.FloatTensor).to(self.device)
            # Model
            pred_mask = self.model(img_torch).squeeze()
            mask      = logit2label(pred_mask,self.thresh) # Input (torch) output (numpy)
            img_mask  = label2pixel(mask)
            # image 
            #img_mask = np.array(mask*255,dtype=np.uint8)
            mask_img  = Image.fromarray(img_mask)
            mask_path = os.path.join(self.sub_mask_dir,file + '.' + self.format)
            mask_img.save(mask_path)
            # mpimg.imsave(os.path.join(self.sub_mask_dir,file),img_mask)
            bar.update(i)
            # Save mask with the same name to temp folder


    def rebuild_ortho_mask(self,files):
        raster_mask = np.zeros((self.height,self.width),dtype=np.uint8)
        root = self.sub_mask_dir
        #bar = progressbar.ProgressBar(max_value=len(prediction)) 
        for i,(file) in enumerate(files):
            # file name parser
            file_path = os.path.join(root,file+'.tiff')
            pred_mask = np.asarray(Image.open(file_path).convert('L'))
            ph,pw = parse_name(file)

            if len(pred_mask.shape)> 2:
                pred_mask = pred_mask.squeeze()
    
            h,w = pred_mask.shape
            lh,hh,lw,hw = ph,ph+h,pw,pw+w
            raster_mask[lh:hh,lw:hw] = pred_mask
        
        shutil.rmtree(self.sub_mask_dir)
        print("[WAN] Directory deleted: %s"%(self.sub_mask_dir))
        shutil.rmtree(self.sub_img_dir)
        print("[WAN] Directory deleted: %s"%(self.sub_img_dir))
        return(raster_mask)


    def pipeline(self,raster):
        # preprocessing
        print("[INF] Preprocessing")
        raster = self.preprocessing(raster)
        # Splitting
        print("[INF] Ortho splitting")
        sub_img_list = self.ortho_splitting(raster)
        # Segmentation network
        print("[INF] Segmentation")
        self.segmentation(sub_img_list)
        # rebuild orthomask path_to_save_mask_raster,path_to_ortho,path_to_masks
        print("[INF] Rebuilding")
        # path_to_save_mask_ortho = os.path.join(path_to_file,'ortho_mask.tif')
        ortho = self.rebuild_ortho_mask(sub_img_list)

        print("[INF] Finished")
        return(ortho)

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
    path_to_input_ortho = os.path.join(ortho_dir,'ortho.tif')
    # Load raster
    raster = orthoseg().load_ortho(path_to_input_ortho)
    # Get sub-masks' names
    img_dir = 'temp/sub_masks'
    path_to_sub_masks = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    
    # Test rebuild function
    # Inputs: 
    # - path_to_input_ortho: 
    # - path_to_sub_masks:
    
    orthoseg().rebuild(path_to_input_ortho,path_to_sub_masks)

if __name__ == '__main__':

    #TEST_SEGMENTATION()
    TEST_REBUILDING()
    print("MAIN")