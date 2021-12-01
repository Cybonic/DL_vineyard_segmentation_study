import os
import sys 
import pathlib
from scipy import ndimage, misc
from torch._C import ErrorReport

dir_path = os.path.dirname(pathlib.Path(__file__).resolve().parent)
# dir_path = os.path.dirname(os.path.realpath(__file__).parent)

sys.path.insert(1, dir_path)

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import utils.utils as utils 
import utils.agro_utils as agro 
from torchvision import transforms as trans  
import torch 
from torchvision.transforms import functional as F
import random
import albumentations as A
from  utils import data_utils
from utils.vis_utils import vis
from utils.data_utils import normalize
import torch.utils.data as torch_utils
import rioxarray
from PIL import Image

import cv2 

MAX_ANGLE = 180
band_to_indice = {'R':0,'G':1,'B':2,'RE':3,'NIR':4,'thermal':5}
dataset_label_to_indice = {'esac1':'esac1','esac2':'esac2','valdo':'valdoeiro'}
DATASET_NAMES = ['valdoeiro','esac','qtabaixo']

def fetch_files(folder):
    dir = os.path.join(folder)

    if not os.path.isdir(dir):
        raise NameError(dir)

    return([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

def comp_agro_indices(bands, indices_to_compute):

    indices_to_use  = indices_to_compute
    
    if indices_to_use['NDVI'] == True:
        red  = bands[:,:,band_to_indice['RE']]
        nir  = bands[:,:,band_to_indice['NIR']]
        ndvi = np.expand_dims(data_utils.normalize(agro.NDVI(nir,red)),axis=2)

        indices = ndvi 
    return(ndvi)

def preprocessing(img,values):
   
    img = img.transpose(2,0,1)
    nrom_bands = []
    for i,C in enumerate(img):
        C = data_utils.normalize(C)

        nrom_bands.append(C)
    nrom_bands = tuple(nrom_bands)
    nrom_bands = np.stack(nrom_bands)
    nrom_bands = nrom_bands.transpose(1,2,0)    
 
    return(nrom_bands)


def split_data_plots(file_array):
    plots = []
    for file_name in file_array:
        plots.append(int(file_name.split('\\')[-1].split('.')[0].split('_')[-1][0]))
    
    u_plots = np.unique(plots)
    data_p_plots = []
    for un in u_plots:
        indx = np.where(plots==un)[0]
        data_p_plots.append(np.array(file_array)[indx])
    return(np.array(data_p_plots))


def tiff2numpy(tiff):
    return(np.array(tiff.values))


def load_file(file):

    if not os.path.isfile(file):
        return(ErrorReport)
    
    file_type = file.split('.')[-1]

    if file_type=='tiff':
        #raster = rioxarray.open_rasterio(file)
        array = np.array(Image.open(file)).astype(np.uint8)
        # array = tiff2numpy(raster)

    elif(file_type=='png'):
        array = np.array(Image.open(file)).astype(np.uint8)
    else:
        array = np.load(file)

    # Get the dim order right: C,H,W
    if array.shape[-1]<array.shape[0]:
        array = array.transpose(2,0,1)

    name = file.split(os.sep)[-1].split('.')[0]
    return(array,name)


class augmentation():
    split_idx = 0
    
    def __init__(self,max_angle = MAX_ANGLE):
        self.max_angle =max_angle  

    def __call__(self, bands, mask, ago_indices):
        
        n_band = bands.shape[2]
        rot_value       = random.randint(0,self.max_angle)
        rotated_bands   = ndimage.rotate(bands, rot_value, reshape=False)
        rotated_mask    = ndimage.rotate(mask, rot_value, reshape=False)

        # if not isinstance(ago_indices,(np.ndarray, np.generic)):
        if  ago_indices is not None and ago_indices.size >0:
            rotated_ago_indices = ndimage.rotate(ago_indices, rot_value, reshape=False)
        else: 
            rotated_ago_indices = ago_indices

        return(rotated_bands,rotated_mask,rotated_ago_indices) 
        # Now we will create a pipe of transformations
        


class greenAIDataStruct():
    def __init__(self,root,vineyard_plot,sensor):
        self.root = root
        self.plot = vineyard_plot # list of vineyard plots
        self.sensor = sensor # sensor name
        # build absolut path 
        self.paths = [ os.path.join(root,p,sensor) for p in vineyard_plot]

    def fetch_files(self,folder):
        image_array = np.array([])
        global_img_array = np.array([])
        
        for path in self.paths:
            path = os.path.join(path,folder)
            if not os.path.isdir(path):
                raise NameError("Data folders do not Exist!: " + path)

            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            global_files = [os.path.join(path,f) for f in files]  
            image_array = np.append(image_array,files)
            global_img_array = np.append(global_img_array,global_files)

        return(image_array,global_img_array)

    def fetch_imgs_files(self):
        return( self.fetch_files('images'))
    
    def fetch_mask_files(self):
        return(self.fetch_files('masks'))
    
    def get_data(self,path_type = 'global',fraction = None):

        imgs,full_img_path = self.fetch_files('images')
        masks,full_mask_path = self.fetch_files('masks')

        if path_type == 'global':
            imgs = full_img_path
            masks = full_mask_path
        
        if fraction != None and fraction > 0 and fraction <1:
            n_samples = len(imgs)
            n_select_samples = int(round(fraction*n_samples,0))
            # Generate linear indices 
            setp =int(n_samples/n_select_samples)
            select_samples_idx = np.array(range(0, n_samples,setp))
            imgs = imgs[select_samples_idx]
            masks = masks[select_samples_idx]
            
        return({'imgs':imgs,'masks':masks})


class dataset_wrapper(greenAIDataStruct):
    def __init__(self, root,
                        vineyard_plot,sensor, 
                        bands = {'R':True,'G':True,'B':True}, 
                        agro_index = {'NDVI':False}, 
                        transform = None, 
                        path_type='global',
                        fraction = None):
        super(dataset_wrapper, self).__init__(root,vineyard_plot,sensor)
        
        self.data   = self.get_data(fraction=fraction)
        self.imgs = np.array(self.data['imgs'])
        self.masks = np.array(self.data['masks'])
        self.bands_to_use = bands
        self.agro_index = agro_index
        self.transform = transform
        self.sensor  =sensor

        self.input_channels = {'bands':bands,'indices':agro_index}
        self.color_value =  0

    def __getitem__(self,itr):

        img_file = self.imgs[itr]
        #print(file)
        img,name = self.load_im(img_file)
       
        img = preprocessing(img, self.color_value)
        
        if  self.sensor != 'x7' and any(self.agro_index.values())==True:
            # HD has no NIR and RE bands to compute NDVI
            agro_indice = comp_agro_indices(img,self.agro_index) 
            agro_indice = agro_indice.transpose(2,0,1)
            #agro_indice = torch.from_numpy(agro_indice).type(torch.FloatTensor)
        else:  
            agro_indice = np.array([])

        mask_file = self.masks[itr]
        mask,name = self.load_bin_mask(mask_file)
        

        if self.transform:
            img,mask,agro_indice = self.transform(img,mask,agro_indice)
            
        #mask  = mask.transpose(2,0,1)
        #input_bands = input_bands.transpose(2,0,1)

        input_bands = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        agro_indice = torch.from_numpy(agro_indice).type(torch.FloatTensor)
        
        path_name = self.paths[0] 
    
        batch = {'bands':input_bands,'mask':mask,'indices':agro_indice,'name':name,'path':path_name}
        # Convert to tensor
        return(batch)
    
    def __len__(self):
        return(len(self.imgs))
    
    def load_im(self,file):
        #print("Image: " + file)
        array,name = load_file(file)
        bands_idx = [band_to_indice[key] for key,value in self.bands_to_use.items() if value == True]
        array =  array[bands_idx,:,:]
        return(array,name)

    def load_bin_mask(self,file):
        #print("Mask: " + file)
        array,name = load_file(file)
        if len(array.shape)>2:
            array = array[0,:,:]
        mask = np.expand_dims(array,axis=0)/255
        
        mask[mask>0.5]  = 1 
        mask[mask<=0.5] = 0

        return(mask,name)

class dataset_loader():
    def __init__(self,
                    root, 
                    sensor,  
                    bands , 
                    agro_index ,
                    augment = False,
                    trainset=['esac1','esac2'],
                    testset = ['Valdoeiro'], 
                    batch_size = 1, 
                    shuffle = True, 
                    workers = 1, 
                    debug = True,
                    fraction = {'train':None, 'test':None}):

        self.sensor = sensor
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.workers = workers
        self.debug = debug
        self.bands = bands

        self.test_loader = None 
        self.train_loader = None

        if debug == True:
            print("---"*10)
            print("[INF] DATASET_LOADER")
            print("[INF] Sensor:",sensor)
            print("[INF] Test Plot:",' '.join(testset))
            print("[INF] Train Plot:",' '.join(trainset)) 

    
        if not self.sensor in ['altum','x7']:
            raise NameError 
        

        # Test set conditions

        test_cond = [True for name in testset if name in DATASET_NAMES]

        if test_cond:
        
            # test loader
            self.test  = dataset_wrapper(root,testset, sensor,bands, agro_index,fraction = fraction['train'])

            self.test_loader = DataLoader(  self.test,
                                    batch_size = 1,
                                    shuffle = False,
                                    num_workers = self.workers,
                                    pin_memory=False)


        train_cond = [True for name in trainset if name in DATASET_NAMES]
        
        if train_cond:
            self.train  = dataset_wrapper(root,testset, sensor,bands, agro_index,fraction = fraction['train'])
            # Train loader
            self.train_loader = DataLoader(  self.train,
                                        batch_size = self.batch_size,
                                        shuffle = self.shuffle,
                                        num_workers = self.workers,
                                        pin_memory=False)
        
        aug = None
        if augment == True:
            aug = augmentation()


        if debug == True:
            if not self.train_loader == None:
                print("[INF] Train: %d"%(len(self.train_loader)))
            else:
                print("[INF] Train:" + str(self.train_loader))

            print("[INF] Test: %d"%(len(self.test_loader)))
            print("[INF] Batch Size: %d"%(self.batch_size))
            print("[INF] Shuffle: %d"%(self.shuffle))
            print("[INF] Workers: %d"%(self.workers))

            print("---"*10)


    def get_train_loader(self):
        return(self.train_loader)
    
    def get_test_loader(self):
        return(self.test_loader)

    
# ==================================================================================================================
# TESTS 

def TEST_FRACTION(root,fraction = None):
    multispectral_test = dataset_wrapper(root,
                                            ['esac2'],
                                            'RGBX7',
                                            bands = {'R':True,'G':True,'B':True},
                                            agro_index = {'NDVI':True},
                                            fraction = fraction)
    
    
    from utils.scrip_utils import __FILE__,__FUNC__
    print("[INF] " + __FUNC__() + "DATASET fraction {} samples {}".format(fraction,len(multispectral_test)))


def TEST_PLOT_DATA(root,fraction = None, pause = 1):

    aug = augmentation()
    multispectral_test = dataset_wrapper(root,
                                        ['esac1'],
                                        'Multispectral',
                                        bands = {'NIR':True},
                                        agro_index = {'NDVI':True}, 
                                        transform = aug,
                                        fraction = fraction)


    fig, ax1 = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    for i in range(len(multispectral_test)):
        batch = multispectral_test[i]
        img  = batch['bands']
        mask = batch['mask']
        ndvi = batch['indices']

        im = img.cpu().numpy().squeeze()
        msk = mask.cpu().numpy().squeeze()
        ndvi = ndvi.numpy().squeeze()

        msk =  np.stack((msk,msk,msk)).transpose(1,2,0)
        ndvi = np.stack((ndvi,ndvi,ndvi)).transpose(1,2,0)

        if len(im.shape)<=2:
            im = np.stack((im,im,im)).transpose(1,2,0)
            vis_img = np.hstack((im,msk,ndvi))
        elif im.shape[1]==3:
            vis_img = np.hstack((im,msk,ndvi))
        elif im.shape[1]>3:
            im1 = im[:,:,0:3]
            im2 = im[:,:,3]
            img =  np.stack((im2,im2,im2)).transpose(1,2,0)
            vis_img = np.hstack((im1,img,msk,ndvi))
        
        vis_img = (vis_img * 255).astype(np.uint8)
        # cv2.imshow('', vis_img)
        ax1.imshow(vis_img)
        
        plt.draw()
        plt.pause(pause)


def TEST_NDVI(root,fraction=0.5):

    #aug = augmentation()
    multispectral_test = dataset_wrapper(root,
                                        ['esac2'],
                                        'Multispectral',
                                        bands = {'R':True,'G':True,'B':True},
                                        agro_index = {'NDVI':True}, 
                                        #transform = aug,
                                        fraction = fraction)


    fig, ax1 = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    ndvi_array = {'global':[],'pred':[],'gt':[]}

    fig, ax1 = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    for i in range(len(multispectral_test)):
        batch = multispectral_test[i]
        img  = batch['bands']
        mask = batch['mask']
        ndvi = batch['indices']

        im = img.cpu().numpy().squeeze()
        msk = mask.cpu().numpy().squeeze()
        ndvi = ndvi.numpy().squeeze()

        gt_ndvi = ndvi.copy()
        gt_ndvi[msk==0] = 0 

        ndvi_array['global'].append(np.mean(ndvi))
        ndvi_array['gt'].append(np.mean(gt_ndvi))

        msk =  np.stack((msk,msk,msk)).transpose(1,2,0)
        gt_ndvi = np.stack((gt_ndvi,gt_ndvi,gt_ndvi)).transpose(1,2,0)
        ndvi = np.stack((ndvi,ndvi,ndvi)).transpose(1,2,0)
        if len(im.shape)<3:
            im = np.stack((im,im,im)).transpose(1,2,0)
        else: 
            im =im.transpose(1,2,0)
   
        vis_img = np.hstack((im,msk,gt_ndvi,ndvi))

        vis_img = (vis_img * 255).astype(np.uint8)
        ax1.imshow(vis_img)
        plt.draw()
        plt.pause(10)

        

        print("[INF] global %f gt %f "%(np.mean(ndvi),np.mean(gt_ndvi)))





if __name__ == '__main__':
    root = 'E:\Dataset\greenAI\learning'
    
    # TEST_FRACTION(root,fraction=1)
    #TEST_NDVI(root,fraction=0.5)

    TEST_PLOT_DATA(root,fraction=0.5)


    


