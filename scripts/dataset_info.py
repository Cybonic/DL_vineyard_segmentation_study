
import argparse
from torch.utils import tensorboard
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import os 
import pathlib
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader 

package_root  = os.path.dirname(pathlib.Path(__file__).parent.absolute())
sys.path.append(package_root)

from dataset.learning_dataset import dataset_wrapper,augmentation

HEIGHT = 240
WIDTH = 240
BATCH_SIZE = 5
MAX_EPOCH = 20

def generate_batches(tensorlist,batch_size):

  tensorlist_np = np.array(tensorlist)
  length = len(tensorlist)
  steps = int(length/batch_size)+1
  upper_idx = np.array(range(batch_size,length,batch_size))
  lower_idx = np.array(range(0,length,batch_size))
  if upper_idx[-1] != length:
    upper_idx = np.append(upper_idx,length)
  
  batch_list = []
  for l,u in zip(lower_idx,upper_idx):
      if tensorlist.dtype == '<U11':
        batch = tensorlist[l:u]
      else:
        batch = tensorlist[l:u,:,:,:]
      batch_list.append(batch)
  
  return(batch_list)


def get_data_from_dataset(dataset):

  image_vector = []
  mask_vector  = []
  pred_vector = []
  names_vector = []

  for sample in tqdm(dataset):
    image_vector.extend(sample['bands'])
    mask_vector.extend(sample['mask'])
    pred_vector.extend(torch.rand(BATCH_SIZE,1,HEIGHT, WIDTH))
    names_vector.extend(sample['name']) 

  image_vector = torch.stack(image_vector,axis=0)
  mask_vector = torch.stack(mask_vector,axis=0)
  pred_vector = torch.stack(pred_vector,axis=0)
  names_vector = np.array(names_vector)
  
  return({'imgs':image_vector,'masks':mask_vector,'preds':pred_vector,'name': names_vector})



def pixel_info(dataset_loader)-> bool:

    images = get_data_from_dataset(dataset)

    imgs = images['imgs']
    masks = images['masks']
    pred = images['preds']
    names = images['name']

    pos = 0
    neg = 0
    total = 0
    for e,(i,m,p,n) in tqdm(enumerate(zip(imgs,masks,pred,names))):
      m = m.numpy()
      pos_pixels = 1*(m == 1).sum()
      neg_pixels = 1*(m == 0).sum()
      tot_pixels = len(m.flatten())
      total = total + tot_pixels
      pos = pos + pos_pixels
      neg = neg + neg_pixels

    print("positive: %d"%(pos))
    print("negative: %d"%(neg))
    print("negative: %d"%(total))


    print("positive: %f"%(pos/total))
    print("Negative: %f"%(neg/total))
    print("Ration: %f"%(pos/neg))

    n_frames = masks.numpy().shape[0]
    print("Frames: %d"%(n_frames))

if __name__ == '__main__':
    

    
    DATASET = ['qtabaixo']
    # DATASET = ['qtabaixo']

    root = '/home/tiago/learning'
    sensor = 'x7'
  
    bands = {'R':True,'G':True,'B':True}
    #bands = {'NIR':True}
    RANDOM = "SHOW_data_NEW_13_norm"

    name = ''.join(['_'.join(DATASET),'s',sensor,'h',str(HEIGHT),'w',str(WIDTH),'b',str(BATCH_SIZE),'e',str(MAX_EPOCH),'_'.join(bands.keys())])
    
 

    #test_tf_writer_on_synthetic_data(writer,images,signals,'train')
    #test_tf_writer_on_synthetic_data(writer,images,signals,'val')


    #root = '/home/tiago/desktop_home/workspace/dataset/learning'

    
    augment = False
    aug = augmentation(sensor_type = sensor)
    aug = None

    dataset= dataset_wrapper(
                          root,
                          DATASET,
                          sensor,
                          bands, 
                          agro_index = {'NDVI':False}, 
                          transform = aug, 
                          path_type='global',
                          fraction = 1
                          )
    
    dataset_loader = DataLoader( dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    num_workers = 0,
                                    pin_memory=False)

    pixel_info(dataset_loader)
