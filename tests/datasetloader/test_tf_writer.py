
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

package_root  = os.path.dirname(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(package_root)

from utils import tf_writer
from dataset.learning_dataset import dataset_wrapper
from dataset.augmentation import augment_rgb as augmentation

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



def image_generator(height,width,batch,n_batches):

    image = torch.rand(n_batches,batch,3,height, width)
    masks = torch.rand(n_batches,batch,1,height, width)
    pred = torch.rand(n_batches,batch,1,height, width)

    return(image,masks,pred)

def signal_generator(samples):

    signal = torch.rand(samples)
    x_axis = torch.tensor(range(samples))

    return(signal,x_axis)

def synthetic_data_generator():
  images,masks,preds  = image_generator(HEIGHT,WIDTH,BATCH_SIZE,MAX_EPOCH)
  f1_signal,epoch     = signal_generator(MAX_EPOCH)
  loss_signal,epoch   = signal_generator(MAX_EPOCH)

  images = {'imgs':images,'masks':masks,'preds':preds}
  signals = {'f1':f1_signal,'loss':loss_signal,'epoch':epoch}

  return(images,signals)

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



def test_tf_writer_on_dataset(writer,dataset,mode)-> bool:

    images = get_data_from_dataset(dataset)

    imgs = images['imgs']
    masks = images['masks']
    pred = images['preds']
    names = images['name']
    imgs = generate_batches(imgs,5)
    masks = generate_batches(masks,5)
    pred = generate_batches(pred,5)
    names = generate_batches(names,5)

    for e,(i,m,p,n) in tqdm(enumerate(zip(imgs,masks,pred,names))):

      tb_frame = tf_writer.build_tb_frame(i,m,p)
      writer.add_image(tb_frame,e,mode)
      #input("Press Enter to continue...")



def test_tf_writer_on_synthetic_data(writer,images,signals,mode)-> bool:

   
    imgs = images['imgs']
    masks = images['masks']
    pred = images['preds']

    epoch = signals['epoch']
    f1 = signals['f1']
    loss = signals['loss']

    for i,m,p,e in zip(imgs,masks,pred,epoch):
      tb_frame = tf_writer.build_tb_frame(i,m,p)
      writer.add_image(tb_frame,e,mode)
      

    for l,f,e in zip(loss,f1,epoch):
      writer.add_f1(f,e,mode)
      writer.add_loss(l,e,mode)




if __name__ == '__main__':
    

    
    DATASET = ['qtabaixo','esac','valdoeiro']
    DATASET = ['esac','valdoeiro']

    root = '/home/tiago/learning'
    sensor = 'altum'
  
    #bands = {'R':True,'G':True,'B':True}
    bands = {'NIR':True}
    RANDOM = "DISPLAY_NIR"

    name = ''.join(['_'.join(DATASET),'s',sensor,'h',str(HEIGHT),'w',str(WIDTH),'b',str(BATCH_SIZE),'e',str(MAX_EPOCH),'_'.join(bands.keys())])
    
    writer = tf_writer.writer(os.path.join('saved',name + RANDOM),mode=['train','val','dataset'])

    #test_tf_writer_on_synthetic_data(writer,images,signals,'train')
    #test_tf_writer_on_synthetic_data(writer,images,signals,'val')


    #root = '/home/tiago/desktop_home/workspace/dataset/learning'

    
    augment = False
    #aug = augmentation(sensor_type = sensor)
    aug = None

    dataset= dataset_wrapper(
                          root,
                          DATASET,
                          sensor,
                          bands, 
                          agro_index = {'NDVI':False}, 
                          transform = aug, 
                          path_type='global', 
                          fraction = 0.1
                          )
    
    dataset_loader = DataLoader( dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    pin_memory=False)

    test_tf_writer_on_dataset(writer,dataset_loader,'dataset')
