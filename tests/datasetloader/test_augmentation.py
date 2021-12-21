
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
from dataset.augmentation import augment_rgb

HEIGHT = 240
WIDTH = 240
BATCH_SIZE = 5
MAX_EPOCH = 20
DATASET = ['qtabaixo']

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

  for sample in tqdm(dataset):
    image_vector.append(sample['bands'])
    mask_vector.append(sample['mask'])
    pred_vector.append(torch.rand(BATCH_SIZE,1,HEIGHT, WIDTH))

  image_vector = torch.stack(image_vector,axis=0)
  return({'imgs':image_vector,'masks':mask_vector,'preds':pred_vector})



def test_tf_writer_on_dataset(writer,dataset,mode)-> bool:

    images = get_data_from_dataset(dataset)

    imgs = images['imgs']
    masks = images['masks']
    pred = images['preds']

    for e,(i,m,p) in tqdm(enumerate(zip(imgs,masks,pred))):
      tb_frame = tf_writer.build_tb_frame(i,m,p)
      writer.add_image(tb_frame,e,mode)



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
    

    RANDOM = "TEST_AUG5"
    name = ''.join(['_'.join(DATASET),'h',str(HEIGHT),'w',str(WIDTH),'b',str(BATCH_SIZE),'e',str(MAX_EPOCH),])

    images,signals = synthetic_data_generator()
    
    writer = tf_writer.writer(os.path.join('saved',name + RANDOM),mode=['train','val','dataset'])

    #test_tf_writer_on_synthetic_data(writer,images,signals,'train')
    #test_tf_writer_on_synthetic_data(writer,images,signals,'val')


    root = 'samples'
    sensor = 'x7'
    bands = ['R','G','B']
    augment = False
    set = DATASET
    aug = augment_rgb()
  


    dataset= dataset_wrapper(
                          root,
                          set,
                          sensor, 
                          bands = {'R':True,'G':True,'B':True}, 
                          agro_index = {'NDVI':False}, 
                          transform = aug, 
                          path_type='global',
                          fraction = 0.1)
    
    dataset_loader = DataLoader( dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    pin_memory=False)

    test_tf_writer_on_dataset(writer,dataset_loader,'dataset')