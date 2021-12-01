
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

package_root  = os.path.dirname(pathlib.Path(__file__).parent.absolute())
sys.path.append(package_root)

from utils import tf_writer

def image_generator(height,width,batch,n_batches):

    image = torch.rand(n_batches,batch,3,height, width)
    masks = torch.rand(n_batches,batch,1,height, width)
    pred = torch.rand(n_batches,batch,1,height, width)

    return(image,masks,pred)

def signal_generator(samples):

    signal = torch.rand(samples)
    x_axis = torch.tensor(range(samples))

    return(signal,x_axis)

def data_generator():
  images,masks,preds  = image_generator(HEIGHT,WIDTH,BATCH_SIZE,MAX_EPOCH)
  f1_signal,epoch     = signal_generator(MAX_EPOCH)
  loss_signal,epoch   = signal_generator(MAX_EPOCH)

  images = {'imgs':images,'masks':masks,'preds':preds}
  signals = {'f1':f1_signal,'loss':loss_signal,'epoch':epoch}

  return(images,signals)

def test_tf_writer(writer,images,signals)-> bool:

    imgs = images['imgs']
    masks = images['masks']
    pred = images['preds']

    epoch = signals['epoch']
    f1 = signals['f1']
    loss = signals['loss']

    for i,m,p,e in zip(imgs,masks,pred,epoch):
      tb_frame = tf_writer.build_tb_frame(i,m,p)
      writer.add_image(tb_frame,e)

    for i,m,p,e in zip(loss,f1,epoch):
      writer.add_f1(f1,e)
      writer.add_loss(loss,e)
    #  tf_writer.writer.add_f1(f1,epoch)
    #  tf_writer.writer.add_loss(loss,epoch)


if __name__ == '__main__':
    
    HEIGHT = 240
    WIDTH = 240
    BATCH_SIZE = 10
    MAX_EPOCH = 20

    root = '/home/tiago/workspace/dataset/learning'
    vineyard_plot = 'qtabaixo'
    sensor = 'x7'

    name = ''.join(['h',str(HEIGHT),'w',str(WIDTH),'b',str(BATCH_SIZE),'e',str(MAX_EPOCH),])

    images,signals = data_generator()
    
    writer = tf_writer.writer(os.path.join('results',name))

    

    test_tf_writer(writer,images,signals)


    
    #img_frame = Image.fromarray(frame)
    #img_frame.save("test.png")
