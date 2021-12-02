
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

from inference import evaluation

HEIGHT = 240
WIDTH = 240
BATCH_SIZE = 1
MAX_EPOCH = 20
DATASET = 'valdoeiro'

def image_generator(height,width,batch,n_batches):

    image = torch.rand(n_batches,batch,3,height, width)
    masks = torch.randint(0, 2,(n_batches,batch,1,height,width),dtype=torch.float32)
    pred = torch.randint(0, 2,(n_batches,batch,1,height,width),dtype=torch.float32)

    
    return(image,masks,pred)



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



def test_metric(masks,pred,expected_value)-> bool:

    metric = evaluation(masks,pred)

    f1 = metric['f1']
    iou = metric['iou']

    return(f1 == expected_value)




if __name__ == '__main__':
    

    RANDOM = "TEST"
    name = ''.join([DATASET,'h',str(HEIGHT),'w',str(WIDTH),'b',str(BATCH_SIZE),'e',str(MAX_EPOCH),])

    img,masks,preds = image_generator(HEIGHT,WIDTH,BATCH_SIZE,MAX_EPOCH)
    print(masks.max())
    print(masks.min())
    print("[Test] Metric %d"%(test_metric(masks,masks,1)))

    
