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

HEIGHT = 240
WIDTH = 240
BATCH_SIZE = 1
MAX_EPOCH = 20
DATASET = 'esac'

def image_generator(height,width,batch,n_batches):

    image = torch.rand(n_batches,batch,3,height, width)
    masks = torch.rand(n_batches,batch,1,height, width)
    pred = torch.rand(n_batches,batch,1,height, width)

    return(image,masks,pred)


def test_metrics():
    pass










if __name__ == '__main__':
    pass