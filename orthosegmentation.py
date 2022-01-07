'''

Author: Tiago Barros 
Project: GreenAi
year: May,2021

'''


import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import tqdm
import signal, sys
import numpy as np


import utils.utils as utils
#from dataset.learning_dataset import  dataset_loader

from datetime import datetime
import random

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from networks.orthoseg import OrthoSeg 
from utils.utils import get_files

from utils.segment_metrics import mean_iou_np,mean_dice_np
from sklearn.metrics import f1_score, precision_score, recall_score

import platform
import matplotlib.pyplot as plt

from networks import unet_bn 
#from networks import MFNet
from networks import segnet
from networks import modsegnet

import orthoseg_pipeline as orthoseg
from PIL import Image

def network_wrapper(session_settings,pretrained_file= None):

  network_param = session_settings['network']
  image_shape = network_param['input_shape']
  bands= network_param['bands']
  index = network_param['index']
  pretrained_path = network_param['pretrained']['path']
  pretrained_flag = network_param['pretrained']['use']
  drop_rate = network_param['drop_rate']


  # Dump to terminal 
  print("-"*100)
  print("[Network] " )
  for key, value in network_param.items():
    print("{}: {} ".format(key,value))
  print("-"*100)
  if pretrained_file == None:
    pretrained_file = network_param['pretrained']['file']

  count = 0
  for key, value in bands.items():
    if value == True:
      count = count +1

  model_name = network_param['model']

  if model_name == 'segnet':
      model = segnet.SegNet(num_classes=1, n_init_features=count,drop_rate= drop_rate)
  elif model_name == 'unet_bn':
      model = unet_bn.UNET(out_channels=1, in_channels=count) # UNet has no dropout
  elif model_name == 'modsegnet':
      model = modsegnet.ModSegNet(num_classes=1, n_init_features=count,drop_rate= drop_rate)

  #model = OrthoSeg(network_param,image_shape,channels=count,drop_rate = drop_rate)

  if pretrained_flag == True:
    pretrained_to_load = os.path.join(pretrained_path,pretrained_file+'.pth')
    if os.path.isfile(pretrained_to_load):
      print("[INF] Loading Pretrained model: " + pretrained_to_load)
      model.load_state_dict(torch.load(pretrained_to_load))
    else:
      print("[INF] No pretrained weights loaded: " + pretrained_to_load)


  # Device configuration
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()

  

  return(model,device)



def main(session_settings,input_file,output_file):

  # Load network with specific parameters
  model, device = network_wrapper(session_settings)

  device = 'cpu'

  model.to(device)

  if not os.path.isfile(input_file):
    print("[ERROR] File does not exist: %s"%(input_file))
    exit(0)

  ortho_pipeline = orthoseg.orthoseg( model  = model,
                                      device = device
                                      )

  raster = ortho_pipeline.load_ortho(input_file)

  ortho_mask = ortho_pipeline.pipeline(raster)

  ortho_mask_pil = Image.fromarray(ortho_mask).convert('L')
  ortho_mask_pil.save(output_file+'.png')




if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--session', '-f',
      type=str,
      required=False,
      default='ms/rgb',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--input_ortho_file', '-i',
      type=str,
      required=False,
      default="/home/tiago/greenai/learning/esac/altum/ortho.tif",
      help='path from the Orthomosaic file from the root of the datast.'
  )
  parser.add_argument(
      '--output_ortho_file', '-o',
      type=str,
      required=False,
      default="pred_mask_ortho",
      help='path from the Orthomosaic file from the root of the datast.'
  )

  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      # default="",
      default = "",
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  
  input_ortho_file  = FLAGS.input_ortho_file
  output_ortho_file = FLAGS.output_ortho_file
  session           = FLAGS.session
  pretrained        = None if FLAGS.pretrained == "" else FLAGS.pretrained

  
  session_file = os.path.join('session',session + '.yaml')
  if not os.path.isfile(session_file):
    print("[ERR] Session file does not exist: " + session_file)
    raise NameError(session_file)

      # load session parameters
  session_settings = utils.load_config(session_file)

  main(session_settings,input_ortho_file,output_ortho_file)



