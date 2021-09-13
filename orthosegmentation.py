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
from utils.saver import saver

from utils.segment_metrics import mean_iou_np,mean_dice_np
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.vis_utils import vis
import platform
import matplotlib.pyplot as plt

import orthoseg 





if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      default = "esac",
      required=False,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--sequence', '-c',
      type=str,
      default= ''
  )

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='modsegnet',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--session', '-f',
      type=str,
      required=False,
      default='hd/rgb',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      # default="",
      default = "checkpoints/t3/ms/modsegnet/nir_f1_81",
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--debug', '-b',
      type=int,
      required=False,
      default=False,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--plot',
      type=int,
      required=False,
      default=0,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--results',
      type=str,
      required=False,
      default='session_results.txt',
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  
  session = FLAGS.session
  model_name = FLAGS.model
  pretrained = None if FLAGS.pretrained == "" else FLAGS.pretrained
  
  session_file = os.path.join('session',session + '.yaml')
  if not os.path.isfile(session_file):
    print("[ERR] Session file does not exist: " + session_file)
    raise NameError(session_file)

  # load session parameters
  session_settings = utils.load_config(session_file)
  # Load network with specific parameters

  pc_name = platform.node() 
  print("PC Name: " + pc_name)
  if pc_name == 'DESKTOP-SSEDT6V':
    root = "E:\\dataset"
  else:
    root = "/home/tiago/BIG/dataset"

  ortho_dir = os.path.join(root,"greenAI/drone/quintabaixo/04_05_2021/60m/x7")
  # ortho_dir = os.path.abspath(ortho_dir)

  if not os.path.isdir(ortho_dir):
    print("[ERROR] dir does not exist")

  path_to_ortho_img = os.path.join(ortho_dir,"ortho.tif")
  # ortho_file = os.path.join(root,path)
  print("[INF] Path to ortho img: %s"%(path_to_ortho_img))

  ortho_pipeline = orthoseg.orthoseg()

  ortho_mask = ortho_pipeline.pipeline(path_to_ortho_img)



