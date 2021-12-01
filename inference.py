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
from dataset.learning_dataset import  dataset_loader

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
from utils import tf_writer



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
    #norm_array[norm_array >= thres]  = 1
    # norm_array[norm_array < thres] = 0
    new_array[bin_class] = 1
    

    return(new_array)


def evaluation(gtArray,predArray):
    '''
    segmentation evaluation 
    param:
     - gtArray: ground truth mask
     - predArray: prediction mask
    
    retrun:
        dictionary { dice , iou }

    '''
    
    #dice = mean_dice_np(gtArray,predArray)
    #iou = mean_iou_np(gtArray,predArray)

    #gtArray = gtArray.flatten()
    #predArray = predArray.flatten()

    gtArray = np.array(gtArray,dtype=np.int32)
    predArray = np.array(predArray,dtype=np.int32)
    f1 = f1_score(gtArray,predArray)

    metrics = {'f1': f1}
    return(metrics)

def network_wrapper(session_settings,model = None ,pretrained_file= None):

  network_param = session_settings['network']
  if model is not None:
    network_param['model'] = model
  image_shape = network_param['input_shape']
  bands= network_param['bands']
  index = network_param['index']
  pretrained_path = network_param['pretrained']['path']
  pretrained_flag = network_param['pretrained']['use']


  count = 0
  for key, value in bands.items():
    if value == True:
      count = count +1

  # for key, value in index.items():
  #   if value == True:
  #    count = count +1

  model = OrthoSeg(network_param,image_shape,channels=count)

  if pretrained_file == None:
    pretrained_file = network_param['pretrained']['file']
    if pretrained_flag == True:
      pretrained_to_load = os.path.join(pretrained_path,pretrained_file+'.pth')
      if os.path.isfile(pretrained_to_load):
        print("[INF] Loading Pretrained model: " + pretrained_to_load)
        model.load_state_dict(torch.load(pretrained_to_load))
      else:
        print("[INF] No pretrained weights loaded: " + pretrained_to_load)
  else: 
    pretrained_to_load = pretrained_file+'.pth'
    if os.path.isfile(pretrained_to_load):
      
      model.load_state_dict(torch.load(pretrained_to_load))
      print("[INF] Loaded Pretrained model: " + pretrained_to_load)
    else:
      print("[INF] No pretrained weights loaded: " + pretrained_to_load)

  if not os.path.isdir(pretrained_path):
    os.makedirs(pretrained_path)
  # Device configuration
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()

  model.to(device)
  model.train()
  return(model, pretrained_path,device)

def dataset_loader_wrapper(root,session_settings):

  pc_name = platform.node()
  print("[INFO]: "+ pc_name)
  dataset = session_settings['dataset']
  #root =  dataset[pc_name] 
  sensor = dataset['name']
  param = dataset['loader']

  shuffle = param['shuffle']
  batch_size = param['batch_size']
  workers = param['workers']

  bands =  session_settings['network']['bands']
  agro_index =  session_settings['network']['index']

  testset  = dataset['test']

  augment = dataset['augment']

  dataset= dataset_loader(root = root,
                          sensor = sensor,
                          bands = bands,
                          agro_index= agro_index,
                          augment = augment,
                          trainset = [],
                          testset = testset,
                          batch_size = batch_size ,
                          shuffle = shuffle ,
                          workers = workers
                          )

  test = dataset.get_test_loader()
  #train = dataset.get_train_loader()
  return(test)

def eval_net(model,loader,device,criterion,writer,epoch,save_flag = False):
  '''
  Network Evaluation 

  Parameters

  - model: deeplearning network
  - loader: dataset loader
  - device: cpu or GPU

  Return: 

  - dictionary{iou,dice}

  '''

  # print("[EVAL_NET] device: %s"%(device))
  
  model.eval()
  #model = model.to(device)

  masks = []
  preds = []
  
  

  ndvi_array = {'global':[],'pred':[],'gt':[]}

  masks = []
  preds = []
  running_loss = 0

  for batch in tqdm.tqdm(loader,'Validation'):

    img  = batch['bands']
    mask = batch['mask']
    ndvi = batch['indices']
    name = batch['name'][0]
    path = batch['path'][0]

    img  = img.type(torch.FloatTensor).to(device)
    msk  = np.array(mask.cpu().detach().numpy(),dtype=np.int32)
    ndvi = np.array(ndvi.cpu().detach().numpy())

    ndvi[np.isnan(ndvi)]=0

    pred_mask = model(img)
    loss_torch = criterion(pred_mask,mask.to(device))
    # transform logit to label  
    pred_mask = logit2label(pred_mask,0.5)
    
    loss = loss_torch.detach().item()
    running_loss += loss
    if torch.isnan(loss_torch):
      print("[WARN] WARNING NAN")

  
    masks.append(msk.flatten())
    preds.append(pred_mask.flatten())
    
    #  # Convert array to image for visualization and storin

  Y = np.concatenate(masks, axis=None)
  PREDS = np.concatenate(preds, axis=None)

  scores = evaluation(Y,PREDS)

  val_loss = running_loss/len(loader)
  scores['loss'] = val_loss

  tb_frame = tf_writer.build_tb_frame(img,mask,pred_mask)
  writer.add_image(tb_frame,epoch,'val')
  writer.add_f1(scores['f1'],epoch,'val')
  writer.add_loss(val_loss,epoch,'val')
  
  return(scores)



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--data_root', '-r',
      type=str,
      required=False,
      default='/home/tiago/workspace/dataset/learning',
      #default='/home/tiago/desktop_home/workspace/dataset/learning/valdoeiro/',
      help='Directory to get the trained model.'
  )

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
      default = "checkpoints/ms/segnet/nir_f1_81",
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
      default=1,
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
  root = FLAGS.data_root
  plot_flag = FLAGS.plot

  pretrained = None if FLAGS.pretrained == "" else FLAGS.pretrained
  
  session_file = os.path.join('session',session + '.yaml')
  if not os.path.isfile(session_file):
    print("[ERR] Session file does not exist: " + session_file)
    raise NameError(session_file)

  # load session parameters
  session_settings = utils.load_config(session_file)
  # Load network with specific parameters
  network, pretrained_path, device = network_wrapper(session_settings,model= model_name, pretrained_file = pretrained)
  # Load dataset 
  # Get train and val loaders
  val_loader = dataset_loader_wrapper(root,session_settings)

  scores  = eval_net(network,val_loader,device,plot_flag = plot_flag,save_flag = True, save_path=os.path.join('fig',model_name))

  print("[INF] Mean f1 %f"%(scores['f1']))

