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
from utils import segment_metrics as seg_metrics
from PIL import Image
import progressbar

def parse_name(file):
    h,w = file.split('_')
    return(int(h),int(w))

def _rebuild_ortho_mask_from_mem(prediction,theight,twidth):
    raster_mask = np.zeros((theight,twidth),dtype=np.uint8)
    
    #bar = progressbar.ProgressBar(max_value=len(prediction)) 
    for i,(pred_mask_dict) in enumerate(prediction):
        # file name parser
        pred_mask = pred_mask_dict['mask']
        name = pred_mask_dict['name']
        
        #print(name)
        ph,pw = parse_name(name)

        if len(pred_mask.shape)> 2:
          pred_mask = pred_mask.squeeze()
        
        h,w = pred_mask.shape
        lh,hh,lw,hw = ph,ph+h,pw,pw+w
        raster_mask[lh:hh,lw:hw] = pred_mask
        
    return(raster_mask)

def rebuild_ortho(prediction_masks,dest_dir,file,save=True):
  if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

  file_name = os.path.join(dest_dir,file + '.png')
  ortho_mask = _rebuild_ortho_mask_from_mem(prediction_masks,23500,22400)
  if save:
    img_pred_mask = (ortho_mask*255).astype(np.uint8)
    image = Image.fromarray(img_pred_mask)
    image.convert('RGB')
    image.save(file_name)
  return(ortho_mask)



def logit2label(array,thres):
    '''
    Convetion from a cxmxn logits torch array - where c represents the number of labels - to a 1xmxn label torch array 
    Parameters 
      - array: c x m x n torch array of logits
    Returns
      - 1 x m x n  array of labels (int) 
    '''
    norm_array = torch.sigmoid(array).detach().cpu().numpy()
    bin_class = (norm_array >= thres).astype(np.uint8)
    return(bin_class)


def evaluation(gtArray,predArray):
    '''
    segmentation evaluation 
    param:
     - gtArray (flatten): ground truth mask
     - predArray (flatten): prediction mask
    
    retrun:
        dictionary { dice , iou }

    '''
    
    if torch.is_tensor(gtArray):
      gtArray = gtArray.cpu().detach().numpy()
    
    if torch.is_tensor(predArray):
      predArray = predArray.cpu().detach().numpy()

    if len(gtArray.shape)>2:
      gtArray = gtArray.flatten()
    
    if len(predArray.shape)>2:
      predArray = predArray.flatten()
    
    gtArray = gtArray.astype(np.uint8)
    predArray = predArray.astype(np.int8)


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

  with torch.no_grad():
    torch.cuda.empty_cache()

  model.eval()

  masks = []
  preds = []
  
  seg_scores = seg_metrics.compute_scores()
  ndvi_array = {'global':[],'pred':[],'gt':[]}

  masks = []
  preds = []
  running_loss = []
  prediction_masks = []

  for batch in tqdm.tqdm(loader,'Validation'):

    img  = batch['bands']
    mask = batch['mask']
    ndvi = batch['indices']
    name = batch['name'][0]
    path = batch['path'][0]

    #img  = img.type(torch.FloatTensor).to(device)
    img  = img.to(device)
    msk  = (mask.cpu().detach().numpy()).astype(np.uint8)

    pred_mask = model(img)
    loss_torch = criterion(pred_mask,mask.to(device))

    bin_pred_mask = logit2label(pred_mask.detach(),0.5)
   
    prediction_masks.append({'mask':bin_pred_mask,'name':name})
    running_loss.append(loss_torch.detach().item())

    if torch.isnan(loss_torch):
      print("[WARN] WARNING NAN")

    masks.append(msk.flatten())
    preds.append(bin_pred_mask.flatten())
    
    #  # Convert array to image for visualization and storin

  Y = np.concatenate(masks, axis=None)
  PREDS = np.concatenate(preds, axis=None)

  scores = evaluation(Y,PREDS)
  #epoch_f1 = seg_scores.get_f1()
  epoch_f1 = scores['f1']
  val_loss = np.array(running_loss).mean()
  scores['val_loss'] = val_loss

  tb_frame = tf_writer.build_tb_frame(img,mask,bin_pred_mask)
  writer.add_image(tb_frame,epoch,'val')
  writer.add_f1(epoch_f1,epoch,'val')
  writer.add_loss(val_loss,epoch,'val')
  dataset_name = '_'.join(loader.dataset.plot)
  
  return({'f1':epoch_f1,'val_loss':val_loss},prediction_masks,dataset_name)



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--data_root', '-r',
      type=str,
      required=False,
      default='/home/tiago/learning',
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
      default='dev',
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
      '--writer',
      type=str,
      required=False,
      default='hd_segnet',
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
  writer_name = FLAGS.writer

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
  criterion = nn.BCEWithLogitsLoss() 

  writer_path = os.path.join('saved',session_settings['run'],writer_name)
  writer = tf_writer.writer(writer_path,mode= ['train','val'])

  metrics,prediction_mask,dataset_name = eval_net(network,val_loader,device,criterion,writer,0)

  name = '%s_f1_%02.2f.pth'%(dataset_name,(metrics['f1']*100)) 
  rebuild_ortho(prediction_mask,'predictions',name)

  print("[INF] Mean f1 %f"%(metrics['f1']))

