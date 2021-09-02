#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


'''


 
'''
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import signal, sys
import numpy as np
# from utils import train_utils as loss
np.seterr(divide='ignore', invalid='ignore')
# from torch.utils.tensorboard import SummaryWriter


import utils.utils as utils
from dataset.learning_dataset import  dataset_loader

#from modules.rattnet import *
from datetime import datetime
import random

import torch
from torch import optim
from torch import nn
#from utils import dump_info
from networks.orthoseg import OrthoSeg 
from utils.saver import saver 
# from inference import eval_net
from inference import eval_net
from inference import logit2label
from inference import evaluation
import platform

LOSS_WEIGHTS = {'RGBX7':4.9,'Multispectral':5.05}

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

  #for key, value in index.items():
  #  if value == True:
  #    count = count +1

  model = OrthoSeg(network_param,image_shape,channels=count,drop_rate = drop_rate)

  if pretrained_flag == True:
    pretrained_to_load = os.path.join(pretrained_path,pretrained_file+'.pth')
    if os.path.isfile(pretrained_to_load):
      print("[INF] Loading Pretrained model: " + pretrained_to_load)
      model.load_state_dict(torch.load(pretrained_to_load))
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

# ==================================================

def dataset_loader_wrapper(session_settings):
  """
  
  """
  pc_name = platform.node()
  print("[INFO]: "+ pc_name)
  dataset = session_settings['dataset']
  root =  dataset[pc_name] 
  sensor = dataset['name']
  param = dataset['loader']
  
  if 'fraction' in dataset:
    fraction = dataset['fraction']
  else: 
    fraction = None

  shuffle     = param['shuffle']
  batch_size  = param['batch_size']
  workers     = param['workers']

  bands       =  session_settings['network']['bands']
  agro_index  =  session_settings['network']['index']

  trainset = dataset['train']
  testset  = dataset['test']

  augment = dataset['augment']

  dataset= dataset_loader(root = root,
                          sensor = sensor,
                          bands = bands,
                          agro_index= agro_index,
                          augment = augment,
                          trainset = trainset,
                          testset = testset,
                          batch_size = batch_size ,
                          shuffle = shuffle ,
                          workers = workers,
                          fraction = {'train':fraction,'test':fraction}
                          )

  test = dataset.get_test_loader()
  train = dataset.get_train_loader()
  return(train,test)


def load_optimizer_wrapper(model,parameters):

  param = session_settings['optimizer']

  lr            = param['lr']
  w_decay       = param['w_decay']
  amsgrad       = param['amsgrad']
  epsilon       = param['epsilon_w']
  betas         = tuple(param['betas'])


  optimizer = optim.AdamW(
              model.parameters(), 
              lr = lr,
              weight_decay=w_decay
              # eps = epsilon,
              # amsgrad = amsgrad,
              # betas = betas,
              )

  # ============================================================================
  # LOSS Function
  '''
  For example, if a dataset contains 100 positive and 300 negative examples of a single class, 
  then pos_weight for the class should be equal to \frac{300}{100}=3 
  The loss would act as if the dataset contains 3\times 100=3003×100=300 positive examples.
  '''

  # The imbalance of the dataset is in mean 5 negatives for 1 positive, both for HD and MS
  pos_weight =  torch.tensor([5]) 
  #criterion_weighted = loss.dice_loss()
  #criterion_weighted = loss.GDiceLoss()
  criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
  # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
  
  return(optimizer, criterion_weighted)

  # ============================================================================= 


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='segnet',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--session', '-f',
      type=str,
      required=False,
      default='ms/rgb',
      #default='unet_rgb_augment',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default="checkpoints/darknet53-512",
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

  

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Plot flag: ", FLAGS.plot)
  print("----------\n")

  plot_flag  = FLAGS.plot
  session    = FLAGS.session
  
  session_file = os.path.join('session',session + '.yaml')
  if not os.path.isfile(session_file):
    print("[ERR] Session file does not exist: " + session_file)
    raise NameError(session_file)
  # load session parameters
  session_settings = utils.load_config(session_file)
  # Load network with specific parameters
  model, pretrained_path, device = network_wrapper(session_settings)
  # Load dataset 
  # Get train and val loaders
  train_loader, val_loader = dataset_loader_wrapper(session_settings)
  # Set the loss function
   
  optimizer, criterion = load_optimizer_wrapper(model,session_settings)
  criterion = criterion.to(device)
  
  # Saving settings
  saveing_param = session_settings['saver']  
  saver_handler = saver(**session_settings['saver'])

  #writer = SummaryWriter(log_dir='log',comment = saver_handler.get_result_file() )

  epochs    = session_settings['max_epochs']
  VAL_EPOCH = session_settings['report_val']

  session_name = session_settings['name']
  print("\n--------------------------------------------------------")
  print("[INF] Max Epochs: %d"%(epochs))
  print("[INF] Validation report epoch: %d"%(VAL_EPOCH))
  print("[INF] Loaded Model: " + session_settings['network']['model'])
  print("[INF] Device: " + device)
  print("[INF] Result File: " + saver_handler.get_result_file())
  print("[INF] Session: " + session_name)
  print("--------------------------------------------------------\n")

  
  global_val_score = {'f1':-1,'epoch':0}
  try:
    for epoch in range(epochs):
          
      running_loss  = 0
      loss_bag = np.array([])
      
      sub_epoch = epoch
      
      masks = []
      preds = []

      for batch in tqdm.tqdm(train_loader):
        model.train()

        img  = batch['bands']
        mask = batch['mask']

        img = img.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        # compute output
        pred_mask = model(img)
        loss_torch = criterion(pred_mask,mask)
        loss_torch.backward()
        optimizer.step()
        
        loss = loss_torch.detach().item()
        running_loss += loss
        if torch.isnan(loss_torch):
          print("[WARN] WARNING NAN")
      
        # Stack to evaluate later
        pred_mask = logit2label(pred_mask,0.5) # Input (torch) output (numpy)
        mask = mask.cpu().detach().numpy()
        
        masks.append(mask.flatten())
        preds.append(pred_mask.flatten())
        # Plotting 
      
      Y = np.concatenate(masks, axis=None)
      PREDS = np.concatenate(preds, axis=None)

      scores = evaluation(Y,PREDS)

      train_loss = running_loss/len(train_loader)
      #writer.add_scalar('Loss/train', train_loss, epoch)
      print("train epoch : {}/{}, loss: {:.6f} f1: {:.3f}".format(epoch, epochs, train_loss,scores['f1']))
      

      if epoch % VAL_EPOCH == 0:
        # Compute valuation
        metric = eval_net( model,val_loader, device,plot_flag) 
        
        print("[INF] Mean f1 %0.2f global %0.2f  gt %0.2f pred %0.2f"%(metric['f1'],metric['ndvi_global'],metric['ndvi_gt'],metric['ndvi_pred']))
        # writer.add_scalar('f1/test', metric['f1'], epoch)
        if metric['f1'] > global_val_score['f1']:
          # Overwite
          global_val_score  = metric
          global_val_score['loss']  = train_loss
          global_val_score['epoch'] = epoch

          if session_settings['network']['pretrained']['use'] == True: 
            trained_model_name = '%s_f1_%02d.pth'%(session_name,(metric['f1']*100))
            checkpoint_dir = os.path.join(pretrained_path,session_settings['network']['model'])
            if not os.path.isdir(checkpoint_dir):
              os.makedirs(checkpoint_dir)
            checkpoint_name = os.path.join(checkpoint_dir,trained_model_name)
            torch.save(model.state_dict(),checkpoint_name) # torch.save(model.state_dict(), trained_weights)
            print("[INF] weights stored at: " + checkpoint_name)
          

          
  
  except KeyboardInterrupt:
    print("[INF] CTR + C")
  
  #except:
  #  print("[INF] Error")
  
  text_to_store = {}
  text_to_store['model'] = session_settings['network']['model']
  text_to_store['session'] =  session_name
  text_to_store['f1'] =  global_val_score['f1']
  text_to_store['ndvi_global'] =  global_val_score['ndvi_global']
  text_to_store['ndvi_gt'] =  global_val_score['ndvi_gt']
  text_to_store['ndvi_pred'] = global_val_score['ndvi_pred']
  text_to_store['epoch'] =  "%d/%d"%(global_val_score['epoch'],epochs)
  text_to_store['drop_rate'] =  session_settings['network']['drop_rate']
  text_to_store['lr'] =  session_settings['optimizer']['lr']
  text_to_store['w_decay'] =  session_settings['optimizer']['w_decay']
  bands =  ' '.join([ key  for key, value in session_settings['network']['bands'].items() if value==True ])
  index =  ' '.join([ key  for key, value in session_settings['network']['index'].items() if value==True ])

  text_to_store['bands']  = bands + ' ' + index
  # Save training information to results file
  output_txt = saver_handler.save_results(text_to_store)
  
  print("[INF] " + output_txt)


