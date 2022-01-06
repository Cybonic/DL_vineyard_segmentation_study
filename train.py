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
from torch.utils import tensorboard
from PIL import Image
from utils import tf_writer
from utils import segment_metrics as seg_metrics
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
from inference import eval_net,rebuild_ortho
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
  model_name = network_param['model']


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
  root = os.path.join('checkpoints',model_name,pretrained_path)

  if pretrained_flag == True:
    
    pretrained_to_load = os.path.join(root,pretrained_file+'.pth')
    if os.path.isfile(pretrained_to_load):
      print("[INF] Loading Pretrained model: " + pretrained_to_load)
      model.load_state_dict(torch.load(pretrained_to_load))
    else:
      print("[INF] No pretrained weights loaded: " + pretrained_to_load)

    if not os.path.isdir(root):
      os.makedirs(root)

  # Device configuration
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'
    device = session_settings['device']
    torch.cuda.empty_cache()

  model.to(device)
  model.train()
  return(model, root,device)

# ==================================================

def dataset_loader_wrapper(root,session_settings,savage_mode):
  """
  dataset parser

  Input: 
   - session_settings: file name to settings
  Output: 
   - testloader
   - trainloader

  """
  pc_name = platform.node()
  print("[INFO]: "+ pc_name)
  dataset = session_settings['dataset']
  # root =  dataset['root'] 
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

  dataset= dataset_loader(root = root, # path to dataset 
                          sensor = sensor, # [Multispectral,RGBX7]
                          bands = bands, # [R,G,B,NIR,RE,Thermal]
                          agro_index= agro_index,# [NDVI]
                          augment = augment, #[True, False]
                          trainset = trainset, #[esac1,esca2,valdoeiro] 
                          testset = testset, #[esac1,esca2,valdoeiro] 
                          batch_size = batch_size ,
                          shuffle = shuffle ,
                          workers = workers,
                          fraction = {'train':fraction,'test':fraction}, # [0,1]
                          savage_mode=savage_mode
                          )

  # Get loaders
  testloader = dataset.get_test_loader()
  trainloader = dataset.get_train_loader()

  return(trainloader,testloader)


# ==================================================

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
              weight_decay=w_decay,
              # eps = epsilon,
              amsgrad = amsgrad,
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
  criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  #criterion_weighted = nn.BCEWithLogitsLoss() 
  # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
  
  return(optimizer, criterion_weighted)

  # ============================================================================= 





if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--data_root', '-r',
      type=str,
      required=False,
      #default='/home/tiago/learning',
      #default='/home/tiago/desktop_home/workspace/dataset/learning'
      default='/home/tiago/workspace/dataset/learning',
      #default='samples',
      help='Directory to get the trained model.'
  )


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
      #default='hd/rgb',
      default='ms/rgb_nir',
      #default='dev',
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
      '--writer',
      type=str,
      required=False,
      default='aug/color_space/soft_aug/full/segnet',
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
  root  = FLAGS.data_root
  writer_name = FLAGS.writer
  
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
  train_loader, val_loader = dataset_loader_wrapper(root,session_settings,savage_mode =0)
  # Set the loss function
   
  optimizer, criterion = load_optimizer_wrapper(model,session_settings)
  criterion = criterion.to(device)
  
  # Saving settings
  saveing_param = session_settings['saver']  
  saver_handler = saver(**session_settings['saver'])
  writer_path = os.path.join('saved',session_settings['run'],writer_name)
  #writer_path = os.path.join('saved',writer_name)
  
  #os.makedirs('log')

  writer = tf_writer.writer(writer_path,mode= ['train','val'])
  seg_scores = seg_metrics.compute_scores()
  epochs    = session_settings['max_epochs']
  VAL_EPOCH = session_settings['report_val']

  session_name = session_settings['name'] #+':' + writer_name
  print("\n--------------------------------------------------------")
  print("[INF] Max Epochs: %d"%(epochs))
  print("[INF] Validation report epoch: %d"%(VAL_EPOCH))
  print("[INF] Loaded Model: " + session_settings['network']['model'])
  print("[INF] Device: " + device)
  print("[INF] Result File: " + saver_handler.get_result_file())
  print("[INF] Session: " + session_name)
  print("[INF] Writer: " + writer_name)
  print("--------------------------------------------------------\n")

  prediction_mask_bundle = []
  mask_to_disply = []
  indice_to_save= random.randint(0, epochs)

  global_val_score = {'f1':-1,'epoch':0,'model':''} # keeps the best values
  try:
    for epoch in range(epochs):
          
      running_loss  = 0
      loss_bag = np.array([])
      
      sub_epoch = epoch
      
      masks = []
      preds = []

      for i,batch in tqdm.tqdm(enumerate(train_loader)):
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

        bin_pred_mask = logit2label(pred_mask.detach(),0.5)
        
        masks.append(mask.detach().cpu().numpy().flatten())
        preds.append(bin_pred_mask.flatten())
        # Plotting 
      
      Y = np.concatenate(masks, axis=None)
      PREDS = np.concatenate(preds, axis=None)
      scores = evaluation(Y,PREDS)
      epoch_f1  = scores['f1']
      #epoch_f1 = seg_scores.get_f1()
      train_loss = running_loss/len(train_loader)
      
      tb_frame = tf_writer.build_tb_frame(img,mask,bin_pred_mask)
      writer.add_image(tb_frame,epoch,'train')
      writer.add_f1(epoch_f1,epoch,'train')
      #writer.add_f1(scores['f1'],epoch,'train')
      writer.add_loss(train_loss,epoch,'train')

      #writer.add_scalar('Loss/train', train_loss, epoch)
      print("[INF|Train] epoch : {}/{}, loss: {:.6f} f1: {:.3f}".format(epoch, epochs, train_loss,epoch_f1))
      

      if epoch % VAL_EPOCH == 0:
        # Compute valuation
        metric,prediction_masks,dataset_name = eval_net(model,val_loader,device,criterion,writer,epoch) 

        print("[INF|Test] Mean Loss %0.2f Mean f1 %0.2f"%(metric['val_loss'],metric['f1']))
        # writer.add_scalar('f1/test', metric['f1'], epoch)
        if metric['f1'] > global_val_score['f1']:
          
          name = '%s_f1_%02.2f.pth'%(dataset_name,(metric['f1']*100)) 
          ortho_mask = rebuild_ortho(prediction_masks,'predictions',name)
          #writer.add_orthomask(ortho_mask,epoch,'val')
          # Overwite
          global_val_score  = metric
          global_val_score['train_loss'] = train_loss
          global_val_score['epoch'] = epoch
         
          if session_settings['network']['pretrained']['save'] == True: 
            # model name
            trained_model_name = '%s_f1_%03d.pth'%(session_name,(metric['f1']*100)) 
            # build model path
            checkpoint_dir = os.path.join(pretrained_path,'all')
            if not os.path.isdir(checkpoint_dir):
              os.makedirs(checkpoint_dir)
            # Build full model path
            checkpoint_name = os.path.join(checkpoint_dir,trained_model_name)
            # save model weights 
            torch.save(model.state_dict(),checkpoint_name) # torch.save(model.state_dict(), trained_weights)
            print("[INF] weights stored at: " + checkpoint_name)
            # keep the best model for later
            global_val_score['model'] = {'path':checkpoint_name,'name':trained_model_name}

  
  except KeyboardInterrupt:
    print("[INF] CTR + C")
  
  #Save best model 
  if session_settings['network']['pretrained']['save'] == True: 
    
    destination = os.path.join(
                    pretrained_path,
                    session_settings['network']['pretrained']['file'] +'.pth'
                    )
    try:
        shutil.copy(global_val_score['model']['path'], destination)
    except:
        print("[Error] File not copied: " + destination ) 

    print("[INF] Saved best model to : " + destination )

  text_to_store = {}
  text_to_store['model'] = session_settings['network']['model']
  text_to_store['session'] =  session_name
  text_to_store['f1'] =  global_val_score['f1']
  #text_to_store['ndvi_global'] =  global_val_score['ndvi_global']
  #text_to_store['ndvi_gt'] =  global_val_score['ndvi_gt']
  #text_to_store['ndvi_pred'] = global_val_score['ndvi_pred']
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


