
import argparse
from torch.utils import tensorboard
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch


def mask_array2PIL(array):
    array_ = (array.squeeze()*255).astype(np.uint8)
    new_mask = Image.fromarray(array_.astype(np.uint8)).convert('P')
    return(new_mask)


viz_transform = transforms.Compose([
            transforms.ToTensor()])

restore_transform = transforms.Compose([
            transforms.ToPILImage()])

def build_tb_frame(img,mask,pred):

    if not isinstance(mask,np.ndarray):
      mask = mask.data.cpu().numpy()
    if not isinstance(pred,np.ndarray):
      pred = pred.data.cpu().numpy()

    val_visual = [[i.data.cpu(), j, k] for i, j, k in zip(img, mask, pred)]
    val_img = []
    for imgs in val_visual:
        imgs = [restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                    else mask_array2PIL(i) for i in imgs]
        imgs = [i.convert('RGB') for i in imgs]
        imgs = [viz_transform(i) for i in imgs]
        val_img.extend(imgs)
    val_img = torch.stack(val_img, 0)
    val_img = torchvision.utils.make_grid(val_img.cpu(),nrow=val_img.size(0)//len(val_visual),padding=5)

    return(val_img)
  

class writer():
  def __init__(self,name,mode='train'):
    self.writer = tensorboard.SummaryWriter(name)
    #writer = SummaryWriter(log_dir='log',comment = saver_handler.get_result_file() )
    self.wrt_step = 0
    self.mode = mode

  def _add_scalar(self,metric,value,step,mode):
    if not mode in self.mode:
      raise NameError(mode)
    self.writer.add_scalar(f'{mode}/{metric}', value, step)

  def add_loss(self,value,step,mode):
    self._add_scalar('loss',value,step,mode=mode)
  
  def add_f1(self,value,step,mode='train'):
    self._add_scalar('f1',value,step,mode=mode)
  
  def add_orthomask(self,value,step,mode='train'):
    value_tensor = viz_transform(value)
    #self.add_image(f'{mode}/ortho',step,value)
    self.writer.add_image(f'{mode}/ortho', value_tensor, step)


  def add_image(self,img_frame,step,mode='train'):
    # build frame
    if not mode in self.mode:
      raise NameError(mode)
    self.writer.add_image(f'{mode}/inputs_targets_predictions', img_frame, step)
