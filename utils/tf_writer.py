
import argparse
from torch.utils import tensorboard
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms


def mask_array2PIL(array):
    array_ = (array.squeeze()*255).astype(np.uint8)
    new_mask = Image.fromarray(array_.astype(np.uint8)).convert('P')
    return(new_mask)


viz_transform = transforms.Compose([
            transforms.ToTensor()])
restore_transform = transforms.Compose([
            transforms.ToPILImage()])

# new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
def build_tb_frame(img,mask,pred):
    
    mask = mask.data.cpu().numpy()
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
    
  def add_loss(self,value,step):
    self.writer.add_scalar(f'{self.mode}/loss', value, step)
  
  def add_f1(self,value,step):
    self.writer.add_scalar(f'{self.mode }/f1', value, step)
  
  def add_image(self,img_frame,step):
    # build frame
   
    self.writer.add_image(f'{self.mode}/inputs_targets_predictions', img_frame, step)

def image_generator(height,width,batch,n_batches):

    image = torch.rand(n_batches,batch,3,height, width)
    masks = torch.rand(n_batches,batch,1,height, width)
    pred = torch.rand(n_batches,batch,1,height, width)

    return(image,masks,pred)

def signal_generator(samples):

    signal = torch.rand(samples)
    x_axis = torch.tensor(range(samples))
    #masks = torch.rand(batch,1,height, width)
    #pred = torch.rand(batch,1,height, width)

    return(signal,x_axis)




    
    #img_frame = Image.fromarray(frame)
    #img_frame.save("test.png")
