
import argparse
from torch.utils import tensorboard
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms


def array2PIL_mask(array):
    array_ = (array.squeeze()*255).astype(np.uint8)
    new_mask = Image.fromarray(array_.astype(np.uint8)).convert('P')
    return(new_mask)

def build_h_frame(img,mask,pred):
  img_cpu = (img*255).astype(np.uint8)
  mask = array2img_mask(mask)
  pred = array2img_mask(pred)
  return(np.hstack((img_cpu,mask,pred),axis=0))

viz_transform = transforms.Compose([
            transforms.ToTensor()])
restore_transform = transforms.Compose([
            transforms.ToPILImage()])

# new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
def add_image_tb(img,mask,pred):
    frame =[]
    
    mask = mask.data.cpu().numpy()
    pred = pred.data.cpu().numpy()


    val_visual = [[i.data.cpu(), j, k] for i, j, k in zip(img, mask, pred)]
    val_img = []
    for imgs in val_visual:
        imgs = [restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                    else array2PIL_mask(i) for i in imgs]
        imgs = [i.convert('RGB') for i in imgs]
        imgs = [viz_transform(i) for i in imgs]
        val_img.extend(imgs)
    val_img = torch.stack(val_img, 0)
    val_img = torchvision.utils.make_grid(val_img.cpu(),nrow=val_img.size(0)//len(val_visual),padding=5)
    #val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)

        #frame.extend([i, j, k]) 
    #    viz_transform().
    #for i,m,p in zip(img,mask,pred):
    #    f = torch.tensor(build_h_frame(i,m,p))
    #    frame.extend(f)
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
    self.writer.add_scalar(f'{self.mode }/f1', 0, self.wrt_step)
  
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--data_root', '-r',
        type=str,
        required=False,
        #default='/home/tiago/workspace/dataset/learning',
        default='samples',
        help='Directory to get the trained model.'
    )



    root = '/home/tiago/workspace/dataset/learning'
    vineyard_plot = 'qtabaixo'
    sensor = 'x7'

    images,masks,preds= image_generator(10,10,3,50)
    f1_signal,epoch = signal_generator(50)
    loss_signal,epoch = signal_generator(50)
    writer = writer('results/test1')

    for image,mask,pred,f1,loss,epoch in zip(images,masks,preds,f1_signal,loss_signal,epoch):
        frame = add_image_tb(image,mask,pred)
        writer.add_f1(f1,epoch)
        writer.add_f1(loss,epoch)
        writer.add_image(frame,epoch)


    
    #img_frame = Image.fromarray(frame)
    #img_frame.save("test.png")
