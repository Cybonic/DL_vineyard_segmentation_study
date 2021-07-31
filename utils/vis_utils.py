import matplotlib.pyplot as plt 
from PIL import Image as im 
import numpy as np

class vis():
    def __init__(self):
        #self.fig, (self.ax1, self.ax2,self.ax3) = plt.subplots(1, 3)
        self.fig, self.ax1 = plt.subplots(1, 1)
        plt.ion()
        plt.show()
    
    def array2img(self, array):
        pred_mask  = array.squeeze()
        img_shape = [pred_mask.shape[0],pred_mask.shape[1],3]
        img_pred_mask = np.zeros(img_shape,dtype=np.int32)
        img_pred_mask[pred_mask==1]=1
        return(img_pred_mask)
    
   
        

    def show(self,img ,gt,pred):
        img_msk = self.array2img(gt)
        img_pred = self.array2img(pred)
        #if img.shape
        # img = img*255
    
        # img = img.astype(int)
        vis_img = np.hstack((img,img_msk,img_pred))
        self.ax1.imshow(vis_img)
        # self.ax2.imshow(img_msk)
        # self.ax3.imshow(img_pred)
        plt.draw()
        plt.pause(0.001)
        # plt.show()
    def close(self):
        plt.close()

