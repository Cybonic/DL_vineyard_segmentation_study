'''
Author: Tiago 


Reference: 
 - Several pytorch implementations of segmentation Networks  https://github.com/meetshah1995/pytorch-semseg

'''


from networks import unet_bn 
#from networks import MFNet
from networks import segnet
from networks import modsegnet
#from networks.fcn import fcn32s, fcn8s
import torch.nn.init as init

import torch.nn as nn
import torch
import shutil
import os
import numpy as np


class OrthoSeg(nn.Module):
    def __init__(self,param,image_shape,channels,drop_rate = 0.5):
        super(OrthoSeg, self).__init__()
        
        #self.use_pretrained_model = param['pretrained']['use']
        #self.pretrained_path = param['pretrained']['path']
        #self.pretrained_file = param['pretrained']['file']
        
        # image_shape = param['input_shape']
        shape = np.array([image_shape[0],image_shape[1], channels])
        seg_name = param['model']

        if seg_name == 'segnet':
            self.model = segnet.SegNet(num_classes=1, n_init_features=channels,drop_rate= drop_rate)
        elif seg_name == 'unet_bn':
            self.model = unet_bn.UNET(out_channels=1, in_channels=channels) # UNet has no dropout
        elif seg_name == 'unet':
            self.model = unet.UNet(num_classes=1, n_init_features=channels) # UNet has no dropout
        elif seg_name == 'modsegnet':
            self.model = modsegnet.ModSegNet(num_classes=1, n_init_features=channels,drop_rate= drop_rate)
        elif seg_name == 'fcn32':
            self.model = fcn8s.FCN8s(n_class=1,in_channels=channels)
        elif seg_name == 'mfnet':
            self.model = MFNet.MFNet(num_classes=1,in_channels_1=channels-1,in_channels_2=1)
        else:
            raise NameError 

        #self.apply_init_weights()

    
    def forward(self,x):
        # x = self.norm(x)
        y = self.model(x)
        return(y)

    def load_pretrained(self,path):
        self.model.load_state_dict(torch.load(path))

    def apply_init_weights(self):
        
        #  https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/prediction_flow/pytorch/utils.py
        # model = self.model

        self.model.apply(weights_init)


        
    

    def save(self,name):
        '''

        Reference: 
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
        '''

        
        model_name = os.path.join(self.pretrained_path,name)
        torch.save(self.model, model_name)

def weights_init(model):

    gain = init.calculate_gain('relu')

    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            #init.normal_(model.bias.data)
            init.xavier_uniform_(model.bias.data,gain)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            #init.normal_(model.weight.data, mean=1, std=0.02)
            init.kaiming_uniform_(model.weight.data, a=0, mode='fan_in', nonlinearity='prelu')
            #init.xavier_uniform_(model.weight.data,gain)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=0, std=1)
            #init.kaiming_uniform_(model.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            #init.xavier_uniform_(model.weight.data,gain)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            #init.normal_(model.weight.data, mean=1, std=0.02)
            init.kaiming_uniform_(model.weight.data, a=0, mode='fan_in', nonlinearity='prelu')
            #init.xavier_uniform_(model.weight.data,gain)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Conv2d):
        if model.weight is not None:
            #init.normal_(model.weight.data, mean=1, std=0.02)
            #init.xavier_uniform_(model.weight.data,gain)
            init.kaiming_uniform_(model.weight.data, a=0, mode='fan_in', nonlinearity='prelu')
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    else: 
        pass 