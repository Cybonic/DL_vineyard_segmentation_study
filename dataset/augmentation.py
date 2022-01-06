'''


Original implementation:  https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py


https://medium.com/mlearning-ai/understanding-torchvision-functionalities-for-pytorch-part-2-transforms-886b60d5c23a

'''


import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import albumentations as A
import cv2

MAX_ANGLE = 180

class augment_rgb():
    split_idx = 0
    # https://datamahadev.com/performing-image-augmentation-using-pytorch/
    # https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e
    def __init__(self,max_angle = MAX_ANGLE):
        self.max_angle =max_angle
        
    
            
        self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        #A.GridDistortion(p=0.5),    
                        #A.RandomCrop(height=120, width=120, p=0.5),  
                        #A.Blur( blur_limit = (3, 7),always_apply=False, p=0.5),
                        A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                        A.ColorJitter (brightness=1, contrast=1, saturation=1, hue=0.1, always_apply=False, p=0.5),
                        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, 
                            scale_limit=0.2,
                            rotate_limit=(0, max_angle),
                            p=0.5)  
                    ], 
                    p=1
        )
                   
    def __call__(self, bands, mask, ago_indices):
        
        transformed = self.transform(image = bands,mask = mask)
        images = transformed['image']
        masks = transformed['mask']
        return(images,masks,[]) 