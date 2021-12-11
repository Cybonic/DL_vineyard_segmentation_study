'''


Original implementation:  https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py


https://medium.com/mlearning-ai/understanding-torchvision-functionalities-for-pytorch-part-2-transforms-886b60d5c23a

'''


import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class ToTensor:
    def __init__(self):
        self.tensor = T.ToTensor()
        #self.tensor = T.ToPILImage()

    def __call__(self,image,target):
        return(self.tensor(image),self.tensor(target))

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomRotate:
    def __init__(self, min_angle, max_angle=None):
        self.min_angle = min_angle
        if max_angle is None:
            max_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image, target):
        angle = random.randint(self.min_angle, self.max_angle)
        rotated_image  = F.rotate(image,angle)
        rotated_target = F.rotate(target,angle)
        return(rotated_image,rotated_target)


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop:
    def __init__(self, size):
        self.size = size


    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class ToPIL:
    def __call__(self, image, target):
        image = F.to_pil_image(image)
        return(image, target)

class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Adjust_Saturation:
    def __init__(self,saturation_value=3):
        self.saturation_value = saturation_value
    def __call__(self,image,target):
        image = F.adjust_saturation(image,self.saturation_value)
        return(image,target)

class Adjust_Brightness:
    def __init__(self,brightness_factor = 3):
        self.brightness_factor = brightness_factor
        array = np.array(range(0,self.brightness_factor))
    def __call__(self,image,target):
    
        image = F.adjust_brightness(image,self.brightness_factor)
        return(image,target)

class Equalize:
    def __init__(self):
        pass
    def __call__(self,image,target):
        image = image.to(torch.uint8)
        #i#mage = F.to_tensor(image)
        image = F.equalize(image)
        #image = F.pil_to_tensor(image)

        return(image,target)