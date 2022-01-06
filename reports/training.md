# Training Proceedure


## Obervations
- [0B_0004] Best score obtained with Batchsize ==1
- [OB_0003] Using rotation in augmentation causes image transistion problems
- [OB_0002] GPU has not enough memory to train and validate Unet with BATCH_SIZE= 10
- [OB_0001] Cross training: Training set T2 as the lowerest performance in all networks. 
    - trainingset ['valdoeiro',qtabaixo], testset [esac]: esac has in general more green color then the other sets.; 

# 

## TESTS:

#
# Savage Mode
The savage mode is a feature that loads all data at the start to mamory, which permits to run faster both in training and testing/validation. 


# Augmentation
[OB_0001] To mutch augmentations decreasy generalization of the networks
## RGB-HD 
[Training] = esac & valsoeiro

[Test]     = Qtabaixo

[Network]  = Segnet 

[batch_size] = 5 

[learning_rate] = 0.0001

0. Without augmentation 
     ![image](fig/study_aug_0.png)
1. Onlye with spatial transformations:
     - HorizontalFlip(p=0.5);
     - ShiftScaleRotate

     ![image](fig/study_aug_1.png)
2. only with ColorJitter

     ![image](fig/study_aug_2.png)
3. only with CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)

     ![image](fig/study_aug_3.png)
4. only with Blur( blur_limit = (3, 7),always_apply=False, p=0.5),

     ![image](fig/study_aug_4.png)
5. using color chaning operation:
     - Blur( blur_limit = (3, 7),always_apply=False, p=0.5),
     - CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
     - ColorJitter (brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, always_apply=False, p=0.5)

     ![image](fig/study_aug_5.png)

6.  using color chaning operation:
     - Blur( blur_limit = (3, 7),always_apply=False, p=0.5),
     - CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
     - ColorJitter (brightness=1, contrast=1, saturation=1, hue=1, always_apply=False, p=0.5),

     ![image](fig/study_aug_5.1.png)
7. using "apply alwase == True for all layers:

     

 
# Preprocessing
There is an issue with the normalization of the data. 
## Without Preprocessing 
1. X7 datasets: 
     - without augmentation  are working correctly: ie colors are correctl
     - with augmentation are working correctly
2. altum (RGB):
     - without augmentation: images are generaly very dark
     - with augmentation: augmentation is wrking correctly: images continue dark 
3. altum (NIR):
     - without augmentation: images are in grayscale with good 
     - with augmentation: augmentation is wrking correctly: images as before
## With ALtum pixel mapping
1.  altum (RGB):
    - With Augmentation: Qtabaixo and esac have the same coloring, valdoeiro on the other hand, is dark.
    [Solved] by adding tif2pixel function to valdoeiro image loading code
    - Without Augmentatio: same as before

2. altum(NIR):
     - Without Augmentatio: Images are very clear/white. 
     - With Augmentation: Same as before

3. Changed Value to (1/4)*255 -> (1/2)) * 255: better color space 

# Weight decay study: 
Networks are overfeating the training data in particlar the NIR band. 
LR = 0.0001
1. WD: 



# Report 
Using the parameters of the original paper:
- MAX_EPOCH 50
- 
## LR Study: NIR Model 
### Unet
This study was perfromed to asses the best LR.
This study suggests that the best LR is 0.0001

![Image](fig/unet_nir_lr_study_overfeating.png)

### Segnet
This study was perfromed on the segnet using as input only the NIR band. 
This study suggests that the best LR is 0.0001
![Image](fig/segnet_nir_lr_study_overfeating.png)


### Modsegnet
This study suggests that the best LR is 0.001
![Image](fig/modsegnet.png)




