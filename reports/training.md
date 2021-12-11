# Training Proceedure

## Obervations
- [OB_0002] GPU has not enough memory to train and validate Unet with BATCH_SIZE= 10
- [OB_0001] Cross training: Training set T2 as the lowerest performance in all networks. 
    - trainingset ['valdoeiro',qtabaixo], testset [esac]: esac has in general more green color then the other sets.; 


## TESTS:


# Savage Mode
The savage mode is a feature that loads all data at the start to mamory, which permits to run faster both in training and testing/validation. 


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




