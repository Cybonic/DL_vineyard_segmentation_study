TO-DO:
DATASET: https://drive.google.com/drive/folders/1PeDqlXa-TISJcPGB2kaJ547LV5M3E_xU?usp=sharing

Learning Pipeline: 
[] add a standard logging framework (like writer) 
[] add a dataset fraction feature to run small experiments  
[x] color mapping score test
[x] band test: run through all combination while training on the same dataset  

 
Production Pipeline
[] orthoimage splitting: when "use_flag" == False then load  existing images, instead of creating new ones
[] plotting results
[] pretrained model load
[] rebuilding function 
[x] plot image prediction and GT
[x] color mapper (map refraction to color )
[x] create net_eval function
[x] solve the dataset loader splitting mechanism 
[x] fixe the backprop bug: possible source is the mismatch of classes

Computation setup:
Laptop: CUDA Version: 11.3 
python 3.7 

Install cuda 
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

Data Augmentation: 
https://towardsdatascience.com/how-to-implement-augmentations-for-multispectral-satellite-images-segmentation-using-fastai-v2-and-ea3965736d1
