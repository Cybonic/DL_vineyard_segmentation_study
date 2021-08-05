
DATASET: https://drive.google.com/drive/folders/1PeDqlXa-TISJcPGB2kaJ547LV5M3E_xU?usp=sharing


The dataset contains data from three vineyards from central Portugal (i.e, two vineyards from Coimbra and one from Valdoeiro). The data was acquired with a UAS that had a multispectral and a high-definition cameras onboard. The acquired images were used to build orthomosaics and digital surface models from the respective sites. 
The dataset's structure: 
- Drone:
    -> paper data
        -> ESAC1
            -> Multispectral
                -> sub-images
                    0000.npy
                    0001.npy
                -> sub-masks
                    0000.
            -> HD
                -> sub-images
                -> sub-masks
        -> ESAC2
            ...
        -> Valdoeiro 
            ...
    -> orthomosaics
        -> 
    -> raw data
        -> Coimbra
        -> Valdoeiro 
- Mulstispectral(RGB, RE, NIR and Thermal) orthomosaics
- High-definition orthomosaics 
- Digital Surface Models 
- the dataset is divided in three sets (ESAC1, ESAC2 and Valdoeiro), which correspond to three vineyards 
- 
- 

Computation setup:
Laptop: CUDA Version: 11.3 
python 3.7 

Install cuda 
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

