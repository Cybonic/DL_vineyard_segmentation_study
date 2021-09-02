
DATASET: https://drive.google.com/drive/folders/1PeDqlXa-TISJcPGB2kaJ547LV5M3E_xU?usp=sharing


The dataset contains data from three vineyards from central Portugal (i.e, two vineyards from Coimbra and one from Valdoeiro). The data was acquired with a UAS that had a multispectral and a high-definition cameras onboard. The acquired images were used to build orthomosaics and digital surface models from the respective sites. 
The dataset's structure: 
- Drone:
    |
    - > paper data
        - > ESAC1
            - > Multispectral
                - > sub-images
                    - 0000.npy
                    - 0001.npy
                - > sub-masks
                    - 0000.
            - > HD
                - > sub-images
                - > sub-masks
        - > ESAC2
            ...
        - > Valdoeiro 
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


Computation setup:
Laptop: CUDA Version: 11.3 
python 3.7 

Install cuda 
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

## TO-DO
- change the root path code accordingly to the new dataset structure
- add dataset structure to READ-ME

## License

### Multispectral Vineyard Segmentation: A Deep Learning approach: MIT

Copyright (c) 2021 Tiago Barros, Pedro Conde, Gil Gonçalves, Cristiano Premebida, Miguel Monteiro, Carla S.S. Ferreira, Urbano J. Nunes.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Pretrained models: Model and Dataset Dependent

The pretrained models with a specific dataset maintain the copyright of such dataset.

## Citations

If you use our framework, model, or predictions for any academic work, please cite the original [paper](https://arxiv.org/abs/2108.01200), and the [dataset](https://drive.google.com/drive/folders/1PeDqlXa-TISJcPGB2kaJ547LV5M3E_xU?usp=sharing).
