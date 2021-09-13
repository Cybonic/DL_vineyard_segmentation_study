
DATASET: https://drive.google.com/drive/folders/1iyrKndWzG9lOM-aVLs2gt3kgJUt0ychE?usp=sharing


The dataset contains data from three vineyards from central Portugal (i.e, two vineyards from Coimbra and one from Valdoeiro). The data was acquired with a UAS that had a multispectral and a high-definition cameras onboard. The acquired images were used to build orthomosaics and digital surface models from the respective sites. 
The dataset's structure: 

- paper data
    - > ESAC1
        - > Multispectral
            - > sub-images
                - 0000.npy
                - 0001.npy
            - > sub-masks
                - 0000.
        - > RGBX7 (HD)
            - > sub-images
            - > sub-masks
    - > ESAC2
        ...
    - > Valdoeiro 
- Mulstispectral(RGB, RE, NIR and Thermal) orthomosaics
- High-definition orthomosaics 
- Digital Surface Models 
- the dataset is divided in three sets (ESAC1, ESAC2 and Valdoeiro), which correspond to three vineyards 


# Computation setup:
Laptop: CUDA Version: 11.3 \
python 3.7 

# Install cuda 
    $ pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


# GDAL Installation on Ubuntu 

based on: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html


    $ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update

    $ sudo apt-get update
    $ sudo apt-get install gdal-bin
    $ sudo apt-get install libgdal-dev
    $ export CPLUS_INCLUDE_PATH=/usr/include/gdal
    $ export C_INCLUDE_PATH=/usr/include/gdal
    $ pip install GDAL == <GDAL VERSION FROM OGRINFO>


# Orthoseg pipeline 

To run the pipeline that is proposed in the [paper](https://arxiv.org/abs/2108.01200). 

run: 
    
    $ orthosegmentation.py 


# Pretrained models: Model and Dataset Dependent

The pretrained models with a specific dataset maintain the copyright of such dataset.

Link to pretrined models will be published soon 

# To-Do
- change multispectral data format from npy to image per band 




## Citations

If you use our framework, model, or predictions for any academic work, please cite the original [paper](https://arxiv.org/abs/2108.01200), and the [dataset](https://drive.google.com/drive/folders/1PeDqlXa-TISJcPGB2kaJ547LV5M3E_xU?usp=sharing).


## License

### Multispectral Vineyard Segmentation: A Deep Learning approach: MIT

Copyright (c) 2021 Tiago Barros, Pedro Conde, Gil Gonçalves, Cristiano Premebida, Miguel Monteiro, Carla S.S. Ferreira, Urbano J. Nunes.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
