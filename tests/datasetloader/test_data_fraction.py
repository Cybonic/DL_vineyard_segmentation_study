
import argparse
import pathlib
import os
import sys 
import torch 
from tqdm import tqdm
package_root  = os.path.dirname(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(package_root)
dataset_path =os.path.join(package_root,'dataset')
sys.path.append(dataset_path)

from dataset.learning_dataset import dataset_wrapper
import numpy as np 


def generate_bounderies(max,min):
  return({'img':np.array([min,max]),'mask':np.array([min,max])})


def get_bound_values(torch_array) -> dict:
  return({'min': torch.min(torch_array).item() ,'max':torch.max(torch_array).item()})

def get_data_bounderies(dataset):

  edge_array_img = []
  edge_array_mask = []

  for sample in tqdm(dataset):
    img = sample['bands']
    mask = sample['mask']

    edge_img_values = get_bound_values(img)
    mask_edge_values = get_bound_values(mask)
    edge_array_img.append([edge_img_values['min'],edge_img_values['max']])
    edge_array_mask.append([mask_edge_values['min'],mask_edge_values['max']])


  return({'img': np.array([np.min(np.array(edge_array_img)),np.max(np.array(edge_array_img))]),\
  'mask': np.array([np.min(np.array(edge_array_mask)),np.max(np.array(edge_array_mask))])}
  )
  
def test_data_bounderies(dataset,lower_bound=0,upper_bound=1):
  
  edge_cases = get_data_bounderies(dataset)
  test_conditions = []
  for name, array in edge_cases.items():
    if (array<lower_bound).any() or (array > upper_bound).any():
      print("[Test Failed] %s| -> min %lf max %lf"%(name,array[0],array[1]))
      test_conditions.append(False)
    else:
      print("[Test passed] %s| -> min %lf max %lf"%(name,array[0],array[1])) 
      test_conditions.append(True)
  
  #sum_bool_cond = np.array([int(c==True) for c in test_conditions]).sum()
  return((np.array(test_conditions)==True).all())

 

if __name__ == '__main__':

  
  root = '/home/tiago/desktop_home/workspace/dataset/learning/'
  sensor = 'x7'
  bands = ['nir']
  augment = False
  set = ['valdoeiro','esac']

  dataset= dataset_wrapper(
                        root,
                        set,
                        sensor, 
                        bands = {'R':True,'G':True,'B':True}, 
                        agro_index = {'NDVI':False}, 
                        transform = None, 
                        path_type='global',
                        fraction = 0.1)
  
  print("Sets:"+f'{dataset.plot}')
  print("Sensor:" + dataset.sensor)
  print("root:" + dataset.root)
  print("# samples: %d"%(len(dataset)))

  print("Test: %d"%(test_data_bounderies(dataset,0,1)))
