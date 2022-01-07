'''

https://gis.stackexchange.com/questions/57005/python-gdal-write-new-raster-using-projection-from-old


'''

import yaml
import os
import rasterio
import numpy as np
from tqdm import tqdm 
from PIL import Image 

NO_DATA = 9999


def parse_name(file):
    h,w = file.split('_')
    return(int(h),int(w))

def _rebuild_ortho_mask_from_mem(prediction,theight,twidth):
    raster_mask = np.zeros((theight,twidth),dtype=np.uint8)
    
    #bar = progressbar.ProgressBar(max_value=len(prediction)) 
    for i,(pred_mask_dict) in enumerate(prediction):
        # file name parser
        pred_mask = pred_mask_dict['mask']
        name = pred_mask_dict['name']
        
        #print(name)
        ph,pw = parse_name(name)

        if len(pred_mask.shape)> 2:
          pred_mask = pred_mask.squeeze()
        
        h,w = pred_mask.shape
        lh,hh,lw,hw = ph,ph+h,pw,pw+w
        raster_mask[lh:hh,lw:hw] = pred_mask
        
    return(raster_mask)

def rebuild_ortho(prediction_masks,dest_dir,file,save=True):
  if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

  file_name = os.path.join(dest_dir,file + '.png')
  ortho_mask = _rebuild_ortho_mask_from_mem(prediction_masks,23500,22400)
  if save:
    img_pred_mask = (ortho_mask*255).astype(np.uint8)
    image = Image.fromarray(img_pred_mask)
    image.convert('RGB')
    image.save(file_name)
  return(ortho_mask)



def save_sub_tiff(input,target_height,target_width,dest_path):
    '''
    Split "input" into subarrays with the size "target_height" and "target_width". 
    Save these subarrays to "dest_path"

    @param: input (Numpy) Array with dim [height,width,bands] 
    @param: target_height (int) height of the subarray
    @param: target_width (int) width of the subarray
    @return: number of images generated
    '''
   
    bands = input.shape[2]
    height= input.shape[0]
    width = input.shape[1]

    if bands>height or bands>width:
        print("[ERROR] band dim > than height or > than width")
        return(0)

    print("H:%d W:%d B:%d"%(height,width,bands))
    
    h_array = list(range(0,height,target_height))
    w_array = list(range(0,width,target_width))

    h_array_len = len(h_array)
    w_array_len = len(w_array)

    for i in tqdm(range(0,h_array_len-1)):
    # reset width counter 
        for j in range(0,w_array_len-1):
            # Sub-image name + absolute path 
            img_path = os.path.join(dest_path,"%05d_%05d.tiff"%(h_array[i],w_array[j]))
            # crop sub-image
            sub_array = input[h_array[i]:h_array[i+1],w_array[j]:w_array[j+1],:]
            # Save image
            tifffile.imsave(img_path, sub_array)

    n_images_saved = h_array_len*w_array_len
    return(n_images_saved)





