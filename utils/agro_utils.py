import numpy as np

def NDVI(nir,red):
    '''
    # https://eos.com/make-an-analysis/ndvi/ 
    Inputs: nxm numpy arrays
        NIR – reflection in the near-infrared spectrum
        RED – reflection in the red range of the spectrum
    '''
    num = nir-red 
    dom = nir+red 
    ndvi = np.divide(num,dom)
    ndvi[np.isnan(ndvi)]=0 # Clean array with nan
    
    return(ndvi)