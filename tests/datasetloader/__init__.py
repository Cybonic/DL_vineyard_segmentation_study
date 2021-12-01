import pathlib
import os
import sys 
path  = os.path.dirname(pathlib.Path(__file__).parent.parent.absolute())
paths = os.path.join(path,'dataset')

sys.path.append(path)
sys.path.append(paths)