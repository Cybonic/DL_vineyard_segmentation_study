
import argparse
from dataset import dataset_loader



  dataset= dataset_loader(root = root, # path to dataset 
                          sensor = sensor, # [Multispectral,RGBX7]
                          bands = bands, # [R,G,B,NIR,RE,Thermal]
                          agro_index= agro_index,# [NDVI]
                          augment = augment, #[True, False]
                          trainset = trainset, #[esac1,esca2,valdoeiro] 
                          testset = testset, #[esac1,esca2,valdoeiro] 
                          batch_size = batch_size ,
                          shuffle = shuffle ,
                          workers = workers,
                          fraction = {'train':fraction,'test':fraction} # [0,1]
                          )

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--data_root', '-r',
      type=str,
      required=False,
      #default='/home/tiago/workspace/dataset/learning',
      default='samples',
      help='Directory to get the trained model.'
  )