
import argparse
from dataset.learning_dataset import dataset_loader,greenAIDataStruct


def test_data_structure(root,vineyard_plot,sensor):


    structure = greenAIDataStruct(root,vineyard_plot,sensor)


    return(False)

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



    root = '/home/tiago/workspace/dataset/learning'
    vineyard_plot = 'qtabaixo'
    sensor = 'x7'

    test_data_structure(root,vineyard_plot,sensor)