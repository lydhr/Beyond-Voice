import numpy as np
import matplotlib.pyplot as plt
import os, argparse

import torch
from torch.utils.data import DataLoader
from model.myDataset import MyDataset
from model.myModel import MyModel
from utils.myUtils import MyUtils
from itertools import combinations
import utils.parameters as params
from parsers.audioParser import AudioParser
import random

# ensure CUDA deterministic
if torch.cuda.is_available():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(0)
    
    mypath = "{}/".format(params.FOLDER_NAME)
    folders = getFolders(mypath, '_all')

    msg = runCrossSession([folders[0]], [folders[1]], load = False, generated = np.arange(-3, 3))
    MyUtils.copyFileWithTimestamp(params.FNAME_SAVED_MODEL, label = "01", TIME=False)

    predict([folders[1]])



def getFolders(mypath, keyword):
    folders = ["{}/{}".format(mypath, folder) for folder in os.listdir(mypath) if keyword in folder]
    folders.sort()
    print(folders)
    return folders


def runCrossSession(train_dirs, test_dirs, test_size = 1, load = True, generated = [0]):
    train_dataset = MyDataset(train_dirs, generated = generated)
    test_dataset = MyDataset(test_dirs).subset(1 - test_size, 1)
    print("cross session: train = {}, test = {}, train_dir = {} = , test_dir = {}".format(len(train_dataset), len(test_dataset), train_dirs, test_dirs))
    
    train_loader = getDataloader(train_dataset, shuffle = True)
    test_loader = getDataloader(test_dataset, shuffle = False)

    mymodel = MyModel()
    msg = mymodel.fit(train_loader, test_loader, VALIDATION = True, SAVE = True, ANCHOR_TRAIN_LOSS = False, LOAD_MODEL = load)
    return msg

def predict(test_dirs, test_size = 1, upperBound = None):
    test_dataset = MyDataset(test_dirs, upperBound = upperBound).subset(1 - test_size, 1)
    test_loader = getDataloader(test_dataset, shuffle = False)
    
    mymodel = MyModel()
    y_predict, loss = mymodel.predict(test_loader, SAVE = True)
    
    return y_predict, loss

def getDataloader(dataset, shuffle = True, batch_size = 128):
    '''
    def collate_fn(batch):
        data = torch.tensor([b['x'] for b in batch])
        label = torch.tensor([b['y'] for b in batch])
        return data, label
    '''
    dataloader = DataLoader(dataset, batch_size = batch_size, 
        shuffle = shuffle, num_workers = 0)#, collate_fn = collate_fn)
    return dataloader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-m', '--loadModel', type = str, default = None, help = "filename of load_model")
    args = arg_parser.parse_args()

    if args.loadModel: params.FNAME_LOAD_MODEL = args.loadModel
    
    main()
