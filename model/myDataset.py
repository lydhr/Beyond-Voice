import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

import os
import utils.parameters as params
from parsers.audioParser import AudioParser
from parsers.groundTruthParser import GroundTruthParser
from utils.myUtils import MyUtils

class MyDataset(Dataset):
    def __init__(self, data_dirs, upperBound = None, generated = [0]):
        '''
            :params data_dirs list(): list of dir/xy.pkl to load
        '''
        self.upperBound = upperBound
        self.generated = generated
        self.angle = ("Angle" in params.NET)
        self.x, self.y = np.array([]), np.array([])#((n, 7, 256, 50), (n, 19/63))
        self.loadXY(data_dirs)#self.x, self.y

    def loadXY(self, data_dirs):
        if len(data_dirs) == 1:
            self.x, self.y = self.getXY(data_dirs[0])
        else:
            xs, ys = [], []
            for data_dir in data_dirs:
                x, y = self.getXY(data_dir)
                xs.append(x)
                ys.append(y)
            assert len(xs) == len(ys)
            if len(xs) > 0:
                self.x, self.y = np.vstack(tuple(xs)), np.vstack(tuple(ys))
        self.shape()
        return


    def subset(self, start, end):
        '''
            :params start, persentage of the starting index
            :params end, persentage of the ending index
        '''
        if start == 0 and end == 1: return self
        start, end = int(start * len(self)), int(end * len(self))
        self.x, self.y = self.x[start:end], self.y[start:end]
        print("take a subset of {} ~ {}".format(start, end))
        self.shape()
        return self
    def append(self, myDataset):
        self.x = np.vstack(tuple([self.x, myDataset.x]))
        self.y = np.vstack(tuple([self.y, myDataset.y]))
        self.shape()
        return self

    def shape(self):
        d_shape = (self.x.shape, self.y.shape)
        print("dataset (x.shape, y.shape) = {}".format(d_shape))
        return d_shape
        
    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        #return sample
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


    def getXY(self, folder):
        fname = self.getFilename(folder)
        if not os.path.exists(fname):
            self.dumpXY(folder)
        
        data = MyUtils.loadPkl(fname)
        corrData, up, windowIdx = data['corrData'], data['upperBound'], data['windowIdx']
        y = data['angles'] if self.angle else data['joints']
        del data
        print("[loaded corrData] = {} (up = {}), windowIdx = {}, labels = {} from {}".format(corrData.shape, up, windowIdx.shape, y.shape, fname))
        if self.angle and not y.any():
            print("need to deserialize the angle in LeapMotion/py2 and delete the xy.pkl")
            exit()
        
        #cut the starting time
        if self.upperBound is not None:
            up = self.upperBound
        
        x0, indices2delete = AudioParser.getSlicedCorr(corrData, windowIdx)
        y0 = np.delete(y, indices2delete, axis = 0)
        # y = LeapMotionParser.centerCoordinates(y)
        
        x, y = self.getGeneratedData(x0, y0, up)
        return x, y


    def getGeneratedData(self, x0, y0, up):
        xs, ys = [], []
        for n_shift in self.generated:
            xx = AudioParser.removeInferenceBound(x0, up + n_shift)
            if self.angle:
                yy = y0
            else:
                yy = y0 + [0, 3.57292 * n_shift, 0] * 21
            assert xx.shape[0] == yy.shape[0]
            xs.append(xx)
            ys.append(yy)
        x, y = np.vstack(tuple(xs)), np.vstack(tuple(ys))

        #otherwise, model.double()
        if type(x) != np.float32: x = x.astype(np.float32)
        if type(y) != np.float32: y = y.astype(np.float32)
        
        return x, y
        

    def dumpXY(self, folder):
        fname = self.getFilename(folder)

        joints, angles, windowIdx = GroundTruthParser.getData(folder)
        audioParser = AudioParser(folder, draw = False)
        corrData = audioParser.corrData#dump without cutting spectrum
        up = audioParser.getInferencePoint()

        MyUtils.dumpPkl({'corrData': corrData.astype(np.float32), 
            'upperBound': up, 
            'windowIdx': windowIdx, 
            'joints': joints.astype(np.float32),
            'angles': angles.astype(np.float32)}, 
            fname)
        print("saved corrData = {} (up = {}), windowIdx = {}, joints = {}, angles = {} in {}".format(corrData.shape, up, windowIdx.shape, joints.shape, angles.shape, fname))
        return

    def getFilename(self, folder):
        return "{}/{}".format(folder, params.FNAME_XY)
