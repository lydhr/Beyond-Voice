"""
https://stackoverflow.com/questions/67278053/plot-a-3d-pose-skeleton-data-in-python-from-numerical-dataset
"""
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

# sys.path.append('yourpath2folder')
import utils.parameters as params
from utils.myUtils import MyUtils
from parsers.groundTruthParser import GroundTruthParser

class Visualizer(object):
    CHAINS_INDICES = params.CHAINS_INDICES
    COORDINATE_RANGE = params.COORDINATE_RANGE
    PLOT_STEP = 10 #set a smaller number if your machine is fast

    def __init__(self, groundTruthFname, predictionFname):
        if groundTruthFname: assert params.GROUND_TRUTH in groundTruthFname
        if predictionFname: assert "prediction" in predictionFname
        self.groundTruthFname = groundTruthFname
        self.predictionFname = predictionFname


    @staticmethod
    def get_chains(dots, chains_indices = CHAINS_INDICES):
        """
            :params dots np.ndarray: shape == (n_dots, 3)
            :params chains_indices tp.List[tp.List[int]]:
                indexes of points forming a continuous chain, 
                example of chain: [MCP, PIP, DIP, TIP]
        """
        chains = []
        for chain_indices in chains_indices:
            chains.append(dots[chain_indices])
        return chains

    @staticmethod
    def subplot_nodes(dots, ax, GREY = False):
        """
            :params dots np.ndarray: shape == (n_dots, 3)
                color = A sequence of n numbers to be mapped to colors using *cmap* and *norm*
        """
        colors = "grey" if GREY else dots[:, -1]
        ax.scatter3D(*dots.T, c = colors)


    @staticmethod
    def subplot_bones(chains, ax, GREY = False):
        """
            :params chains tp.List[np.ndarray]: shape == (n_chain x n_dots_per_chain x 3)
        """
        for chain in chains: 
            if GREY: ax.plot(*chain.T, color = "grey")
            else: ax.plot(*chain.T)

    @staticmethod
    def resetPlot(ax, t):
        ax.cla()
        # ax.view_init(0, -90)
        ax.set_xlim(*Visualizer.COORDINATE_RANGE[0])
        ax.set_ylim(*Visualizer.COORDINATE_RANGE[1])
        ax.set_zlim(*Visualizer.COORDINATE_RANGE[2])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_zlim(ax.get_zlim()[::-1])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_title("t = {:.2f}s".format(t))

    @staticmethod
    def subPlot(ax, dots, timestamp, RESET = True, GREY = False):
        if RESET:
            Visualizer.resetPlot(ax, timestamp)
        
        Visualizer.subplot_nodes(dots, ax, GREY = GREY)
        Visualizer.subplot_bones(Visualizer.get_chains(dots), ax, GREY = GREY)

    @staticmethod
    def plotOneFrame(ax, dots):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        Visualizer.subPlot(ax, dots, 0) #dummy timestamp = 0        
        plt.show(block=True) # block=True lets the window stay open at the end of the animation.

    def on_key(self, event):
        exit()

    def visualizeOne(self, joints, timestamps, step = PLOT_STEP):
        """
            Plot the skeletons of one file
        """
        assert len(joints) == len(timestamps)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for i, dots in enumerate(joints):
            if i%step != 0: continue
            
            Visualizer.subPlot(ax, dots, timestamps[i])

            plt.pause(0.0001)
            fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show(block=True) # block=True lets the window stay open at the end of the animation.

    def visualizeTwo(self, joints0, joints1, timestamps, step = PLOT_STEP):
        """
            Plot the skeletons of two file, joints0 is grey
        """
        assert len(joints0) == len(joints1) == len(timestamps)
        print("GREY hand is {}, cyan hand is {}".format(self.groundTruthFname, self.predictionFname))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(len(joints0)):
            if i%step != 0: continue

            Visualizer.subPlot(ax, joints0[i], timestamps[i], GREY = True)
            Visualizer.subPlot(ax, joints1[i], None, RESET = False, GREY = False)

            plt.pause(0.0001)

            fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show(block=True) # block=True lets the window stay open at the end of the animation.
    
    @staticmethod
    def loadGroundTruth(filename, deleteIndices = False):
        folder = "/".join(filename.split("/")[:-1])
        joints, timestamps, windowIdx = GroundTruthParser.getVisData(folder)

        if deleteIndices:
            maxWindowIdx = float('inf') # should be audio corrData.shape[2] - 1, but unkown here since we do not load audio
            _, indices2delete = GroundTruthParser.getValidWindowIdx(maxWindowIdx, windowIdx)
            joints = np.delete(joints, indices2delete, axis = 0)
            timestamps = np.delete(timestamps, indices2delete, axis = 0)
            assert len(joints) - deleteIndices < 3 #only unmatchable g and p cause large diff
            print("indices2delete front = {}, back = {}".format(len(indices2delete), len(joints) - deleteIndices))
            joints, timestamps = joints[:deleteIndices], timestamps[:deleteIndices]#finally, cut off the ones exceed the maxWindowIdx
        return joints, timestamps

    @staticmethod
    def loadPrediction(filename):
        """
            return joints [n, 21, 3]
        """
        joints = MyUtils.loadPkl(filename)
        joints = joints.reshape(joints.shape[0], 21, 3)

        #show joints status
        for i, axis in zip(range(3), ['x', 'y', 'z']):
            positions = joints[:,:,i]
            print("- {}: {} ~ {},\tmean={}, std={}".format(axis, np.min(positions), np.max(positions),
                np.mean(positions), np.std(positions))
            )
        return joints

    def postProcessing(self, joints):
        """
            joints: n_sample x 21 x 3
        """
        windowSize = 5 # 5/90 ~= 0.05s
        palm = 16 #index 16 in [0:21]
        for i in range(windowSize, joints.shape[0]):
            anchor = np.average(joints[i-windowSize:i, palm], axis = 0) # (5, 1, 3) -> (1, 3)
            diff = joints[i][palm] - anchor 
            joints[i] = [keypoint - diff for keypoint in joints[i]] #21 keypoints
        return joints

    def cutFromStartTime(self, percentage, *arrs):
        if percentage > 0:
            start = int(percentage*len(arrs[0]))
            arrs = [arr[start:] for arr in arrs]
            print("start from {:%}".format(percentage))
        return arrs

    def getDummyTimesteps(self, n):
        timestamps = np.linspace(0, n/params.ESTIMATED_GT_FS, n)#dummy timestamps
        print("set dummy timestamps = {:.2f}~{:.2f}".format(timestamps[0], timestamps[-1]))
        return timestamps
            
    def run(self, *filenames, startTimePercent = 0): 
        if not (self.groundTruthFname or self.predictionFname):
            print("no files")

        else:
            joints, timestamps = [np.array([]), np.array([])], None
            
            if self.predictionFname:
                joints[1] = Visualizer.loadPrediction(self.predictionFname)
            if self.groundTruthFname:
                joints[0], timestamps = Visualizer.loadGroundTruth(self.groundTruthFname, deleteIndices = len(joints[1])) # len(prediction) <= len(groundTruth) because of getSlicedCorr()
            if timestamps is None: timestamps = self.getDummyTimesteps(len(joints[1]))

            print("loaded joints = {} timestamps = {} from {}, {}, ".format([j.shape for j in joints], timestamps.shape, self.groundTruthFname, self.predictionFname))
            for j in [j for j in joints if len(j) > 0]: assert len(j) == len(timestamps)

            joints[0], joints[1], timestamps = self.cutFromStartTime(startTimePercent, joints[0], joints[1], timestamps)
            # joints = self.postProcessing(joints)
            
            if len(joints[0]) and len(joints[1]):
                self.visualizeTwo(joints[0], joints[1], timestamps)
            else:
                self.visualizeOne(joints[0], timestamps) if len(joints[0]) else self.visualizeOne(joints[1], timestamps)

        return
