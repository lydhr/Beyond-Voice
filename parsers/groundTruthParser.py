import numpy as np
import utils.parameters as params
from parsers.leapmotionParser import LeapMotionParser
from parsers.mediapipeParser import MediaPipeParser

class GroundTruthParser(object):
    gt = params.GROUND_TRUTH
    N_SWEEP_INPUT = params.N_SWEEP_INPUT

    @staticmethod
    def getValidWindowIdx(maxIdx, windowIdx, input_width = N_SWEEP_INPUT):
        valid = []
        indices2delete = []
        for i, right in enumerate(windowIdx):
            if input_width - 1 <= right <= maxIdx:
                valid.append(right)
            else:
                indices2delete.append(i)
        print("deleted windows: {}".format(indices2delete))
        return valid, indices2delete

    @staticmethod
    def getData(folder, gt = gt):
        if gt == 'leapmotion':
            parser = LeapMotionParser(folder, ANGLE = False)
            joints = parser.getData()
            windowIdx = parser.getWindowIdx()
            parser = LeapMotionParser(folder, ANGLE = True)
            angles = parser.getData()
            return joints, angles, windowIdx

        elif gt == 'mediapipe':
            parser = MediaPipeParser(folder)
            joints = parser.getData()
            windowIdx = parser.getWindowIdx()
            angles = np.array([])#empty placeholder to make myDataset consistent
            return joints, angles, windowIdx
        else:
            print("wrong ground truth name", gt)
            exit(0)


    @staticmethod
    def getVisData(folder, gt = gt):
        if gt == 'leapmotion':
            parser = LeapMotionParser(folder)
        elif gt == 'mediapipe':
            parser = MediaPipeParser(folder)
        else:
            print("wrong ground truth name", gt)
            exit(0)

        joints = parser.getJoints(dim = 21)
        timestamps = parser.timestamps
        windowIdx = parser.getWindowIdx()
        parser.showJointStats()

        return joints, timestamps, windowIdx
