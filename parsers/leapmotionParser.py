import utils.parameters as params
from utils.myUtils import MyUtils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class LeapMotionParser(object):
    FNAME_JOINTS = params.FNAME_LEAPMOTION_JOINTS
    FNAME_ANGLES = params.FNAME_LEAPMOTION_ANGLES
    FS = params.FS
    N_SAMPLE_SWEEP = params.N_SAMPLE_SWEEP
    N_SWEEP_INPUT = params.N_SWEEP_INPUT


    def __init__(self, foldername, ANGLE = False):
        self.foldername = foldername
        self.hands = np.array([]) #joints(n, 63) or angles(n ,19)
        self.timestamps = np.array([]) #(n,)
        self.ANGLE = ANGLE

        #load self.timestamps, self.hands
        filename = "{}/{}".format(foldername, self.FNAME_ANGLES if ANGLE else self.FNAME_JOINTS)
        self.loadData(filename)


    def loadData(self, filename):
        data = MyUtils.loadPkl(filename) #n_frame x [curTime, hands_data]
        if self.ANGLE and data is None: return #angle pkl file not found
        cnt, deleteIndices = self.loadHands(data[:, 1])
        self.loadTimestamps(data[:, 0], deleteIndices)
        assert self.hands.shape[0] == self.timestamps.shape[0]

        print("timestamps = {} + {}s,\t{}".format(self.timestamps[0], self.timestamps[-1], cnt))
        return


    def loadTimestamps(self, data, deleteIndices):
        """
            load self.timestamps
        """
        #handle incompaitable old data collection    
        is_old_data = (data[0] > 10000)
        if not is_old_data:
            timestamps = data
        else:
            timestamps = data - data[0]
            print("subtract first timestamp")

        #delete some early starting samples
        self.timestamps = np.delete(timestamps.astype('float32'), deleteIndices)
        return
        

    def loadHands(self, data):
        """
            load self.hands
            hands_data = n_hand x [is_right, joints 4+4+4+4+5 (thumb2pinky)] or
            hands_data = n_hand x [is_right, angles 3+4+4+4+4]
        """
        hands_data = []
        cnt_redundant, cnt_empty, cnt_invalid, deleteIndices = 0, 0, 0, []#delete frames which has no hands or redundant hands

        IS_RIGHT = False if "left" in self.foldername else True
        for i, hands in enumerate(data): 
            if IS_RIGHT:
                valid_hands = [hand[1] for hand in hands if hand[0]]#hand[0] = is_right
            else:
                valid_hands = [hand[1] for hand in hands if not hand[0]]#hand[0] = is_right
                
            if len(valid_hands) == 0:
                deleteIndices.append(i)
                if len(hands) == 0: cnt_empty += 1
                else: cnt_invalid += 1
            else:
                if len(valid_hands) > 1: cnt_redundant += 1
                hand = valid_hands[0]#select only the first
                hands_data.append(sum(hand, [])) #dim-1
                """
                    finger[THUMB, INDEX, MIDDLE, RING] x joint[MCP, PIP, DIP, TIP]
                    + [PINKY] x [CARP, MCP, PIP, DIP, TIP]
                """

        self.hands = np.array(hands_data, dtype = 'float32')
        #joints = joints - np.tile(joints[0, 0:3], 21)

        cnt = "{} {}, cnt_redundant = {}, cnt_empty = {}, cnt_{} = {}".format(
            "angles" if self.ANGLE else "joints",
            self.hands.shape, 
            cnt_redundant, cnt_empty, "left" if IS_RIGHT else "right", cnt_invalid)

        return cnt, deleteIndices


    def getJoints(self, dim = 63):
        shp = self.hands.shape
        if dim == 21: #n x 21 x 3
            return self.hands
        elif dim == 42: # n x 42, i.e. xy
            return self.hands[:, :, :2].reshape(shp[:-2] + (-1,)) #only xy
        elif dim == 63:#n x 63, i.e. xyz
            return self.hands.reshape(shp[:-2] + (-1,))#xyz
        else:
            print("invalid dimension for joints", dim)
            return

    def getAngles(self, dim = 19):
        if dim == 19:
            return self.hands # n x 19
        elif dim == 15: #remove fingertips as FingerTrack paper did
            idx = np.arange(19)
            idx = np.delete(idx, [2, 6, 10, 14, 18])
            return self.hands[:, idx]
        else:
            print("invalid dimension for angles", dim)
            return


    def getData(self):
        if self.ANGLE:
            return self.getAngles()
        else:
            return self.getJoints()


    def showJointStats(self):
        joints = self.getJoints(dim = 21)
        for i, axis in zip(range(3), ['x', 'y', 'z']):
            positions = joints[:,:,i]
            print("- {}: {} ~ {},\tmean={}, std={}".format(axis, np.min(positions), np.max(positions),
                np.mean(positions), np.std(positions))
            )
        return

    def getWindowIdx(self):
        # timestamps2WindowIdx
        return (self.timestamps * self.FS // self.N_SAMPLE_SWEEP).astype(int)


    def normCoordinates(joints):
        joints = joints.reshape(joints.shape[0], 21, 3)
        COORDINATE_RANGE = params.COORDINATE_RANGE
        min_lim = np.array([ax[0] for ax in COORDINATE_RANGE])
        max_lim = np.array([ax[1] for ax in COORDINATE_RANGE])
        joints = (joints - min_lim)/(max_lim - min_lim)
        return joints.reshape(joints.shape[0], -1)
    
    def centerCoordinates(joints):
        startingTime = 100*3 #~3s
        joints = joints.reshape(joints.shape[0], 21, 3)

        metacarpal0 = joints[:, 0]
        startingMeta = np.mean(metacarpal0[:startingTime], axis = 0)

        joints = (joints - startingMeta).reshape(joints.shape[0], -1)
        return joints
