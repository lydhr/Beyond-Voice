import utils.parameters as params
from utils.myUtils import MyUtils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class MediaPipeParser(object):
    FNAME_JOINTS = params.FNAME_MEDIAPIPE_JOINTS
    FS = params.FS
    N_SAMPLE_SWEEP = params.N_SAMPLE_SWEEP
    N_SWEEP_INPUT = params.N_SWEEP_INPUT


    def __init__(self, foldername):
        self.foldername = foldername
        self.hands = np.array([]) #joints(n, 63)
        self.timestamps = np.array([]) #(n,)

        #load self.timestamps, self.hands
        filename = "{}/{}".format(foldername, self.FNAME_JOINTS)
        self.loadData(filename)


    def loadData(self, filename):
        data = MyUtils.loadPkl(filename) #n_frame x [curTime, hands_data]
        cnt, deleteIndices = self.loadHands(data[:, 1])
        self.loadTimestamps(data[:, 0], deleteIndices)
        assert self.hands.shape[0] == self.timestamps.shape[0]

        print("timestamps = {:.2f}~{:.2f}s,\t{}".format(self.timestamps[0], self.timestamps[-1], cnt))
        return


    def loadTimestamps(self, data, deleteIndices):
        """
            load self.timestamps
        """
        #delete some early starting samples
        self.timestamps = np.delete(data.astype('float32'), deleteIndices)
        return
        

    def loadHands(self, data):
        """
            load self.hands
            hands_data = n_hand x [is_right, joints 4+4+4+4+5 (thumb2pinky)] or
            hands_data = n_hand x [is_right, angles 3+4+4+4+4]
        """
        hands_data = []
        cnt_frozen, cnt_redundant, cnt_empty, cnt_invalid, deleteIndices = 0, 0, 0, 0, []#delete frames which has no hands or redundant hands
        IS_RIGHT = False if "left" in self.foldername else True
        lastHand = None
        for i, hands in enumerate(data): 
            valid_hands = [hand[1] for hand in hands if hand[0] is IS_RIGHT]#hand[0] = is_right
                
            if len(valid_hands) == 0:
                deleteIndices.append(i)
                if len(hands) == 0: cnt_empty += 1
                else: cnt_invalid += 1
            else:
                if len(valid_hands) > 1: cnt_redundant += 1
                hand = valid_hands[0]#select only the first
                if lastHand is not None and np.sum(np.absolute(lastHand-hand)) < 0: 
                    deleteIndices.append(i)
                    cnt_frozen += 1
                else:
                    hands_data.append(hand) #(21,3)
                lastHand = hand
        
        self.hands = np.array(hands_data, dtype = 'float32')
        #joints = joints - np.tile(joints[0, 0:3], 21)

        cnt = "{}, cnt_frozen = {}, cnt_redundant = {}, cnt_empty = {}, cnt_{} = {}".format(
            self.hands.shape, 
            cnt_frozen, cnt_redundant, cnt_empty, "left" if IS_RIGHT else "right", cnt_invalid)

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


    def getData(self):
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

