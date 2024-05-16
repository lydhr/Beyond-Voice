"""
    load prediction from file; 
    for each frame, correlate with the pre-defined skeleton of activation gesture;
    print/plot the arr of correlation
"""


import numpy as np
import os, sys, argparse
import matplotlib.pyplot as plt

sys.path.append('/Users/lynn/Documents/workspace/dolposWorkspace/DolPosGit/analysis')
from utils.myUtils import MyUtils
from utils.vis import Visualizer
import itertools

class Activation(object):
    ANCHOR = np.array([[ -9.113112,  144.84204,    64.62693  ],
                [ -28.385408,  144.09674,    29.525267 ],
                [ -38.219517,  140.92705,     4.3686113],
                [ -47.453403,  141.48453,   -11.405985 ],
                [   5.7260356, 151.0919,     -1.0545366],
                [   6.7699594, 142.67708,   -34.132328 ],
                [   6.180135,  134.30156,   -51.398888 ],
                [   5.2095985, 126.43846,   -62.487778 ],
                [  23.226616,  148.36641,     3.2000184],
                [  18.354212,  119.69075,   -17.807592 ],
                [  12.570728,  102.996056,  -19.441698 ],
                [   8.479497,   93.487785,  -17.679966 ],
                [  38.863377,  144.39339,    12.600001 ],
                [  35.001377,  117.37461,    -6.604682 ],
                [  28.689426,  101.53369,    -8.899474 ],
                [  23.89571,    92.58487,    -8.261904 ],
                [  29.897055,  146.08305,    63.878426 ],
                [  51.461605,  137.24718,    22.639675 ],
                [  63.33033,   126.65933,    -0.7370185],
                [  68.21456,   119.01007,   -13.460281 ],
                [  71.41175,   110.9745,    -24.2127   ]])
    
    def loadPrediction(filename):
        """
            return joints [n, 21, 3]
        """
        joints = MyUtils.loadPkl(filename)
        joints = joints.reshape(joints.shape[0], 21, 3)
        #n_sample = 12k/2min, so get rid of first and last ~5s
        joints = joints[500:]
        joints = joints[:-500]

        #show joints status
        for i, axis in zip(range(3), ['x', 'y', 'z']):
            positions = joints[:,:,i]
            print("- {}: {} ~ {},\tmean={}, std={}".format(axis, np.min(positions), np.max(positions),
                np.mean(positions), np.std(positions)))
        return joints

    def getSimilarityScores(hands, anchor = ANCHOR):
        return [Activation.getSimilarityScore(anchor, h) for h in hands]

    def getSimilarityScore(hand0, hand1):
        """
            hand: [21, 3]
        """
        hand0, hand1 = Activation.preprocess(hand0), Activation.preprocess(hand1)
        return np.sum(abs(hand0 - hand1))/21 #MAE, normed by palm size

    def preprocess(hand, palm = 16, index_MCP = 4):
        # print(hand[palm], hand[index_MCP])
        hand -= hand[palm] #recenter
        R = Activation.getRotationMatrix(hand[index_MCP], [0, 0, 1])
        hand = np.matmul(hand, R.transpose()) #([3, 3] x [21, 3]^T)^T
        # print(Activation.getMagnitude(hand[index_MCP]),  hand[palm], hand[index_MCP])
        return hand/Activation.getMagnitude(hand[index_MCP])#normalize by palm size

    def getMagnitude(M):
        return np.sqrt(sum([pow(element, 2) for element in M]))

    def getRotationMatrix(A, B):
        au, bu = A/Activation.getMagnitude(A), B/Activation.getMagnitude(B)
        R = np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], 
                    [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], 
                    [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]]])
        return R


def main():
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-f', '--predictionFolder', type=str, default='saved/prediction/gestures', help="filename")
    args = arg_parser.parse_args()
    
    # if not args.prediction:
    arg_parser.usage = arg_parser.format_help()
    arg_parser.print_usage()

    #load files
    mypath = args.predictionFolder
    sim_arrs = []
    filenames = ["{}/{}".format(mypath, f) for f in os.listdir(mypath) if '.pkl' in f and '+' not in f]
    
    #get MAE/similarities
    for f in filenames:
        joints = Activation.loadPrediction(f)
        sim_arrs += [Activation.getSimilarityScores(joints)]
        # plt.plot(sim_arrs[-1])
        # plt.show(block=True)
    for sim, f in zip(sim_arrs, filenames):
        print("{}, mean={}, var={}".format(f, np.mean(sim), np.var(sim)))

    #print mean, var, 1-99 percentile for negtive and positive activation ges samples
    def getstats(sim_arrs):
        sims = []
        for s in sim_arrs: sims += s #flatten
        sims = np.array(sims)
        print()
        print("files={}, frames={}, mean={}, var={}, percentile={}".format(
            len(sim_arrs), sims.shape,
            np.mean(sims), np.var(sims),
            ["{}th={:.6f}".format(n,np.percentile(sims, n)) for n in [0.1, 1, 2, 3, 97, 98, 99, 99.9]]))
    getstats([sim for sim, f in zip(sim_arrs, filenames) if 'sign' not in f])
    getstats([sim for sim, f in zip(sim_arrs, filenames) if 'sign' in f])

    
if __name__ == "__main__":
    main()
