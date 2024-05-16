#os
FOLDER_NAME = "../data"
FNAME_AUDIO = "25db-17k-20k"
FNAME_LEAPMOTION_JOINTS = "leapmotion_joints.pkl"
FNAME_LEAPMOTION_ANGLES = "leapmotion_angles.pkl"
FNAME_MEDIAPIPE_JOINTS = "mediapipe_joints.pkl"
FNAME_XY = "xy.pkl"
FNAME_SAVED_MODEL = 'saved/models/saved_model.pth'
FNAME_LOAD_MODEL = 'saved/models/saved_model.pth'
FNAME_PREDICTION = 'saved/prediction/prediction.pkl'
# FNAME_LEAPMOTION_FRAMES = "leapmotion_frames.data"

FNAME_VIDEO = ""
#audio general
FS = int(48e3)
N_CHANNEL = 7
N_SAMPLE_SWEEP = 512
CORR_HEIGHT = 256 #256/48000*343/2 ~= 0.91m
N_SWEEP_INPUT = 50 #512/48000 = 0.010666667s, hand speed 0.2m/s, 5 * 512/48000 * 0.2 = 0.010666667m
HIGHPASS_FILTER = 17e3
HIGHPASS_FILTER_ORDER = 5# order of filter ~= steepness of cut off

V_LIGHT = 343#speed of light

N_EPOCH = 50

NET = 'LSTMNet' #myDataset: if 'Angle' in NET

GROUND_TRUTH = ['leapmotion', 'mediapipe'][0]

#VIS
ESTIMATED_GT_FS = 100 if GROUND_TRUTH == 'leapmotion' else 30
COORDINATE_RANGE = [[-200, 200], [0, 400], [-200, 200]] if GROUND_TRUTH == 'leapmotion' else [[0, 1], [0, 1], [-2, 2]]#x, y, z
CHAINS_INDICES = [[0, 1, 2, 3],  # analysis/imgs/finger.png
              [4, 5, 6, 7],  
              [8, 9, 10, 11],
              [12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [4, 8, 12, 17],
              [0, 16]] if GROUND_TRUTH == 'leapmotion' else [[1, 2, 3, 4],  # analysis/imgs/fingerMediapipe.png
              [5, 6, 7,  8],
              [9, 10, 11, 12],
              [13, 14, 15, 16],
              [0, 17, 18, 19, 20],
              [5, 9, 13, 17],
              [1, 0]]