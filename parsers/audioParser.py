import utils.parameters as params
from parsers.myFFT import MyFFT
from parsers.crossCorrelation import CrossCorrelation
from parsers.groundTruthParser import GroundTruthParser
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import Counter

class AudioParser(object):
    FNAME = params.FNAME_AUDIO
    FS = params.FS
    N_CHANNEL = params.N_CHANNEL
    #self.t #float, time/s
    N_SAMPLE_SWEEP = params.N_SAMPLE_SWEEP
    N_SWEEP_INPUT = params.N_SWEEP_INPUT
    HIGHPASS_FILTER = params.HIGHPASS_FILTER
    HIGHPASS_FILTER_ORDER = params.HIGHPASS_FILTER_ORDER
    CORR_HEIGHT = params.CORR_HEIGHT

    START_FREQ, END_FREQ = 17e3, 20e3
    BIT_PER_SAMPLE = 24

    # T_SWEEP = N_SAMPLE_SWEEP/FS#~0.0107
    # FREQUENCY_RESOLUTION = FS/FFT_WINDOWSIZE

   
    def __init__(self, foldername, draw = False):
        self.filename = "{}/{}".format(foldername, self.FNAME)
        self.myFFT = MyFFT()
        self.corr = CrossCorrelation()
        
        self.rawData = np.array([])
        self.simulationData = np.array([])
        self.corrData = np.array([])

        self.loadAudio()
        self.setSimulationSamples()
        self.setCorr()

        if draw:
            self.drawSpectrum()
            self.drawSimulationSpectrum()
            self.drawBeginningDelay()
            self.drawCorr()
            plt.show()

    def loadAudio(self):
        """
            self.rawData = (n_channel x n_samples)
        """
        with open(self.filename) as f:
            print("......loading audio")
            lines = f.readlines()
            raw = np.array([int(line) for line in lines])
            data = np.array(raw) #sometimes audio is longer than video because of fps problem
            self.t = len(data)/self.N_CHANNEL/self.FS #len = self.t * self.FS * self.N_CHANNEL
            #transform
            assert len(data)%self.N_CHANNEL == 0
            data = data.reshape(data.shape[0]//self.N_CHANNEL, self.N_CHANNEL)
            data = np.transpose(data)#n_channel x n_frames
            data = self.butter_highpass_filter(data)
            print("loaded {}: {:.3f}s, (n_channel x n) = {} = {}".format(self.filename, self.t, data.shape, data.size))
        self.rawData = data
        
        return
    def butter_highpass_filter(self, data, fs = FS, cutoff = HIGHPASS_FILTER, order = HIGHPASS_FILTER_ORDER):
        '''
            params: data = (n_channel, n_frame)
        '''
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        #Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        b, a = signal.butter(order, normal_cutoff, btype = 'high', analog = False)
        for channel in range(data.shape[0]):
            data[channel] = signal.filtfilt(b, a, data[channel])
        return data

    def setSimulationSamples(self, shift = 0):
        """
            self.simulationData = (n_samples, )
            simulate same length of recording
        """
        maxAmplitude = int(np.power(2, self.BIT_PER_SAMPLE - 1) - 1)

        t = np.arange(0, self.N_SAMPLE_SWEEP,1)/self.FS
        k = (self.END_FREQ - self.START_FREQ)/(self.N_SAMPLE_SWEEP/self.FS)
        phase = np.pi * 2 * (self.START_FREQ * t + 0.5 * k * np.power(t, 2)) % (2 * np.pi)
        samples = np.round(np.sin(phase) * maxAmplitude)
        samples = np.tile(samples, int(np.ceil(self.t * self.FS / self.N_SAMPLE_SWEEP) + 1))#repeative chirp, +1 assumes shift <= N_SAMPLE_SWEEP
        samples = samples[shift:(shift + int(self.t * self.FS))]
        
        # samples = np.expand_dims(samples, axis=0)
        # print("simulation samples  = {} = {}s".format(samples.shape, samples.shape[0]/self.FS))
        self.simulationData = samples
        
        return

    def setCorr(self):
        """
            self.corrData = n_channel x N_SAMPLE_SWEEP x (n_window = n_sample / N_SAMPLE_SWEEP)
            cross correlation of self.simulationData & self.rawData
            defaut: select all channels
        """
        if self.simulationData.size == 0: 
            self.setSimulationSamples()
        if self.rawData.size == 0: 
            self.loadAudio()

        data = []
        for c in range(self.N_CHANNEL):
            spectrum = self.corr.sliding_cross_corr(self.simulationData, self.rawData[c])

            #postprocessing
            spectrum = abs(spectrum)
            spectrum = self.corr.getSpectrumDifference(spectrum)
            spectrum[spectrum < 0] = 0 #for removing the green background in visualization

            data.append(spectrum)
            
        data = np.array(data)
        print("corr {}".format(data.shape))
        self.corrData = data

        return

    def _removeInferenceBound(self, height = CORR_HEIGHT):
        '''
            :return n_channel x height x n_window
        '''
        up = self.getInferencePoint()
        down = up - height
        print("sub height CORR [{}:{}]".format(down, up))
        if down < 0: exit()
        return self.corrData[:, :, down:up, :]
    
    def removeInferenceBound(corrData, up, height = CORR_HEIGHT):
        down = up - height
        print("sub height CORR [{}:{}]".format(down, up))
        if down < 0: exit()
        return corrData[:, :, down:up, :]

    def getInferencePoint(self):
        '''
            :return: the row_idx of the direct path in CORR spectrum
        '''
        rawCorr = self.getRawCorr(channels = [0])[0]
        maxPeaks = np.argmax(rawCorr, axis = 0)
        cnt = Counter(maxPeaks).most_common(10)
        print("counter of CORR argmax(colom): {}".format(cnt))
        return max([i[0] for i in cnt])

    def getCorrInput(self):
        return self._removeInferenceBound()


    def getRawCorr(self, channels = np.arange(N_CHANNEL)):
        if self.simulationData.size == 0: 
            self.setSimulationSamples()
        if self.rawData.size == 0: 
            self.loadAudio()
        data = []
        for c in channels:
            spectrum = self.corr.sliding_cross_corr(self.simulationData, self.rawData[c])
            data.append(spectrum) 
        data = np.array(data)
        return data

    @staticmethod
    def getSlicedCorr(corrData, windowIdx, input_width = N_SWEEP_INPUT):
        """
            corrData = (n_channel x N_SAMPLE_SWEEP x n_window)
        """
        validWindows, indices2delete = GroundTruthParser.getValidWindowIdx(corrData.shape[2] - 1, windowIdx)
        corr = np.array([corrData[:, :, (right - input_width + 1):(right + 1)] for right in validWindows])
        print("sliced to {}".format(corr.shape))

        return corr, indices2delete


    def drawCorr(self, selectedChannels = [0]):
        if self.corrData.size == 0:
            self.setCorr()

        #default only draw channel 0
        for c in selectedChannels:
            title = "{}&{}, channel {}".format("simulation", self.getFilenameKeyword(), c)
            self.corr.drawCorr(title, self.corrData[c], self.t)
        return


    def drawSimulationSpectrum(self, window_size = N_SAMPLE_SWEEP//4):
        """
            set a smaller window_size to see the change within one sweep
        """
        if self.simulationData.size == 0: 
            self.setSimulationSamples()

        spectrums = np.absolute(self.myFFT.getSpectrum(self.simulationData, window_size))
        self.myFFT.drawSpectrogram("simulation", spectrums[0], self.t)

        return

    def drawSpectrum(self, window_size = N_SAMPLE_SWEEP//4, selectedChannels = [0]):
        """
            default only show channel0
            set a smaller window_size to see the change within one sweep
        """
        if self.rawData.size == 0: 
            self.loadAudio()
        spectrums = self.myFFT.getSpectrum(self.rawData, window_size)
        for c in selectedChannels:
            title = "{}_channel{}".format(self.getFilenameKeyword(), c)
            spec = np.absolute(spectrums[c])
            self.myFFT.drawSpectrogram(title, spec, self.t)

        return

    def drawBeginningDelay(self, n_beginning = 2500, selectedChannels = [0]):
        if self.simulationData.size == 0: 
            self.setSimulationSamples()
        if self.rawData.size == 0: 
            self.loadAudio()

        simulationDataAmplitudeScaler = 200
        x = np.arange(0, n_beginning)
        plt.figure(figsize=(10,4))
        for c in selectedChannels:
            label = "channel{}".format(c)
            y = self.rawData[c, 0:n_beginning]
            plt.scatter(x, y, label = label, s = 1)
        y = self.simulationData[0:n_beginning]/simulationDataAmplitudeScaler
        plt.scatter(x, y, label = "simulation", s = 1)

        plt.title("The beginning of time-domain raw signals")
        plt.ylabel("amplitude")
        plt.xlabel("n_sample")
        plt.legend()
        plt.draw()
        return

    def getFilenameKeyword(self):
        return self.filename.split("/")[-2]

