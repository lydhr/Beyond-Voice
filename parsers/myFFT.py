import utils.parameters as params
import numpy as np
# from scipy import signal, stats, fftpack
from scipy.fftpack import fft, ifft, fftshift
import matplotlib.pyplot as plt


class MyFFT(object):
    FFT_WINDOWSIZE = params.N_SAMPLE_SWEEP
    FS = params.FS

    def getSpectrum(self, raw, window_size = FFT_WINDOWSIZE, selectedFreqRange = None, overlapRatio = 0): # raw = n_channel x samples
        """
            param: raw = (n_channel x n_sample)
        """
        if len(raw.shape) < 2: 
            # raw = raw.reshape(1, raw.shape[0])
            raw = np.expand_dims(raw, axis=0)

        overlap = int(window_size*overlapRatio);
        step = window_size - overlap;
        n_window = np.floor((raw.shape[1] - window_size)/step).astype('int')+1
        
        X = []# n_window x n_channel x n_bin
        for w in range(n_window):
            X_cur = raw[:, w*step:w*step+window_size]; #nChannel x window_size
            X.append(self.getFFTComplex(X_cur, self.FS, selectedFreqRange))
        X = np.transpose(np.array(X), (1, 2, 0))# n_channel x n_bin x n_window
        print("FFT => {} n_channel x n_bin x n_window".format(X.shape))
        return X

    # my selected FFT
    def getFFTComplex(self, raw, fs, selectedFreqRange): # raw = nChannel x n, n could be != L; selectedFreqRange = [start, end] or None/all
        L = raw.shape[1]
        X = fft(raw, axis = 1)##2 dimension, i.e fft of each row
        X = X[:, :int(L/2)]#first half
        if selectedFreqRange is None: 
            return X #no specific selected bins
        else:
            #P2 = abs(X*2)/L;#power
            #[B, I] = maxk(P2, 1, 2);I
            freq_resolution = fs/L#nyquist_freq/(L/2)
            #f = freq_resolution*(0:L/2-1);
            target_idx = self.getFreqBinIndex(selectedFreqRange, freq_resolution)
            selectedFFT = X[:, target_idx]
            return selectedFFT

    def getFreqBinIndex(self, selectedFreqRange, freq_resolution):
        f = np.arange(selectedFreqRange[0], selectedFreqRange[1], freq_resolution)
        return np.floor(f/freq_resolution).astype('int')#left-closed, right-open interval

    #plot
    def drawSpectrogram(self, title, spectrogram, t, f = [0, FS/2], yticksResolution = 1000):
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', 
                   extent=[0, t, f[0], f[1]])
        plt.yticks(np.arange(f[0], f[1] + yticksResolution, yticksResolution))
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.title('Spectrogram of ' + title)
        plt.ylabel('frequency/Hz')
        plt.xlabel('seconds/s')
        plt.draw()

        return

