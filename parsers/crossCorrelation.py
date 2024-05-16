import utils.parameters as params
import numpy as np
from scipy.fftpack import fft, ifft, fftshift
import matplotlib.pyplot as plt
from utils.myUtils import MyUtils

class CrossCorrelation(object):
    """
        - zero-pad the input signals or apply a taper as we talked last week. (I didn't do this, since I know my two signal is zero at both ends, so I skip it)
        - take the FFT of both signals
        - multiply the first signal, and the reverse of the signal (or the conjugate, note that in the frequency domain, the complex conjugation is equivalent to time reversal in the time domain)
        - do the inverse FFT and get the shift
    """
    N_SAMPLE_SWEEP = params.N_SAMPLE_SWEEP
    FFT_WINDOWSIZE = N_SAMPLE_SWEEP
    FS = params.FS
    SHIFT_RESOLUTION = 1/FS*params.V_LIGHT/2# 1 shift = 1/FS*343/2 ~= 0.003572917 m ~=3mm

    def sliding_cross_corr(self, x, y, window_size = FFT_WINDOWSIZE, hanning = False):
        """
            @return: 7 x 512 x 11250
        """
        assert x.shape == y.shape
        n_sample = x.shape[0]
        
        overlap = 0
        step = window_size - overlap
        n_window = np.floor((n_sample - window_size) / step).astype('int')+1
        corr = np.zeros((n_window, window_size))#n_window * n_bin
        hanning_window = np.hanning(window_size)
        for w in range(n_window):
            x_cur = x[w * step:w * step + window_size] 
            y_cur = y[w * step:w * step + window_size]
            if hanning:
                x_cur = x_cur * hanning_window
                y_cur = y_cur * hanning_window
            corr[w] = self.cross_correlation_using_fft(x_cur, y_cur)
    #         corr[w] = signal.correlate(x_cur, y_cur)#
        corr = np.transpose(corr)
        return corr
    
    
    def cross_correlation_using_fft(self, x, y):#negative half when freq_y > freq_x
        """
            fftshift
            n : int Window length.m d : scalar, optional    Sample spacing (inverse of the sampling rate). Defaults to 1.
            - freqs = np.fft.fftfreq(10, 0.1)
                - array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
            - np.fft.fftshift(freqs)
                - array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
        """
        f1 = fft(x)
        # flip the signal of y
        f2 = fft(np.flipud(y)) #or the conjugate
        # note that in the frequency domain, the complex conjugation is equivalent to time reversal in the time domain
        # f2 = np.conjugate(fft(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc) #just rearrange the first half and second half

    def cross_correlation_using_fft2(self, x, y):#negative half when freq_y > freq_x
        f1,f2 = fft(x), fft(y)
        return ifft(f1 * np.conjugate(f2))

    def drawCorr(self, filename, data, t):
        win_size = data.shape[0]
        u = [int(win_size/2), int(-win_size/2)]
        
        plt.figure(figsize=(10, 4))
        plt.imshow(data, aspect='auto', origin='lower', extent=[0, t, u[0], u[1]])
        e = 10**np.floor(np.log10(win_size/20))
        yticksResolution = int(np.floor(win_size/20/e) * e)
        plt.yticks(np.arange(u[0], u[1] - yticksResolution, -yticksResolution))
        plt.tick_params(axis = 'y', which = 'both', labelleft = 'on', labelright = 'on')
        plt.title('[0,0] = LL,'+'Freq-CORR over time - ' + filename)
        plt.ylabel('n_sample of shift')
        plt.xlabel('Seconds')
        plt.draw()
        return

    def drawCorrDemo(self, filename, data, t, fname, width, windowIdxRight):
        win_size = data.shape[0]
        u = [windowIdxRight, windowIdxRight+int(-win_size)]
        
        plt.figure(figsize=(width, 4))
        plt.imshow(data, aspect='auto', origin='lower', extent=[0, t, u[0], u[1]])
        # e = 10**np.floor(np.log10(win_size/20))
        # yticksResolution = int(np.floor(win_size/20/e) * e)
        yticksResolution = 16
        plt.yticks(np.arange(u[0], u[1] - yticksResolution, -yticksResolution))
        # plt.tick_params(axis = 'y')
        # plt.ylabel('(zoom in) fast time/n_shift')
        # plt.xlabel('slow time/seconds')
        plt.savefig("imgs/out/{}.pdf".format(fname), bbox_inches = "tight")
        plt.draw()
        return

    def getSpectrumDifference(self, spectrum):
        n_window = spectrum.shape[1]
        spec_new = np.roll(spectrum, 1)
        spec_new[:, 0] = 0
        return spectrum - spec_new

    def mySoftmax(self, x, ax = 0):
        denom = np.exp(x).sum(axis=ax)
        return np.exp(x)/denom

    def pltDistance(self, corrs, selectedChannels, title = "", win_size = FFT_WINDOWSIZE, resolution = SHIFT_RESOLUTION):
        #corrs: nChannel x nBin x nWindow
        zeroIdx = win_size/2 -1
        plt.figure(figsize=(14,8))
        for i, corr in zip(selectedChannels, corrs):
            idxes =  zeroIdx - np.argmax(corr, axis = 0)
            d = idxes*resolution#*sound_speed/2
            plt.plot(d, label=i)
            print(i, d.shape)
        #ground truth
        plt.plot(np.arange(93, 5, (93-5)/len(corrs)), label="ground truth")
        plt.title(title)
        plt.ylabel("distance")
        plt.xlabel("n_window")
        plt.legend()
        plt.draw()
        
    def plotCorrSamples(self, corr, filename = "", win_size = FS/N_SAMPLE_SWEEP):
        idx = (np.arange(win_size, corr.shape[1], win_size)).astype(int)
        #samples visualization around every 1s
        plt.figure(figsize = (10,5))
        for i in idx:
            arr = corr[:, i]
            arr = signal.savgol_filter(arr, 21, 1)
            arr = MyUtils.unitNormalize(arr)
            plt.plot(arr, label= str(i//idx[0])+"s")
            plt.ylabel("corr (smoothed+normalize)")
            print("{}s, argmax={}, value={}".format(i//idx[0], np.argmax(arr), max(arr)))
        plt.title("corr peaks, "+filename)
        plt.legend()
        plt.draw()
