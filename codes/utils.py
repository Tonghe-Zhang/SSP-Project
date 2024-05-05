import numpy as np
import scipy 
import scipy.io
from scipy.fft import fft

from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy.signal import butter, filtfilt

def plot_answer(data, N, min_time:int, max_time:int):
    print(f"there are {data.size} sample point in the raw data")  # 20,000
    # plt.subplot(1,2,1)
    # for i in range(16):
    #     plt.subplot(4,4,i+1)
    #     plt.plot(data[(i)*200:(i+1)*200])
    #     plt.title('Signal')
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title(f'Time in ({(i-1)*200},{(i)*200})')
    
    # plt.show()

    freq=np.fft.fftfreq(N)
    power=1/N*np.abs(fft(data))**2
    freq_peak=freq[np.argmax(np.abs(power))]
    plt.plot(freq,power)
    plt.title(f'Average Power spectrum of the raw data={freq_peak:.10f}')
    plt.xlabel('Frequency')
    plt.ylabel('Average Power')
    plt.show()


def filter(data,target_frequency = 10.0,sampling_frequency = 10.0*1000):
    # low-pass filter the data, to remove high frequency noises.
    # our BPSK flipping frequency is 10HZ, while the data is sampled at the rate of 10KHZ
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_bandstop(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    

    def apply_lowpass_filter(data, cutoff_freq, fs, order=5):
        b, a = butter_lowpass(cutoff_freq, fs, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def apply_bandstop_filter(data, target_frequency, fs, order=5):
        filter_lowcut=target_frequency*0.99
        filter_highcut=target_frequency*1.01
        b, a = butter_bandstop(filter_lowcut, filter_highcut, fs=1000)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def apply_bandpass_filter(data, target_frequency, fs, order=5):
        filter_lowcut=target_frequency*0.99
        filter_highcut=target_frequency*1.01
        b, a = butter_bandpass(filter_lowcut, filter_highcut, fs=1000)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    # Cut-off frequency in Hz
    # Sampling frequency in Hz
    filtered_data = apply_bandpass_filter(data, target_frequency=target_frequency, fs=sampling_frequency)
    return filtered_data







