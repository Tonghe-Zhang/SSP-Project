# data_shift_one=data[1:-1]
# difference=data[0:-2]-data_shift_one
# plt.plot(data[0:-2], label='data',c='yellow')
# plt.plot(data_shift_one, label='data shift',c='pink')         
# plt.plot(np.abs(difference),label='difference',c='black')
# plt.legend(loc='upper left')
# plt.show()

# batches_of_data=[data[i*1000, (i+1)*1000] for i in range(20)]
# batches_of_phase=[np.zeros(1000) for _ in range(20)]
# f_0_range=np.fft.fftfreq(N)
# for i in range(20):
#     # fetch a batch of real data.
#     batch_data=batches_of_data[i]
#     # use fft to calculate the phase and manitude aat each t
#     fft_result = np.fft.fft(batch_data)
#     batch_magnitudes = np.abs(fft_result)
#     batch_phases = np.angle(fft_result)

#     batches_of_phase[i]=batch_phases



# print(np.max(np.abs(phase_diff)))
# print(flip_ids)
# print(flip_ids.size)
# plt.plot(phase_diff)
# plt.plot(phase_diff)
# for x_idx in flip_ids:
#     plt.axvline(x=time_ids[x_idx], color='r', linestyle='--',label=f'x={time_ids[x_idx]}')
# plt.plot(time_ids,180*np.ones_like(time_ids),label=r'$\phi=180$',c='black')
# plt.plot(time_ids,-180*np.ones_like(time_ids),label=r'$\phi=-180$',c='black')
# plt.xlabel('Time Step')
# plt.ylabel('Phase Difference')
# plt.title('Phase Difference Information')
# plt.ylim([-200,200])
# plt.legend()
# plt.show()

# t0=np.min(flip_ids)
# print(t0)

import numpy as np
import scipy 
import scipy.io
from scipy.fft import fft

from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy.signal import butter, filtfilt
import utils
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import fft
import numpy as np
from scipy.signal import butter, filtfilt

# # load data
# # mat is a dictionary 
# mat = scipy.io.loadmat('data1.mat')
# # data is a numpy.complex128 array
# data=mat['bpsk_s'][0,:]
# # which is of length 10,000 :=N
# # under the sampling rate of 10KHz, the data is sampled within 1 second.
# N=data.shape[0]
# time=np.arange(N)

def filter_freq_noise(data):
    target_frequency = 0.10005

    def butter_notch(target_frequency, bandwidth, fs, order=5):
        nyquist = 0.5 * fs
        low = (target_frequency - bandwidth/2) / nyquist
        high = (target_frequency + bandwidth/2) / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a

    # Define the bandwidth around the target frequency to filter out
    bandwidth = 0.01  #

    # Design the notch filter
    b, a = butter_notch(target_frequency, bandwidth, fs=1.0)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def plot_filter_result(data, filtered_data):
    N=data.size
    time=np.arange(N)
    # plot
    import matplotlib.pyplot as plt
    plt.figure()

    plt.subplot(2,3,1)
    plt.plot(time,data, label='Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Amplitude')
    plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(time,np.angle(fft(data)), label='Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Phase')
    plt.legend()
    plt.title('Phase')
    plt.grid(True)

    plt.subplot(2,3,3)
    freq=np.fft.fftfreq(N)
    power=np.abs(fft(data))**2
    power_filt=np.abs(fft(filtered_data))**2
    plt.plot(freq,power, label='Original Signal')
    plt.grid(True)
    plt.xlabel('Freq')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.legend()


    plt.subplot(2,3,4)
    plt.plot(time, filtered_data, label='Filtered Signal', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Amplitude')
    plt.legend()
    plt.grid(True)


    plt.subplot(2,3,5)
    plt.plot(time,np.angle(fft(filtered_data)), label='Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Phase')
    plt.legend()
    plt.title('Phase')
    plt.grid(True)

    plt.subplot(2,3,6)
    freq=np.fft.fftfreq(N)
    power=1/N*np.abs(fft(data))**2
    power_filt=1/N*np.abs(fft(filtered_data))**2
    plt.plot(freq,power_filt,  label='Filtered Signal')
    plt.grid(True)
    plt.xlabel('Freq')
    plt.ylabel('Average Power')
    plt.legend()
    plt.title('Average Power Spectrum filtering')

    plt.suptitle('Filtering')
    plt.show()