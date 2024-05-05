import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft


def filter_freq_noise(data,bandwidth = 0.01):
    target_frequency = 0.10005
    def butter_notch(target_frequency, bandwidth, fs, order=5):
        nyquist = 0.5 * fs
        low = (target_frequency - bandwidth/2) / nyquist
        high = (target_frequency + bandwidth/2) / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a
    # Define the bandwidth around the target frequency to filter out
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