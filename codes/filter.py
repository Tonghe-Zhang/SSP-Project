import numpy as np
from scipy.signal import butter, filtfilt



# Generate BPSK signal (1s and -1s representing binary data)
bpsk_signal = np.array([1, -1, 1, -1, 1, -1, 1, -1,1, -1, 1, -1,1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, ])

# Generate single-frequency noise (e.g., sinusoidal signal)
noise_frequency = 0.1  # Frequency of the single-frequency noise
time = np.linspace(0, 1, len(bpsk_signal))
noise = np.sin(2 * np.pi * noise_frequency * time)
corrupted_signal = bpsk_signal + noise

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

cutoff_frequency = 0.09  # Choose a cutoff frequency below the noise frequency
sampling_frequency = 1.0  # For simplicity, assume unit time between samples

b, a = butter_lowpass(cutoff_frequency, sampling_frequency)
print(b.size)
print(a.size)
print(corrupted_signal.size)
filtered_signal = filtfilt(b, a, corrupted_signal)

import matplotlib.pyplot as plt

plt.figure()
plt.stem(time, bpsk_signal, label='BPSK Signal')
plt.plot(time, corrupted_signal, label='Corrupted Signal with Noise', linestyle='--')
plt.plot(time, filtered_signal, label='Filtered Signal', linestyle='-.')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()