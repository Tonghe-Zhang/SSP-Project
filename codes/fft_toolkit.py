import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import fft


def generate_data():
    # sampling frequency
    fs=10*1e3

    # data generation
    load_data=False
    if load_data:
        # load data
        # mat is a dictionary 
        mat = scipy.io.loadmat('data1.mat')
        # data is a numpy.complex128 array
        data_raw=mat['bpsk_s'][0,:]
        # which is of length 10,000 :=N
        # under the sampling rate of 10KHz, the data is sampled within 1 second.
        N=data_raw.shape[0]
        time=np.arange(N)
    else:
        # signal frequency
        f=2.00*1e3
        time=np.arange(0,1,1/fs)  # signal of 1 second, 10,000 points.
        data=0.32*np.sin(2*np.pi*f*time)
        N=len(data)
    return fs,data,N

def fft_all(data:np.array,
            fs:float,
            plot:bool,
            title='Frequency Response of the Input Data'):
    """
    Inputs:
        data
        fs: sampling frequency
        plot: whether to plot the amplitude and frequency spectrums
    Outputs:
        freq: the positive frequency range, [0:fs/2]
        F_pos_freq: 
    """
    N=len(data)
    # Amplitude Spectrum
    # the amplitude/phase information of at each time. F is of the same shape of data.
    F=np.fft.fft(data)
    # get the range of frequencies, from DC(f=0) to sampling frequency
    # and we only save the positive part of the frequencies(by Nyquist sampling theorem.)
    freq=np.linspace(0,fs,N)
    freq_pos=freq[0:int(N/2)]
    amp=np.abs(F)
    amp_pos=amp[0:int(N/2)]
    # scale the amplitude spectrum from FFT to correct the power spectrum's bias. 1/N, 2/N
    amp_pos[0]/=N
    amp_pos[-1]/=N
    amp_pos[1:-1]/=(N/2)
    if plot==True:
        plt.subplot(1,2,1)
        plt.plot(freq_pos,amp_pos)
        plt.xlabel('Freq/Hz')
        plt.ylabel('Amplitude')
        plt.title('Amplitude Spectrum')
        plt.grid()

    

    # Phase Spectrum
    # Remember that we should only calculate phase information at the frequencies that exists in the signal.
    # but not each and every point, since due to numerical value, some frequency components, though 
    # they does not actually correspond to practical amplitudes, still could carry non=zero 
    # Im and Re parts, which will bring noisy phase information to the data.

    # First we check whether this frequency component correspond to a sufficiently large amplitude
    # which indicates that it does exist in the real signal. We will filter out (set to zero)
    # those frequencies that does not actually exists in the data, and keep the rest unchanged.
    epsilon=np.max(amp)/1e4
    F_keep_exist=np.where(amp>epsilon,F,0)
    phase=np.angle(F_keep_exist)*(180/np.pi)  # in degree.
    phase_pos=phase[0:int(N/2)] # only save the positive frequency part.

    if plot==True:
        plt.subplot(1,2,2)
        plt.plot(freq_pos,phase_pos)
        plt.xlabel('Freq/Hz')
        plt.ylabel('Phase')
        plt.title('Phase Spectrum'+'\n'+f'Min={np.min(phase_pos):.3f}, Max={np.max(phase_pos):.3f}, Diff={np.max(phase_pos)-np.min(phase_pos):.3f}')
        plt.grid()
        plt.suptitle(title)
        plt.show()   

    return freq, F, amp_pos, phase_pos
    '''
    You can read the amplitude and phase information from the spectrums. 
    The phase information represent the phase for cos(\cdot) components in default.
    '''

