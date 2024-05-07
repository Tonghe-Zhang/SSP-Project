import utils
from utils import plot_answer
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import fft
from fft_toolkit import fft_all


###############################################################################################################################################
###############################################################################################################################################
########################################################         load data                #####################################################
###############################################################################################################################################
###############################################################################################################################################

# sampling frequency
fs=10*1e3  
# load data
# mat is a dictionary 
mat = scipy.io.loadmat('data1.mat')
# data is a numpy.complex128 array
data_raw=mat['bpsk_s'][0,:]
# which is of length 10,000 :=N
# under the sampling rate of 10KHz, the data is sampled within 1 second.
N=data_raw.shape[0]
time=np.arange(N)
# fft_all(data_raw,fs,True)

###############################################################################################################################################
###############################################################################################################################################
########################################################   remove fixed frequency noise   #####################################################
###############################################################################################################################################
###############################################################################################################################################

data=data_raw


# for i in range(19):
#     batch=data[i*2000:(i+1)*2000]
#     _,_,_,phase_all=fft_all(batch,fs,False)
#     # phase_all_difference=np.diff(np.unwrap(phase_all,period=360))
#     # plt.plot(np.arange(len(phase_all_difference)),phase_all_difference)
#     plt.plot(np.arange(len(phase_all)),phase_all)
#     plt.show()





###############################################################################################################################################
###############################################################################################################################################
########################################################   discover first flip time       #####################################################
###############################################################################################################################################
###############################################################################################################################################
# extract first flip time
fft_result = np.fft.fft(data)
magnitudes = np.abs(fft_result)
phases = np.angle(fft_result)
def extract_first_flip(phases):
    # # phase jump detection. If there is a time when the BPSK flips, then the phase changes +-\pi instantly, causing the difference to jump from
    # # values near zero to almost pi or almost -pi:
    # |\phi[n+1]-\phi[n]| \approx \pi
    # eps=0.005
    # flip_ids = np.where(np.abs(
    #     np.abs(phases[1:-1]-phases[0:-2])-np.full_like(phases[1:-1],np.pi)
    # ) < eps)[0]
    # phases=np.unwrap(phases)
    flip_ids=np.where(np.abs(np.diff(phases))>np.pi*0.9992)[0]

    print(f"there are {flip_ids.size} times of flipping in the data")
    print(f"the time indices at which a flipping occurs are {flip_ids}")
    t0=np.min(flip_ids)
    print(f"initial flip is at time={t0}")
    return t0, flip_ids.size,flip_ids

t0, flip_number,flip_ids=extract_first_flip(phases)


###############################################################################################################################################
###############################################################################################################################################
########################################################   after t0 is found   ################################################################
###############################################################################################################################################
###############################################################################################################################################

# extract the 1000-point batch samples.
batch_size=1000
num_batch=int((N-1-t0)/batch_size)
batches_of_data=np.zeros([num_batch,batch_size])
for i in range(num_batch):
    batches_of_data[i,:]=data[t0+batch_size*i:t0+batch_size*(i+1)]
print(batches_of_data.shape)  # (16,1000)

# compute appproximate MLE of f,A,phi for each batch of data.
A_mle_samples=np.zeros(num_batch,dtype=np.float32)
phi_mle_samples=np.zeros(num_batch,dtype=np.float32)
f_mle_samples=np.zeros(num_batch,dtype=np.float32)
for i in range(num_batch):
    batch=batches_of_data[i]

    freq, F, amp_pos, phase_pos=fft_all(batch,fs,plot=True,title=f'data[{i*1000+t0}:{(i+1)*1000+t0}]')
    
    optimal_freq_id=np.argmax(amp_pos)

    f_mle=freq[optimal_freq_id]
    phi_mle=-phase_pos[optimal_freq_id]
    A_mle=amp_pos[optimal_freq_id]

    f_mle_samples[i]=f_mle
    A_mle_samples[i]=A_mle
    phi_mle_samples[i]=phi_mle


###############################################################################################################################################
###############################################################################################################################################
########################################################        plot           ################################################################
###############################################################################################################################################
###############################################################################################################################################

# plot the MLE estimate results in each batch of 1000-point data.
plt.subplot(3,1,1)
plt.grid()
plt.plot(np.arange(num_batch),f_mle_samples,label=r'$f$')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{f}_{{MLE}}$')
plt.title(r'$\hat{f}_{{MLE}}$ Average='+f'{np.mean(f_mle_samples):.3f}')

plt.subplot(3,1,2)
plt.grid()
plt.plot(np.arange(num_batch),A_mle_samples,label='A')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{A}_{{MLE}}$')
plt.ylim([0,np.max(A_mle_samples)*1.2])
plt.title(r'$\hat{A}_{{MLE}}$ Average='+f'{np.mean(A_mle_samples):.3f}')

plt.subplot(3,1,3)
plt.grid()
plt.plot(np.arange(num_batch),phi_mle_samples,label=r'$\varphi$')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{\varphi}_{{MLE}}$')
plt.ylim([-180,180])
plt.title(r'$\hat{\varphi}_{{MLE}}$ Average='+f'{np.mean(phi_mle_samples):.3f}')

plt.suptitle(f'MLE estimates. First Flip at t={t0}')
plt.show()




"""
    freq=np.fft.fftfreq(batch_size)

    frequency_spectrum=fft(batch)
    
    average_power_spectrum=1/batch_size*np.abs(frequency_spectrum)**2
    optimal_freq_id=np.argmax(average_power_spectrum)



        f_mle=freq[optimal_freq_id]
    A_mle=np.abs(frequency_spectrum[optimal_freq_id])
    phi_mle=-np.angle(frequency_spectrum[optimal_freq_id])


 plt.subplot(2,1,1)
    plt.plot(freq,np.abs(frequency_spectrum))
    plt.xlabel('frequency')
    plt.ylabel('angle')
    plt.title(f'Amplitude of data[{i*1000+t0}:{(i+1)*1000+t0}]')

    
    plt.subplot(2,1,2)
    plt.plot(freq,np.angle(frequency_spectrum))
    plt.xlabel('frequency')
    plt.ylabel('angle')
    plt.title(f'Phase of data[{i*1000+t0}:{(i+1)*1000+t0}]')
    
    plt.show()

    
"""
















