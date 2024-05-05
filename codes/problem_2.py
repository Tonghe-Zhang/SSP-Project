import utils
import matplotlib.pyplot as plt
from utils import plot_answer
import numpy as np
import scipy
from scipy.fft import fft

# load data
# mat is a dictionary 
mat = scipy.io.loadmat('data1.mat')
# data is a numpy.complex128 array
data_raw=mat['bpsk_s'][0,:]
# which is of length 10,000 :=N
# under the sampling rate of 10KHz, the data is sampled within 1 second.
N=data_raw.shape[0]
time_ids=np.arange(N)
# show the frequency spectrum of data.
plot_answer(data_raw,N,0,N-1)


###############################################################################################################################################
###############################################################################################################################################
########################################################   remove fixed frequency noise   #####################################################
###############################################################################################################################################
###############################################################################################################################################

# filter frequency noises
from filter_new import filter_freq_noise, plot_filter_result
data=filter_freq_noise(data_raw,bandwidth=0.05)
plot_filter_result(data=data_raw,filtered_data=data)

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
    phase_diff=np.unwrap(phases)
    flip_ids=np.where(np.abs(np.diff(phase_diff))>np.pi*0.9992)[0]
    # # get the top 19 indices at which the phase difference exceeds pi
    # # Find the indices where x[t] exceeds np.pi
    # exceed_indices = np.where(phase_diff > np.pi*0.99)[0]
    # # Get the top 19 indices with the highest values exceeding np.pi
    # flip_ids = np.argsort(phase_diff[exceed_indices])[::-1][:19]
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

freq=np.fft.fftfreq(batch_size)
for i in range(num_batch):
    batch=batches_of_data[i]
    frequency_spectrum=fft(batch)
    average_power_spectrum=1/batch_size*np.abs(frequency_spectrum)**2
    optimal_freq_id=np.argmax(average_power_spectrum)
    
    # plt.plot(freq,average_power_spectrum)
    # plt.xlabel('frequency')
    # plt.ylabel('power spectrum')
    # plt.title(f'Power Spectrum  of data[{i*1000+t0}:{(i+1)*1000+t0}]')
    # plt.show()

    A_mle=2*np.max(average_power_spectrum)
    phi_mle=-np.angle(frequency_spectrum[optimal_freq_id])
    f_mle=freq[optimal_freq_id]

    A_mle_samples[i]=A_mle
    phi_mle_samples[i]=phi_mle
    f_mle_samples[i]=f_mle

# plot the MLE estimate results in each batch of 1000-point data.
plt.subplot(3,1,1)
plt.plot(np.arange(num_batch),f_mle_samples,label=r'$f$')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{f}_{{MLE}}$')
plt.title(r'$\hat{f}_{{MLE}}$ Average='+f'{np.mean(f_mle_samples):.3f}')

plt.subplot(3,1,2)
plt.plot(np.arange(num_batch),A_mle_samples,label='A')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{A}_{{MLE}}$')
plt.ylim([0,np.max(A_mle_samples)*1.2])
plt.title(r'$\hat{A}_{{MLE}}$ Average='+f'{np.mean(A_mle_samples):.3f}')

plt.subplot(3,1,3)
plt.plot(np.arange(num_batch),phi_mle_samples,label=r'$\varphi$')
plt.legend()
plt.xlabel('Batch Number')
plt.ylabel(r'$\hat{\varphi}_{{MLE}}$')
plt.ylim([-np.pi,np.pi])
plt.title(r'$\hat{\varphi}_{{MLE}}$ Average='+f'{np.mean(phi_mle_samples):.3f}')

plt.suptitle(f'MLE estimates. First Flip at t={t0}')
plt.show()

















