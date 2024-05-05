import scipy.io
from scipy.fft import fft
import numpy as np

from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter


# mat is a dictionary 
mat = scipy.io.loadmat('data2.mat')
# data is a numpy.complex128 array
data=mat['x_complex'][0,:]
# which is of length 10,000 :=N
# under the sampling rate of 10KHz, the data is sampled within 1 second.
N=data.shape[0]
f_0_lis=[]
alpha_lis=[]
obj_lis=[]
t=np.linspace(0,1,N)

# the resolution of \alpha.
alpha_resolution='coarse'
if alpha_resolution=='coarse':
    alpha_range=np.arange(-50000,50000,200)
elif alpha_resolution=='fine':
    alpha_range=np.arange(195,205,0.01)
elif alpha_resolution=='superfine':
    alpha_range=np.arange(199,201,0.005)
else:
    raise ValueError('alpha_resolution must be coarse, fine, or superfine')

f_0_range=np.fft.fftfreq(N)
Power_Spectrum=np.zeros([alpha_range.size,N],dtype=np.float32)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, alpha in enumerate(alpha_range):
    phase_shift=np.complex128(np.cos(np.pi*alpha*t**2)+np.sin(-np.pi*alpha*t**2)*1j)
    data_shifted=data*phase_shift
    
    freq_coeff=fft(data_shifted)
    power_spectrum = 1/N* np.abs(freq_coeff) ** 2
    
    alpha_lis.append(alpha)
    f_0_lis.append(f_0_range[power_spectrum.argmax()])
    obj_lis.append(power_spectrum.max())
    
    Power_Spectrum[i,:]=power_spectrum
    
    # print(power_spectrum.max())
    # we plot different spectrums and align them along the x axis. 
    # for this reason each slice of the spectrum shares the same x coordinate, while their (y,z) coordinates are different.
    ax.plot(np.full_like(power_spectrum,alpha),
            f_0_range,
            power_spectrum/1e3)

argmax_alpha,argmax_f0=np.unravel_index(np.argmax(Power_Spectrum), Power_Spectrum.shape)
alpha_optimal=alpha_range[argmax_alpha]
f0_optimal=f_0_range[argmax_f0]

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_0$')
ax.set_zlabel(r'Power Spectrum/$10^3$')
ax.set_title(r'Power Spectrum of Chipr Signal at Different $(\alpha, f_0)$ '+'\n Optimal point: '+r'$(\hat{\alpha}_{MLE},\hat{f}_{0,MLE})=$'+f'({alpha_optimal:.3f},{f0_optimal:.3f})')
plt.show()

# print(Power_Spectrum.shape)
# print(Power_Spectrum)

