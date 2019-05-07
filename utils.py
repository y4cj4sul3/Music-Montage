from scipy.io import wavfile as wav
import numpy as np

def read_wav(f):

    sr, y = wav.read(f)

    # reformat to 32-bit
    if y.dtype == np.int16:
        y = y/2**(16-1)
    elif y.dtype == np.int32:
        y = y/2**(32-1)
    elif y.dtype == np.int8:
        y = (y-2**(8-1))/2**(8-1)

    # 2 channel to 1
    if y.ndim == 2:
        y = y.mean(axis=1)

    return (sr, y)

def genSoundBank(audio, sr, ms=1000):
    '''
    slice audio into clips 
    '''
    T = np.shape(audio)[0]
    print(T)
    # clip length
    t = (sr*ms)//1000
    S = T//t

    SB = np.reshape(audio[:S*t], (S, t))

    return SB
