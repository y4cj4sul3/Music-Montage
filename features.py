import numpy as np
#import tensorflow
#import librosa
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

def getTempo(f):
    # beat activation: the probability of a beat at each frame (fps=100)
    act = RNNBeatProcessor()(f)
    # processor
    proc = TempoEstimationProcessor(fps=100)
    # return estimate tempi (in bpm) and their relative strength
    tempo = proc(act)
   
    threshold = 0.08*tempo[0, 0]
    times = np.array([2, 3, 4, 6, 8])*tempo[0, 0]
    hamonic = []
    for t in tempo[:, 0]:
        if np.any(np.abs(t-times) <= threshold):
            hamonic.append(t)
            
    if hamonic != []:
        return np.max(hamonic)
    else:
        return tempo[0, 0]