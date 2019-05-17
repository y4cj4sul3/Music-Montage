import numpy as np
import utils
from scipy.io import wavfile as wav

def patternMatching(target, SB, step_size=None):
    '''
    target: F * T, F: feature range, T: time
    SB: S * F * t, S: # clips, t: time
    '''

    T = np.shape(target)[-1]
    S = np.shape(SB)[0]
    t = np.shape(SB)[-1]
    print(T, S, t)
    if step_size == None:
        step_size = t

    symbolic = []

    for timestep in range(0, T, step_size):
        # segment target to S * F * t
        target_seg = target[timestep:timestep+t]
        if len(target_seg) < t:
            target_seg = np.pad(target_seg, (0, t-len(target_seg)), 'constant')

        min_dist = np.inf
        idx = 0
        for i in range(len(SB)):
            # distance
            dist = distanceFunction(target_seg, SB[i])
            if min_dist > dist:
                idx = i
                min_dist = dist

        # pick the best one
        symbolic.append(idx)

    return symbolic


def distanceFunction(a, b):
    '''
    a, b: 3-D matrix, S * F * T
    '''
    # MSE
    # return np.mean(np.mean(np.power(a-b, 2), axis=2), axis=1)
    return np.mean(np.power(a-b, 2))

def synthesize(symbolic, SB):
    output = np.array([])

    for idx in symbolic:
        output = np.append(output, SB[idx])

    return output

def dumpWav(audio, params, filename='output.wav'):
    with wave.open(filename, 'w') as wave_file:
        wave_file.setparans(params)
        

    

def featureExtraction(input_audio):
    pass

if __name__ == '__main__':
    # load data
    #target_sr, target = utils.read_wav('dataset/input_data/BPS_piano/23.wav')
    target_sr, target = utils.read_wav('dataset/input_data/senponsakura/senponsakura1.wav')
    SB_sr, sound_source = utils.read_wav('dataset/input_data/BPS_piano/3.wav')
    #SB_sr, sound_source = utils.read_wav('dataset/input_data/only my railgun/only my railgun7.wav')

    SB = utils.genSoundBank(sound_source[:44100*30], SB_sr, 100)

    print(np.shape(SB))
    if target_sr != SB_sr:
        print("[Warning]: sample rate doesn't match")

    # extract feature
    

    # pattern matching
    sym = patternMatching(target[:44100*30], SB)
    print(np.shape(sym))
    print(sym)

    # synthesize
    output = synthesize(sym, SB)
    print(np.shape(output))

    # dump as wav
    wav.write("output.wav", SB_sr, output)

