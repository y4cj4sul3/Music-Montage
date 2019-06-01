import numpy as np
import utils
from scipy.io import wavfile as wav
from scipy.stats import pearsonr
import features

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
    #return np.mean(np.power(a-b, 2))
    cor, _ = pearsonr(a, b)
    return cor

def synthesize(symbolic, SB):
    output = np.array([])

    for idx in symbolic:
        output = np.append(output, SB[int(idx)])

    return output

def dumpWav(audio, params, filename='output/output.wav'):
    with wave.open(filename, 'w') as wave_file:
        wave_file.setparans(params)

def featureExtraction(input_audio):
    pass

if __name__ == '__main__':

    from argparse import ArgumentParser
    
    # argument parser
    parser = ArgumentParser()
    parser.add_argument('-t', '--target', dest='target', default='', help='target music')
    parser.add_argument('-s', '--sound-bank', dest='sound_bank', default='', help='sound bank')
    parser.add_argument('-o', '--output', dest='output', default='', help='output path')
    args = parser.parse_args()
    
    # parameters
    target_filepath = args.target
    SB_filepath = args.sound_bank
    output_path = args.output
    
    assert target_filepath != '', 'You must give a target (use -t or --target)'
    assert SB_filepath != '', 'You must give a sound bank (use -s or --sound-bank)'
        
    '''
    load data
    '''
    # load target file
    target_filetype = target_filepath.split('.')[-1]
    if target_filetype == 'npy':
        target = np.load(target_filepath)
        target_sr = 44100
    elif target_filetype == 'wav':
        target_sr, target = utils.read_wav(target_filepath)
        target = [target]
    else:
        assert True, 'Unknown target file type'
    
    # load sound bank
    SB_filetype = SB_filepath.split('.')[-1]
    if SB_filetype == 'npy':
        SB = np.load(SB_filepath)
        SB_sr = 44100
    elif SB_filetype == 'wav':
        SB_sr, sound_source = utils.read_wav(SB_filepath)
        # ms per beat
        mspb = int(60000//features.getTempo(SB_filepath))
        print('mspb:', mspb)
        SB = utils.genSoundBank(sound_source, SB_sr, mspb)
    else:
        assert True, 'Unknown sound bank file type'
    
    
    if target_sr != SB_sr:
        print("[Warning]: sample rate doesn't match")

    '''
    Extract Feature
    '''
    
    '''
    Pattern Matching
    '''
    print(np.shape(target))
    symbolic = np.array([])
    for target_clip in target:
        sym = patternMatching(target_clip, SB)
        print(sym)
        symbolic = np.concatenate((symbolic, sym))
    print(np.shape(symbolic))
    print(symbolic)

    '''
    Synthesize
    '''
    output = synthesize(symbolic, SB)
    print(np.shape(output))

    # dump as wav
    if output_path == '':
        target_name = target_filepath.split('/')[-1].split('.')[0]
        SB_name = SB_filepath.split('/')[-1].split('.')[0]
        output_path = 'output/' + target_name + '_' + SB_name + '.wav'
    wav.write(output_path, SB_sr, output)
    print('Save file: ' + output_path)

