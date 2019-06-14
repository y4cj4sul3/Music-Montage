import numpy as np
import utils
import os
from glob import glob
from pathlib import Path
from librosa import stft
from librosa.feature import chroma_stft

def gen_clips(filepath, outputpath, ms=1000, padding=True, sr_th=0):
    # read wav
    y, sr = read_wav(filepath)
    print(np.shape(y))
    
    # filter sr
    if(sr_th > 0):
        if(sr != sr_th):
            print('sr doesn\'t match:', sr)
            return y
    
    # extract feature
    n_fft_ms = 40  # 1/10 clip size
    hop_length_ms = 10    # 1/4 window size
    n_fft = int(n_fft_ms*sr//1000)
    hop_length = int(hop_length_ms*sr//1000)
    print(n_fft, hop_length)
    spectrum = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    chroma = chroma_stft(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length)
    print(np.shape(spectrum))
    print(np.shape(chroma))
        
    # cut into clips
    clips = cut_into_clips(y, sr, ms=ms, padding=padding)
    spectrum_clips = cut_into_clips_sc(spectrum, sr, ms=ms, hop_length_ms=hop_length_ms)
    chroma_clips = cut_into_clips_sc(chroma, sr, ms=ms, hop_length_ms=hop_length_ms)
    
    # output file
    save_npy(outputpath % 'raw', clips)
    save_npy(outputpath % 'spectrum', spectrum_clips)
    save_npy(outputpath % 'chroma', chroma_clips)

    return clips

def read_wav(filepath):
    # read wav
    print('[Log] read: ' + filepath)
    sr, y = utils.read_wav(filepath)

    # check sample rate
    if sr != 44100:
        print('[Warnning] sampling rate is', sr)
        
    y = np.array(y)
    
    return y, sr
    
def save_npy(filepath, y):
    print('[Log] shape:') 
    print(np.shape(y))
    print('[Log] save file: ' + filepath)
    np.save(filepath, y)
    
def cut_into_clips(y, sr, ms=1000, padding=False):
    
    total_len = np.shape(y)[0]
    # clip length
    clip_len = (sr*ms)//1000
    num_clips = total_len//clip_len
    align_len = num_clips*clip_len

    clips = np.reshape(y[:align_len], (num_clips, clip_len))
    if padding and align_len != total_len:
        # zero pad last clip
        last_clip = [np.pad(y[align_len:], (0, clip_len-(total_len-align_len)), 'constant')]
        clips = np.concatenate((clips, last_clip))
    
    return clips
    
def cut_into_clips_sc(y, sr, ms=1000, hop_length_ms=0):
    
    total_len = np.shape(y)[1]
    feature_len = np.shape(y)[0]
    # clip length
    clip_len = ms//hop_length_ms
    num_clips = total_len//clip_len
    align_len = num_clips*clip_len*feature_len
    
    clips = []
    for i in range(0, (total_len//clip_len)*clip_len, clip_len):
        clips.append(y[:, i:i+clip_len])
    print(np.shape(np.array(clips)))
    return clips
    '''
    clips = np.reshape(y[:align_len], (num_clips, clip_len, feature_len))
    if padding and align_len != total_len*feature_len:
        # zero pad last clip
        last_clip = [np.pad(y[align_len:], (0, clip_len-(total_len-align_len)), 'constant')]
        clips = np.concatenate((clips, last_clip))
    
    return clips
    '''
        
if __name__ == '__main__':
    
    from argparse import ArgumentParser
    
    # argument parser
    parser = ArgumentParser()
    parser.add_argument('-ms', '--million-sec', dest='ms', default=1000, type=int, help='clip size')
    parser.add_argument('-o', '--output', dest='output', default='dataset/clips', help='output folder name')
    parser.add_argument('-f', '--feature', dest='feature', default='raw', help='extracted feature')
    parser.add_argument('-i', '--input', dest='input', default='dataset/wav/', help='input folder')
    parser.add_argument('-sr', '--sample-rate', dest='sr', default=44100, type=int, help='sampling rate')
    args = parser.parse_args()
    
    ms = args.ms
    outputfolder = args.output
    feature = args.feature
    inputfolder = args.input
    sr_th = args.sr
    
    input_len = len(inputfolder)
    print(input_len)
    print('Input folder: ' + inputfolder)
    
    #outputfolder = '%s_%s_%i_%i' % (outputfolder, feature, ms, sr_th)
    outputfolder = ('%s_' % outputfolder) + '%s' + ('_%i_%i' % (ms, sr_th))
    print('Output folder: ' + outputfolder)
    
    
    # get file path list
    files = Path(inputfolder).glob('**/*.wav')
    
    for filepath in files:
        print(filepath)
        filepath_str = str(filepath)
        
        subfolder = filepath_str[input_len:].split('.')[0]+'.npy'
        #print(subfolder)
        outputpath = os.path.join(outputfolder, subfolder)
        #print(outputpath)
        
        # make folder
        outputfolderpath = outputpath.split('/')[:-1]
        outputfolderpath = os.path.join(*outputfolderpath)
        for feature in ['raw', 'spectrum', 'chroma']:
            if not os.path.isdir(outputfolderpath % feature):
                os.makedirs(outputfolderpath % feature)
        #print(outputfolderpath)
        
        gen_clips(filepath_str, outputpath, ms=ms, sr_th=sr_th)
        
        
