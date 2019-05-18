import numpy as np
import utils
import os
from glob import glob

def target_music(filepath, outputpath):
    # read wav
    y, sr = read_wav(filepath)
    if sr == 44100:
        return y
    
    clips = cut_into_clips(y, sr, ms=30000, padding=True)

    # output file
    save_npy(outputpath, clips)

    return clips
    
def sound_bank(filepath, outputpath, ms=1000):
    # read wav
    y, sr = read_wav(filepath)
    if sr == 44100:
        return y
    
    clips = cut_into_clips(y, sr, ms=ms)

    # output file
    save_npy(outputpath, clips)

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
        
if __name__ == '__main__':
    # get file path list
    files = glob('dataset/wav/GTZAN/*/*.wav')
    
    for filepath in files:
        outputpath = filepath.split('/', 2)[2].split('.')[0]+'.npy'
        target_music(filepath, 'dataset/target_music/' + outputpath)
        sound_bank(filepath, 'dataset/sound_bank/' + outputpath)
    
    
