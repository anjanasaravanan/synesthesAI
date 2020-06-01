"""Generating input 'music' vectors for generator."""

import librosa
import numpy as np
import os

sample_rate = 22050
noise_dim = 100

def batch(data_file, dimension):
    batched_data = []
    new_length = len(data_file)//dimension
    for i in range(new_length):
        batched_data.append(data_file[i*dimension:(i+1)*dimension])
    return batched_data

def preprocessing():
    data_directory = 'data/wav_files'
    # data/wav_files contains WAV data from the MAESTRO dataset
    test_files = os.listdir(data_directory)[0:15]

    music_vectors = []
    for test_file in test_files:
        print('Extracting from', test_file)
        wav_data, sr = librosa.load(os.path.join(data_directory, test_file))
        batched_wav_data = np.array(batch(wav_data, noise_dim))
        music_vectors.extend(batched_wav_data)

    np.save('music_vectors', music_vectors)
    print('Data saved to file!')

    print(np.array(music_vectors).shape)

preprocessing()

def postprocessing():
    data_dir = 'data/post'
    test_file = os.listdir(data_dir)[0]
    wav_data, sr = librosa.load(os.path.join(data_dir, test_file))
    batched_wav_data = np.array(batch(wav_data, noise_dim))

    np.save('postprocessing/test1.npy', batched_wav_data)
