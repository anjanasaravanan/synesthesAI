"""
Do k-means clustering on a bunch of input WAV data, THEN fit the data to the
chopin scherzo.

TRYING WITH MFCCS rather than RMS. gah

"""

import librosa
import numpy as np
import os

from sklearn.cluster import KMeans

from IPython import display
import imageio

import moviepy.editor as mpy

MODE = 'mfcc'

input_dir = 'input_wavs'

im_dim = 140

anim_file = 'test.gif'

test_file = 'chp_op31.wav'

sample_rate = 32768


def batch(array, batch_size):
    batched_array = []
    for i in range(0, len(array), batch_size):
        batched_array.append(array[i:i+batch_size])
    return batched_array


def kmeans_fit():

    input_red, input_green, input_blue = [], [], []

    for input_file in os.listdir(input_dir):
        print('Extracting from', input_file)

        input_wav = librosa.load(os.path.join(input_dir, input_file), sr=sample_rate)[0]

        print('Original WAV length is', len(input_wav))
        if MODE == 'chromagram':
            input_feature = librosa.feature.chroma_cens(input_wav, n_chroma=3, sr=sample_rate)
        elif MODE == 'mfcc':
            input_feature = librosa.feature.mfcc(input_wav, n_mfcc=3, sr=sample_rate)

        temp_red = batch(input_feature[0], 64)
        temp_green = batch(input_feature[1], 64)
        temp_blue = batch(input_feature[2], 64)

        temp_red.pop()
        temp_green.pop()
        temp_blue.pop()

        input_red.extend(temp_red)
        input_green.extend(temp_green)
        input_blue.extend(temp_blue)


    kmeans_red = KMeans(n_clusters=10)
    kmeans_red.fit(input_red)

    kmeans_green = KMeans(n_clusters=10)
    kmeans_green.fit(input_green)

    kmeans_blue = KMeans(n_clusters=10)
    kmeans_blue.fit(input_blue)

    print('K-means fitting complete.')

    return kmeans_red, kmeans_green, kmeans_blue

def generate_predictions(kmeans_red, kmeans_green, kmeans_blue):

    raw_wav, sr = librosa.load(test_file, sr=sample_rate)

    print('Test input loaded.')

    if MODE == 'chromagram':
        test_feature = librosa.feature.chroma_cens(raw_wav, n_chroma=3, sr=sample_rate)
    elif MODE == 'mfcc':
        test_feature = librosa.feature.mfcc(raw_wav, n_mfcc=3, sr=sample_rate)
    # current hop length is 512, or 32786//64. 64 chroma per second.

    # we can batch up the chromagram for each bin into segments of 64.

    test_red = batch(test_feature[0], 64)
    test_green = batch(test_feature[1], 64)
    test_blue = batch(test_feature[2], 64)


    test_red.pop()
    test_green.pop()
    test_blue.pop()

    test_red = np.array(test_red).astype(np.float64)
    test_green = np.array(test_green).astype(np.float64)
    test_blue = np.array(test_blue).astype(np.float64)

    red_predictions = [kmeans_red.predict(red_elem.reshape(1, -1))[0] for red_elem in test_red]
    green_predictions = [kmeans_green.predict(green_elem.reshape(1, -1))[0] for green_elem in test_green]
    blue_predictions = [kmeans_blue.predict(blue_elem.reshape(1, -1))[0] for blue_elem in test_blue]

    print('Predictions generated.')

    zipped_predictions = zip(red_predictions, green_predictions, blue_predictions)

    generated_colors = []

    for r, g, b in zipped_predictions:
        scaled_r, scaled_g, scaled_b = r*25, g*25, b*25
        generated_colors.append((scaled_r, scaled_g, scaled_b))

    return generated_colors

def make_gif(generated_colors):
    print('Compiling images...')
    writer = imageio.get_writer(anim_file, mode='I', fps=1)

    for color in generated_colors:
        img = np.zeros((im_dim, im_dim, 3))
        for x in range(im_dim):
            for y in range(im_dim):
                img[x][y] = color

        writer.append_data(img)

    display.Image(filename=anim_file)

def compile_video():
    print('Compiling video and audio...')
    song = mpy.AudioFileClip(test_file)
    clip = mpy.VideoFileClip(anim_file)
    clip = clip.set_audio(song)

    clip.write_videofile('test_video.mp4', codec='mpeg4')


red_clusters, green_clusters, blue_clusters = kmeans_fit()
color_data = generate_predictions(red_clusters, green_clusters, blue_clusters)
make_gif(color_data)
compile_video()
