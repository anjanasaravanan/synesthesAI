"""
Do k-means clustering on a bunch of input WAV data, THEN fit the data to the
chopin scherzo.

"""

import librosa
import numpy as np
import os

from sklearn.cluster import KMeans

from IPython import display
import imageio

import moviepy.editor as mpy

def batch(array, batch_size):
    batched_array = []
    for i in range(0, len(array), batch_size):
        batched_array.append(array[i:i+batch_size])
    return batched_array

sample_rate = 32768

input_dir = 'input_wavs'

input_red, input_green, input_blue = [], [], []

for input_file in os.listdir(input_dir):
    print('Extracting from', input_file)

    input_wav = librosa.load(os.path.join(input_dir, input_file), sr=sample_rate)[0]
    input_chromagram = librosa.feature.chroma_cens(input_wav, n_chroma=3, sr=sample_rate)

    temp_red = batch(input_chromagram[0], 64)
    temp_green = batch(input_chromagram[1], 64)
    temp_blue = batch(input_chromagram[2], 64)

    temp_red.pop()
    temp_green.pop()
    temp_blue.pop()

    input_red.extend(temp_red)
    input_green.extend(temp_green)
    input_blue.extend(temp_blue)

#input_file = 'input_wavs/input_wav_01.wav'
#input_wav, sr = librosa.load(input_file, sr=sample_rate)
#input_chromagram = librosa.feature.chroma_cens(input_wav, n_chroma=3, sr=sample_rate)

#input_red = batch(input_chromagram[0], 64)
#input_green = batch(input_chromagram[1], 64)
#input_blue = batch(input_chromagram[2], 64)

#input_red.pop()
#input_green.pop()
#input_blue.pop()

kmeans_red = KMeans(n_clusters=10)
kmeans_red.fit(input_red)

kmeans_green = KMeans(n_clusters=10)
kmeans_green.fit(input_green)

kmeans_blue = KMeans(n_clusters=10)
kmeans_blue.fit(input_blue)

print('K-means fitting complete.')


test_file = 'chp_op31.wav'

raw_wav, sr = librosa.load(test_file, sr=sample_rate)

print('Test input loaded.')

#print('raw wav shape', raw_wav.shape)

test_chromagram = librosa.feature.chroma_cens(raw_wav, n_chroma=3, sr=sample_rate)
# current hop length is 512, or 32786//64. 64 chroma per second.

# we can batch up the chromagram for each bin into segments of 64.

test_red = batch(test_chromagram[0], 64)
test_green = batch(test_chromagram[1], 64)
test_blue = batch(test_chromagram[2], 64)


test_red.pop()
test_green.pop()
test_blue.pop()

# now we k-means the shit out of this. try 10 bins first.

red_predictions = [kmeans_red.predict(red_elem.reshape(1, -1))[0] for red_elem in test_red]
green_predictions = [kmeans_green.predict(green_elem.reshape(1, -1))[0] for green_elem in test_green]
blue_predictions = [kmeans_blue.predict(blue_elem.reshape(1, -1))[0] for blue_elem in test_blue]

print('Predictions generated.')

zipped_predictions = zip(red_predictions, green_predictions, blue_predictions)

generated_colors = []

for r, g, b in zipped_predictions:
    scaled_r, scaled_g, scaled_b = r*25, g*25, b*25
    generated_colors.append((scaled_r, scaled_g, scaled_b))

im_dim = 140

anim_file = 'test.gif'

def make_gif():
    print('Compiling images...')
    writer = imageio.get_writer(anim_file, mode='I', fps=1)

    for color in generated_colors:
        img = np.zeros((im_dim, im_dim, 3))
        for x in range(im_dim):
            for y in range(im_dim):
                img[x][y] = color

        writer.append_data(img)

    display.Image(filename=anim_file)

make_gif()

def compile_video():
    print('Compiling video and audio...')
    song = mpy.AudioFileClip(test_file)
    clip = mpy.VideoFileClip(anim_file)
    clip = clip.set_audio(song)

    clip.write_videofile('test_video.mp4', codec='mpeg4')


compile_video()
