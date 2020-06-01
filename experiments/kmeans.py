import librosa
import numpy as np

from sklearn.cluster import KMeans

from IPython import display
import imageio

import moviepy.editor as mpy

def batch(array, batch_size):
    batched_array = []
    for i in range(0, len(array), batch_size):
        batched_array.append(array[i:i+batch_size])
    return batched_array



test_file = 'chp_op31.wav'
sample_rate = 32768

raw_wav, sr = librosa.load(test_file, sr=sample_rate)

print('raw wav shape', raw_wav.shape)

chromagram = librosa.feature.chroma_cens(raw_wav, n_chroma=3, sr=sample_rate)
# current hop length is 512, or 32786//64. 64 chroma per second.

# we can batch up the chromagram for each bin into segments of 64.

red_bin = batch(chromagram[0], 64)
green_bin = batch(chromagram[1], 64)
blue_bin = batch(chromagram[2], 64)


red_bin = red_bin[:len(red_bin)-1]
green_bin = green_bin[:len(green_bin)-1]
blue_bin = blue_bin[:len(blue_bin)-1]

# now we k-means the shit out of this. try 10 bins first.

kmean_red = KMeans(n_clusters=10)
kmean_red.fit(red_bin)

kmean_green = KMeans(n_clusters=10)
kmean_green.fit(green_bin)

kmean_blue = KMeans(n_clusters=10)
kmean_blue.fit(blue_bin)


zipped_raw = zip(kmean_red.labels_, kmean_green.labels_, kmean_blue.labels_)

generated_colors = []

for r, g, b in zipped_raw:
    scaled_r, scaled_g, scaled_b = r*25, g*25, b*25
    generated_colors.append((scaled_r, scaled_g, scaled_b))

print(generated_colors)
print(len(generated_colors))
print(generated_colors[0])

im_dim = 140

anim_file = 'test.gif'

def make_gif():
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
    song = mpy.AudioFileClip(test_file)
    clip = mpy.VideoFileClip(anim_file)
    clip = clip.set_audio(song)

    clip.write_videofile('test_video.mp4', codec='mpeg4')


compile_video()
