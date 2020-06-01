"""Creating a video out of the results!"""

import cv2
import numpy as np

from IPython import display
import imageio

import moviepy.editor as mpy

import ffmpy


def write_gif():

    img_data = np.load('img_array.npy', allow_pickle=True)
    anim_file = 'test.gif'

    writer = imageio.get_writer(anim_file, mode='I', fps=1)

    for img in img_data:
        writer.append_data(img)

    display.Image(filename=anim_file)


def reformat_gif():
    song = mpy.AudioFileClip('chp_op31.wav')
    clip = mpy.VideoFileClip('test.gif')
    clip = clip.set_audio(song)

    clip.write_videofile('myvideo.mp4', codec='mpeg4')

#reformat_gif()
