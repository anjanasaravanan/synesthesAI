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

reformat_gif()




def create_video():
    img_array = []
    for filename in raw_data:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('chp_op31.avi', cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    # 15 frames per second.

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
