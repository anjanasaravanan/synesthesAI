"""Generating color swaths to train our network on."""

import numpy as np
import random

import matplotlib.pyplot as plt


im_dim = 28

def generate_swath(rgb_vector):
    swath = np.zeros((im_dim, im_dim, 3))
    for i in range(im_dim):
        for j in range(im_dim):
            swath[i][j] = rgb_vector

    return swath


colors = []
for r in range(0, 255, 10):
    for g in range(0, 255, 10):
        for b in range(0, 255, 10):
            new_color = [r/255, g/255, b/255]
            new_swath = generate_swath(new_color)
            colors.append(new_swath)

np.save('colors', colors)
# number of colors generated: 17576
