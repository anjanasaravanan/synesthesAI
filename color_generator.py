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


#colors = []
#for r in range(0, 255, 10):
#    for g in range(0, 255, 10):
#        for b in range(0, 255, 10):
#            new_color = [r/255, g/255, b/255]
#            new_swath = generate_swath(new_color)
#            colors.append(new_swath)

#np.save('colors', colors)
# number of colors generated: 17576

#colors = np.load('colors.npy')
#num_colors = len(colors)


def generate_palette(rgb_vectors):

    palette = np.zeros((im_dim, im_dim, 3))

    for i in range(im_dim):
        for j in range(im_dim):
            if i<=im_dim//2 and j<=im_dim//2:
                palette[i][j] = rgb_vectors[0]
            elif i>im_dim//2 and j<=im_dim//2:
                palette[i][j] = rgb_vectors[1]
            elif i<=im_dim//2 and j>im_dim//2:
                palette[i][j] = rgb_vectors[2]
            else:
                palette[i][j] = rgb_vectors[3]


    return palette



#palettes = []
#
#for i in range(num_colors):
#    rgb_vectors = []
#    for j in range(4):
#        rgb_vectors.append(random.choice(colors)[0][0])
#    new_palette = generate_palette(rgb_vectors)
#    palettes.append(new_palette)

#print('Number of palettes generated is', len(palettes))
#np.save('palettes', palettes)

palettes = np.load('palettes.npy')
plt.imshow(palettes[50])
plt.show()
