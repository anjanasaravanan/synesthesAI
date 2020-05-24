"""Misadventures with GAN, resume!"""



"""
Notes:
- currently training on: SWATHS.
- current noise input: RANDOM.

Notes on results:
- swath training, no convergence even after 150 epochs, trying 200
- 200: seeing convergence. trying 250!

Current:
- training the generator on raw (normalized?) wav data rather than random
    normalized 'noise'


"""
import tensorflow as tf
from tensorflow.keras import layers
import time # w o w

import numpy as np
import matplotlib.pyplot as plt

import os

from IPython import display

import librosa


noise_mode = 'music'
EPOCHS = 250
noise_dim = 100
num_examples_to_generate = 16
im_dim = 28

BATCH_SIZE = 500 # increasing to work towards convergence
BUFFER_SIZE = 17576 # exact size of colors array, full shuffling

train_images = np.load('data/colors.npy')
# 17576 colors.

if noise_mode == 'music':
    music_vectors = iter(np.load('preprocessing/music_vectors.npy', allow_pickle=True))

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print('DATASET LOADED.')

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 3]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()

#noise = tf.random.normal([1, noise_dim])
#generated_image = generator(noise, training=False)


#generated_image = tf.linalg.normalize(generated_image, axis=3)[0]
#generated_image = tf.math.abs(generated_image)
#plt.imshow(generated_image[0])
#plt.show()

discriminator = make_discriminator_model()
#decision = discriminator(enhanced_image)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

if noise_mode == 'random':
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
elif noise_mode == 'music':
    seed = np.array([next(music_vectors) for _ in range(num_examples_to_generate)])

@tf.function
def train_step(images):
    if noise_mode == 'random':
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
    elif noise_mode == 'music':
        noise = np.array([next(music_vectors) for _ in range(BATCH_SIZE)])

    print('step,')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        generated_images = tf.linalg.normalize(generated_images, axis=3)[0]
        generated_images = tf.math.abs(generated_images)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch+1, seed)

        print('Time for epoch {} is {} seconds.'.format(epoch+1, time.time()-start))

    generator.save_weights('SAVED_WEIGHTS/generator')

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = tf.linalg.normalize(predictions, axis=3)[0] # write this function.
    predictions = tf.math.abs(predictions)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig('training_examples/image_at_epoch_{:04}.png'.format(epoch))
    plt.close()

#print('TRAINING INITIALIZED.')
#train(train_dataset, EPOCHS)

def music_tings(new_noise, loaded_gen):

    temp_result = loaded_gen(new_noise)
    edited_result = tf.linalg.normalize(temp_result, axis=3)[0]
    edited_result = tf.math.abs(edited_result)

    elem_counter = 0
    for elem in edited_result:
        elem_counter += 1

        fig = plt.figure()
        plt.imshow(elem)
        plt.axis('off')
        plt.savefig('test/images/image_elem_{:03}.png'.format(elem_counter))
        plt.close()

def post_processing():

    generator.load_weights('SAVED_WEIGHTS/generator')

    test_file = os.listdir('data/post')[0]




    noise_sampler = np.load('postprocessing/test1.npy', allow_pickle=True)

    hop_length = len(noise_sampler)//num_examples_to_generate

    sample_indices = range(0, len(noise_sampler), hop_length)

    # sample_indices = [np.random.choice(range(len(noise_sampler))) for _ in range(num_examples_to_generate)]

    test_input = np.array([noise_sampler[i] for i in sample_indices])

    test_predictions = generator(test_input, training=False)
    test_predictions = tf.linalg.normalize(test_predictions, axis=3)[0]
    test_predictions = tf.math.abs(test_predictions)

    num_outputs = len(test_predictions)//num_examples_to_generate

    for i in range(num_outputs):
        fig = plt.figure(figsize=(4, 4))

        for j in range(num_examples_to_generate):
            plt.subplot(4, 4, j+1)
            plt.imshow(test_predictions[j+i*num_examples_to_generate, :, :, :])
            plt.axis('off')

        plt.savefig('post_training_examples/test1_{:03}.png'.format(i))
        #plt.show()
        plt.close()

post_processing()

#from sklearn.preprocessing import normalize


#generator.load_weights('SAVED_WEIGHTS/01swaths/generator')
#new_noise = np.load('test/track02_batched_wav.npy')
