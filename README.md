# synesthesAI
A neural network framework to visualize sound.
Currently in the development phase.


CONCEPT

The idea behind this project is to mimic the phenomenon of synesthesia, which allows people to create visual representations
for sounds and music. A generative adversarial network (GAN) was used to this end. The generator network is trained with an input of 
batched WAV data collected from the MAESTRO dataset, and the discriminator is trained on 28 by 28 color "swaths", with each pixel in the
swath having the same RGB value. 


CURRENT RESULTS

The GAN was trained for 250 epochs with a batch size of 500 on 17500000 WAV "music vectors". After training, WAV music vectors from 
Chopin's Waltz in B-flat minor were inputted to the generator, and the results were evenly sampled to produce a chronological
"color map" of the piece. The results are below. 

![results](results/test1_even_dist.png)


UPDATE

A very low-quality video has been produced with an abysmally low 1 frame per second. Much work to be done, including but not limited to:
increasing the image quality, increasing the frame rate, improving image/song alignment, exploring abberations. Most importantly,
the GAN should now be run on features extracted from the music rather than raw WAV data.

![download](results/myvideo.mp4)

FUTURE WORK

- Research synesthesia as it manifests in humans for potential relevant insights
- Switch from color "swaths" to gradients or multi-color palettes
- Extract features from WAV data before feeding into network
- Evaluate current results, particularly non-uniform outputs


