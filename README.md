# synesthesAI
A neural network framework to visualize sound.
Currently in the development phase.

The idea behind this project is to mimic the phenomenon of synesthesia, which allows people to create visual representations
for sounds and music. The generator network is trained with an input of batched WAV data collected from the MAESTRO dataset, and 
the discriminator is trained on 28 by 28 color "swaths", with each pixel in the swath having the same RGB value. 

Current results. 
The GAN was trained for 250 epochs with a batch size of 500 on 17500000 WAV "music vectors". After training, WAV music vectors from 
Chopin's Waltz in B-flat minor were inputted to the generator, and the results were evenly sampled to produce a 4 by 4 "color map" of
the piece. The results are below.

[results](results/test1_even_dist)
