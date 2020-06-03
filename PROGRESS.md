

FUTURE WORK
The next logical step is to k-means cluster with more WAV data, rather than focusing only one one song. With a larger variety of music inputs, hopefully k-means clustering will yield more general results and therefore a smoother transition between generated colors. Also, it would be nice to increase the frame rate of the video results.


EARLY JUNE
I decided to simplify my efforts and while working on an Arduino color lamp, I was hit with inspiration. The color lamp works by using three phototransistors with red, green, and blue gen placed on top of them respectively to filter out light of a certain color. Then a LED is lit up according to the current generated in each phototransistor. Splitting the data into three channels before processing became an interesting prospect. To do this, I used the LibROSA Python library, which allows users to extract features from WAV data. I extracted a chromogram with 3 bins and assigned the bins to red, green, and blue from lowest frequency to highest (mimicking light). I fed in the Chopin data and now had three channels of 'music data'. Then, departing from the GAN model in the spirit of simplification, I used k-means clustering (from the scikit library) with 10 clusters for each channel. After generating the labels for each constituent vector, I multiplied the label number by 25 to arrive at a R, G, and B value (each component takes on a value between 0 and 255). After doing this, with some manipulation of the sampling rate used by LibROSA, I was able to generate a video of my results. I find it more promising than my earlier results because 1) the generated results are by design uniform and 2) there appears to be some patterns in the progression of colors.


MAY AND BEFORE
I designed a template for synesthesAI which relied on Generative Adversarial Networks (GANs). The concept I came up with was to train the discriminator on color "blocks", or 28 x 28 RGB squares, and train the generator with an input of WAV data from the MAESTRO dataset. This would, ideally, allow the generator to discover the ideal way of mapping what I'll call 'music vectors' to colors, which is analogous to the phenomenon of synesthesia. 
For my results, I created a 'color map' of Chopin's Scherzo in B-flat minor by feeding its WAV data into the trained generator and sampling evenly from throughout the generated results to produce what is below. I also produced a video, but the results weren't satisfactory. This was for a number of reasons. For one, there was no logical continuity between colors. Also, as seen in the results below, some frames had nonconformities, which suggested that the generator had failed to formulate a mapping of WAV data to pure color. 



