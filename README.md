# An Autoencoder-based Defense Against Adversarial Attacks

## Implementation
In this implementation, instead of pretraining a classifier and then training an autoencoder or vice versa, the autoencoder and classifier are trained simultaneously. 

## Setup
Create a new conda environment by running `conda env create -f environment.yml`. To load the trained models, follow the code below.
```
from models import Inception, downsampler, upsampler

with tf.keras.utils.custom_object_scope({
    'Inception': Inception,
    "downsampler": downsampler,
    "upsampler": upsampler
    }):
    decoder = tf.keras.models.load_model("decoder.keras")
    encoder = tf.keras.models.load_model("encoder.keras")
    classifier = tf.keras.models.load_model("classifier.keras")
```