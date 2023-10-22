# An Autoencoder-based Defense Against Adversarial Attacks

## Implementation
In this implementation, instead of pretraining a classifier and then training an autoencoder or vice versa, the autoencoder and classifier are trained simultaneously. 

## Setup
Create a new conda environment by running `conda env create -f environment.yml`. To load the trained models and run a simple experiment, follow the code below. Note: The models included in this repository were trained on the CIFAR-10 dataset with IFGSM and ε=0.06.
```
from models import Inception, downsampler, upsampler
from utils import ifgsm, color_dataset_from_arrays
import tensorflow as tf

# Load models
with tf.keras.utils.custom_object_scope({
    "Inception": Inception,
    "downsampler": downsampler,
    "upsampler": upsampler
    }):
    encoder = tf.keras.models.load_model("encoder.keras")
    decoder = tf.keras.models.load_model("decoder.keras")
    classifier = tf.keras.models.load_model("classifier.keras")

# Load data
*_, (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
data_test = color_dataset_from_arrays(X_test, y_test)
x, y = list(data_test.take(1))[0]

# Get predictions
x_adv = ifgsm(classifier, tf.keras.losses.CategoricalCrossentropy(), x, y, epsilon=.06)
y_pred_adv = classifier(x_adv)
x_pur = decoder(encoder(x_adv))
y_pred_pur = classifier(x_pur)
```