# An Autoencoder-based Defense Against Adversarial Attacks

## Implementation
In this implementation, instead of pretraining a classifier and then training an autoencoder, the autoencoder and classifier are trained simultaneously. 

## Setup
Create a new conda environment by running `conda env create -f environment.yml`. To load the trained models and run a simple experiment, follow the code below. Note: The models included in this repository were trained on the CIFAR-10 dataset with IFGSM and ε=0.06.
```
from models import Inception, downsampler, upsampler
from utils import ifgsm, preprocess, BATCH_SIZE
import tensorflow as tf

# Load models
with tf.keras.utils.custom_object_scope({
    "Inception": Inception,
    "downsampler": downsampler,
    "upsampler": upsampler
    }):
    ae = tf.keras.models.load_model("autoencoder.keras")
    classifier = tf.keras.models.load_model("classifier.keras")

# Load data
*_, (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
x, y = preprocess(X_test, y_test)

# Get predictions (change slice to anything)
x_adv = ifgsm(classifier, x[:BATCH_SIZE], epsilon=.06) 
y_pred_adv = classifier.predict(x_adv)
x_pur = ae.predict(x_adv)
y_pred_pur = classifier.predict(x_pur)
```