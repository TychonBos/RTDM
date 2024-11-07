# An Autoencoder-based Defense Against Adversarial Attacks

## Implementation
In this implementation, instead of pretraining a classifier and then training an autoencoder, the autoencoder and classifier are trained simultaneously. 

## Setup
Create a new conda environment by running `conda env create -f environment.yml`. To load the trained models and run a simple experiment, follow the code below. Note: The models included in this repository were trained on the Fashion-MNIST dataset.
```
from models import block
from utils import ifgsm, to_dataset
import tensorflow as tf

# Load models
with tf.keras.utils.custom_object_scope({
    "block": block
    }):
    ae = tf.keras.Sequential([
        tf.keras.models.load_model("encoder.keras"),
        tf.keras.models.load_model("decoder.keras")
    ])
    classifier = tf.keras.models.load_model("classifier.keras")

# Load data
*_, (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x, y = list(to_dataset(X_test, y_test).take(1))[0]

# Get predictions
x_adv = ifgsm(classifier, x, epsilon=.3) 
y_pred_adv = classifier.predict(x_adv)
x_pur = ae.predict(x_adv)
y_pred_pur = classifier.predict(x_pur)
```