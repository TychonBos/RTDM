# Packages, environment, and policies
import os, json
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda" # Replace with correct location
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from models import *
from utils import *
distributor = tf.distribute.MirroredStrategy()
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Get data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Convert to tf dataset 
data_train = dataset_from_arrays(X_train, y_train)
data_test= dataset_from_arrays(X_test, y_test)

# Build models distributed and use mixed-precision loss scaling
with distributor.scope():
    # Encoder
    encoder = build_encoder()
    # Decoder
    decoder = build_decoder()
    # Combine because why not
    ae = tf.keras.Sequential([encoder, decoder], name="Autoencoder")
    # Share an optimizer
    ae_optimizer = tf.keras.optimizers.Adam() 
    ae_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(ae_optimizer)

    # Classifier
    classifier = build_classifier()
    # Optimizer
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) 
    classifier_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(classifier_optimizer)

# Define the per-batch training procedure
clf_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
alpha = tf.Variable(.84, trainable=True, name="alpha")
ae_loss_fn = SSIM_L1(alpha)
def train_step(batch):
    """
    Takes one batch of data and trains the models for one step.
    """
    # Split images and labels
    imgs, labels = batch

    # Get adversarial examples
    adv_imgs = fgsm(classifier, clf_loss_fn, imgs, labels)

    # Watch variables
    with tf.GradientTape(persistent=True) as tape:
        # Get latent representation of adversarial examples
        z = encoder(adv_imgs)
        # Reconstruct to originals
        reconstructed = decoder(z)
        # Classify
        predictions = classifier(reconstructed)

        # Calculate loss for classifier and prevent overflow
        classifier_loss = clf_loss_fn(predictions, labels)
        classifier_loss = classifier_optimizer.get_scaled_loss(classifier_loss)
        # Calculate loss for encoder and decoder and prevent overflow
        ae_loss = ae_loss_fn(reconstructed, imgs)
        ae_loss = ae_optimizer.get_scaled_loss(ae_loss)

    # Backpropagate the ae loss
    ae_gradients, alpha_gradient = tape.gradient(ae_loss, [ae.trainable_variables, alpha])
    ae_gradients = ae_optimizer.get_unscaled_gradients(ae_gradients)
    alpha_gradient = ae_optimizer.get_unscaled_gradients([alpha_gradient])
    ae_optimizer.apply_gradients(zip(ae_gradients, ae.trainable_variables))
    alpha.assign_sub(tf.constant(1e-4, name="lr")*tf.squeeze(alpha_gradient))
    # Ensure alpha in [0,1]
    if tf.greater(alpha,1.):
        alpha.assign(1.)
    elif tf.less(alpha,0.):
        alpha.assign(0.)
    # Backpropagate the classifier loss
    classifier_gradients = tape.gradient(classifier_loss, classifier.trainable_variables)
    classifier_gradients = classifier_optimizer.get_unscaled_gradients(classifier_gradients)
    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))

    # Return progress
    return {
        "classifier_loss": tf.reduce_mean(classifier_loss),
        "ae_loss": tf.reduce_mean(ae_loss), 
    }

# Convert to distributed train step
@tf.function(reduce_retracing=True)
def distributed_train_step(batch):
    """
    Takes one batch of data and trains the models for one step.
    """
    per_replica_losses = distributor.run(train_step, args=(batch,))
    return distributor.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Train batches
history = {}
for i, batch in enumerate(data_train):
    # Stop training at some point
    if i==2e3:
        break
    # Train the networks
    losses = distributed_train_step(batch)
    history[i] = {key: float(value) for key, value in losses.items()}
    # Periodically update
    if (i+1)%100==0:
        # Save progress
        json.dump(history, open("./history.json", mode="w"))
        # Save weights
        encoder.save("encoder.keras")
        decoder.save("decoder.keras")
        classifier.save("classifier.keras")