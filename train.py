# Packages, environment, and policies
import os, json
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda" # Replace with correct location
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from models import *
from utils import *
distributor = tf.distribute.MirroredStrategy()
tf.keras.mixed_precision.set_global_policy("mixed_float16")

def build_models():
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
    return encoder, decoder, ae, classifier, ae_optimizer, classifier_optimizer

# Define the per-batch training procedure
def train_step(encoder, decoder, ae, classifier, batch, adv_attack, epsilon, clf_loss_fn, ae_loss_fn, ae_optimizer, classifier_optimizer):
    """
    Takes one batch of data and trains the models for one step.
    """
    # Split images and labels
    imgs, labels = batch

    # Get adversarial examples
    adv_imgs = adv_attack(classifier, clf_loss_fn, imgs, labels, epsilon)

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
    ae_gradients, alpha_gradient = tape.gradient(ae_loss, ae.trainable_variables)
    ae_gradients = ae_optimizer.get_unscaled_gradients(ae_gradients)
    ae_optimizer.apply_gradients(zip(ae_gradients, ae.trainable_variables))
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
def distributed_train_step(
    encoder, 
    decoder, 
    ae,
    classifier, 
    batch, 
    adv_attack, 
    epsilon, 
    clf_loss_fn, 
    ae_loss_fn, 
    ae_optimizer, 
    classifier_optimizer
    ):
    """
    Takes one batch of data and trains the models for one step.
    """
    per_replica_losses = distributor.run(train_step, args=(
        encoder, 
        decoder, 
        ae,
        classifier, 
        batch, 
        adv_attack, 
        epsilon, 
        clf_loss_fn, 
        ae_loss_fn, 
        ae_optimizer, 
        classifier_optimizer
    ))
    return distributor.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Run the script
def run(dataset, adv_attack, epsilon):
    # Build models
    encoder, decoder, ae, classifier, ae_optimizer, classifier_optimizer = build_models()
    # Define some objects
    clf_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    ae_loss_fn = SSIM_L1()

    # Train batches
    history = {}
    for i, batch in enumerate(dataset):
        # Stop training at some point
        if i==2e3:
            break
        # Train the networks
        losses = distributed_train_step(
            encoder, 
            decoder, 
            ae,
            classifier, 
            batch, 
            adv_attack, 
            epsilon, 
            clf_loss_fn, 
            ae_loss_fn, 
            ae_optimizer, 
            classifier_optimizer
        )
        history[i] = {key: float(value) for key, value in losses.items()}
        # Periodically update
        if (i+1)%100==0:
            # Save progress
            json.dump(history, open("./history.json", mode="w"))
            # Save weights
            encoder.save("encoder.keras")
            decoder.save("decoder.keras")
            classifier.save("classifier.keras")