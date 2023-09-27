# Packages, environment, and policies
import os, json
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from setup import *
distributor = tf.distribute.MirroredStrategy()
tf.keras.mixed_precision.set_global_policy("mixed_float16")

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
    classifier_optimizer = tf.keras.optimizers.Adam() 
    classifier_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(classifier_optimizer)

# Define function for calculating and applying perturbations
cl_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
def fgsm(imgs, labels, epsilon=tf.constant(0.01)):
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        predictions = classifier(input_image)
        loss = cl_loss_fn(labels, predictions)

    # Get the gradients of the loss w.r.t to the input image.
    gradients = tape.gradient(loss, imgs)
    # Get the sign of the gradients to create the perturbation
    signed_gradients = tf.sign(gradients)
    return imgs + epsilon*signed_gradients

# Define the per-batch training procedure
ae_loss_fn = tf.keras.losses.BinaryCrossentropy()
def train_step(data):
    # Split images and labels
    imgs, labels = data

    # Get adversarial examples
    adv_imgs = fgsm(imgs)

    # Watch variables
    with tf.GradientTape(persistent=True) as tape:
        # Get latent representation of adversarial examples
        z = encoder(adv_imgs)
        # Reconstruct to originals
        reconstructed = decoder(z)
        # Classify
        predictions = classifier(reconstructed)

        # Calculate loss for classifier and prevent overflow
        classifier_loss = cl_loss_fn(predictions, labels)
        classifier_loss = classifier_optimizer.get_scaled_loss(classifier_loss)
        # Calculate loss for encoder and decoder and prevent overflow
        ae_loss = ae_loss_fn(reconstructed, adv_batch)
        ae_loss = ae_optimizer.get_scaled_loss(ae_loss)

    # Backpropagate the ae loss
    ae_gradients = tape.gradient(ae_loss, ae.trainable_variables)
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
def distributed_train_step(data):
    per_replica_losses = distributor.run(train_step, args=(data,))
    return distributor.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Train batches
history = {}
for i, batch in enumerate(data):
    # Stop training at some point
    if i==1e4:
        break
    # Train the networks
    losses = distributed_train_step(batch)
    history[i] = {key: float(value) for key, value in losses.items()}
    # Periodically update
    if (i+1)%100==0:
        # Save progress
        json.dump(history, open("./history.json", mode="w"))
        # Save weights
        generator.save("generator.keras")
        discriminator.save("discriminator.keras")