# Packages, environment, and policies
import json, tensorflow as tf
from utils import distributor, fgsm, SSIM_L1

# Define the per-batch training procedure
def train_step(classifier, ae, batch, epsilon, clf_loss_fn, ae_loss_fn, ae_optimizer, classifier_optimizer):
    """
    Takes one batch of data and trains the models for one step.
    """
    # Split images and labels
    imgs, labels = batch
    # Apply random jitter (done here to avoid double application or wrong reconstruction)
    imgs_jitter = tf.image.random_brightness(imgs, max_delta=.2)
    imgs_jitter = tf.image.random_contrast(imgs_jitter, lower=-.2, upper=.2)

    # Get adversarial examples
    adv_imgs = fgsm(classifier, imgs_jitter, epsilon)

    # Update autoencoder
    with tf.GradientTape() as tape:
        # Get latent representation and reconstruct to originals
        reconstructed = ae(adv_imgs, training=True)
        # Calculate loss for encoder and decoder and prevent overflow
        ae_loss = ae_loss_fn(reconstructed, imgs)
        ae_loss = ae_optimizer.get_scaled_loss(ae_loss)
    # Backpropagate the ae loss
    ae_gradients = tape.gradient(ae_loss, ae.trainable_variables)
    ae_gradients = ae_optimizer.get_unscaled_gradients(ae_gradients)
    ae_optimizer.apply_gradients(zip(ae_gradients, ae.trainable_variables))

    # Update classifier on both originals and reconstructions
    for x in [imgs_jitter, reconstructed]:
        with tf.GradientTape() as tape:
            # Classify
            predictions = classifier(x, training=True)
            # Calculate loss for classifier and prevent overflow
            classifier_loss = clf_loss_fn(predictions, labels)
            classifier_loss = classifier_optimizer.get_scaled_loss(classifier_loss)
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
    classifier, 
    ae, 
    batch, 
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
        classifier, 
        ae, 
        batch, 
        epsilon, 
        clf_loss_fn, 
        ae_loss_fn, 
        ae_optimizer, 
        classifier_optimizer
    ))
    return distributor.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Run the script
def run(dataset, epsilon, classifier, ae, ae_optimizer, classifier_optimizer):
    """
    Runs the entire training procedure for 1e4 steps.\n
    Args:\n
    \t- dataset: An instance of `tf.data.dataset` with output shape ((None,None,channels),(batch,10))\n
    \t- epsilon: A tf float between 0 and 1, the perturbation level.
    """
    # Define some objects
    clf_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    ae_loss_fn = SSIM_L1()

    # Train batches
    history = {}
    for i, batch in enumerate(dataset):
        # Stop training at some point
        if i==5e4:
            break
        # Train the networks
        losses = distributed_train_step(
            classifier, 
            ae, 
            batch, 
            epsilon, 
            clf_loss_fn, 
            ae_loss_fn, 
            ae_optimizer, 
            classifier_optimizer
        )
        history[i] = {key: float(value) for key, value in losses.items()}
        # Periodically update
        if (i+1)%1e3==0:
            # Save progress
            json.dump(history, open("./history.json", mode="w"))
            # Save weights
            ae.save("autoencoder.keras")
            classifier.save("classifier.keras")