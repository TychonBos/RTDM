import tensorflow as tf
from models import BATCH_SIZE

# Define function for calculating and applying perturbations
def fgsm(classifier, clf_loss_fn, imgs, labels, epsilon):
    """
    Transforms images into adversarial examples using the Fast Gradient Sign Method.
    Args:
    \t- classifier: The classification model.
    \t- clf_loss_fn: The loss function used for the classifier.
    \t- imgs: The images to be perturbed.
    \t- labels: The corresponding labels.
    \t- epsilon: A perturbation level between 0 and 1.
    """

    # Watch the image
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        predictions = classifier(imgs)
        loss = clf_loss_fn(labels, predictions)

    # Get the gradients of the loss w.r.t to the input image.
    gradients = tape.gradient(loss, imgs)
    # Get the sign of the gradients to create the perturbation
    signed_gradients = tf.sign(gradients)
    return imgs + epsilon*signed_gradients

def ifgsm(classifier, clf_loss_fn, imgs, labels, epsilon, iterations=10, alpha=1.0):
    """
    Transforms images into adversarial examples using the Iterative Fast Gradient Sign Method.
    Args:
    \t- classifier: The classification model.
    \t- clf_loss_fn: The loss function used for the classifier.
    \t- imgs: The images to be perturbed.
    \t- labels: The corresponding labels.
    \t- epsilon: A perturbation level between 0 and 1.
    """

    perturbed_imgs = tf.identity(imgs)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_imgs)
            predictions = classifier(perturbed_imgs)
            loss = clf_loss_fn(labels, predictions)

        gradients = tape.gradient(loss, perturbed_imgs)
        signed_gradients = tf.sign(gradients)
        perturbed_imgs = perturbed_imgs + alpha * signed_gradients
        perturbed_imgs = tf.clip_by_value(perturbed_imgs, imgs - epsilon, imgs + epsilon)
        perturbed_imgs = tf.clip_by_value(perturbed_imgs, 0, 1)  # Ensure pixel values are in [0, 1] range

    return perturbed_imgs

# Define dataset transformations
def dataset_from_arrays(x, y):
    """
    Takes NumPy arrays and uses these to create a `tf.data.Dataset`
    Args:
    \t- x: The images.
    \t- y: The corresponding labels.
    """

    return tf.data.Dataset.from_tensor_slices(
        (x, y)
        ).map(
            # Transform for compatibility with 2d conv
            lambda x, y: (tf.expand_dims(x, axis=-1), y)
        ).map(
            lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y)
        ).map(
            # Resize for decoding stability
            lambda x, y: (tf.image.resize(x, (32,32)), y)
        ).map(
            # One-hot encode labels
            lambda x, y: (x, tf.one_hot(y, depth=10))
        ).batch(BATCH_SIZE).cache().repeat()

# A non-pixelwise loss
class SSIM_L1(tf.keras.losses.Loss): 
    """
    A loss function specialized for autoencoders, introduced in https://arxiv.org/pdf/1511.08861.pdf
    """

    def __init__(self, alpha=.84, **kwargs):
        """
        Initialize the loss object.
        Args:
        \t- alpha: The weighting factor between SSIM and L1. 
        """

        kwargs.pop("reduction", None)
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.l1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.alpha = alpha
    def call(self, y_true, y_pred):
        """
        Calculate loss.
        Args:
        \t- y_true: True images.
        \t- y_pred: Predicted images.
        """

        ssim = -tf.image.ssim(y_true, y_pred, max_val=1., return_index_map=True)
        l1 = tf.reduce_mean(self.l1(y_true, y_pred), axis=tf.range(1, tf.rank(y_true)-1), keepdims=True)
        return self.alpha*ssim + (1-self.alpha)*l1