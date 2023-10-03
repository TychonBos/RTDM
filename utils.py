import tensorflow as tf
from models import BATCH_SIZE

# Define function for calculating and applying perturbations
def fgsm(classifier, clf_loss_fn, imgs, labels, epsilon=tf.constant(0.01)):
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        predictions = classifier(imgs)
        loss = clf_loss_fn(labels, predictions)

    # Get the gradients of the loss w.r.t to the input image.
    gradients = tape.gradient(loss, imgs)
    # Get the sign of the gradients to create the perturbation
    signed_gradients = tf.sign(gradients)
    return imgs + epsilon*signed_gradients

# Define dataset transformations
def dataset_from_arrays(x, y):
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

# A loss that is not per-pixel --> https://arxiv.org/pdf/1511.08861.pdf
class MS_SSIM_L1(tf.keras.losses.Loss): 
    def __init__(self, alpha=.84, **kwargs):
        kwargs.pop("reduction", None)
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.l1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.alpha = alpha
    def call(self, y_true, y_pred):
        ssim = -tf.image.ssim_multiscale(y_true, y_pred, max_val=1., return_index_map=True)
        l1 = tf.reduce_mean(self.l1(y_true, y_pred), axis=tf.range(1, tf.rank(y_true)), keepdims=True)
        return self.alpha*ssim + (1-self.alpha)*l1