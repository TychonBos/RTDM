import tensorflow as tf
from models import BATCH_SIZE
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2 #https://github.com/cleverhans-lab/cleverhans/issues/1205#issuecomment-1028411235
distributor = tf.distribute.MirroredStrategy()

# Wrap the cleverhans attacks to standardize parameters
def fgsm(classifier, imgs, epsilon):
    return fast_gradient_method(classifier, imgs, epsilon, float("inf"))
def ifgsm(classifier, imgs, epsilon):
    return basic_iterative_method(model_fn=classifier, x=imgs, eps=epsilon, norm=float("inf"), nb_iter=20, eps_iter=.01)
def cw(classifier, imgs, epsilon=None):
    return carlini_wagner_l2(model_fn=classifier, x=imgs, max_iterations=10)

def preprocess(x, y):
    """
    Convert data to correct format.\n
    Args:\n
    \t- x: The images.\n
    \t- y: The labels.
    """
    # Transform for compatibility with 2d conv
    if len(x.shape)<3: # tf.rank does not work
        x = tf.expand_dims(x, axis=-1)
    x = tf.ensure_shape(x, (None,None,None))
    # Make sure it becomes a float in range [0,1]
    x = tf.image.convert_image_dtype(x, dtype=tf.float32)
    # Resize for decoding stability
    x = tf.image.resize(x, (32,32)) # RESIZE TO NEAREST POWER OF 2 GREATER THAN 32
    # One-hot encode labels
    y = tf.one_hot(tf.squeeze(y), depth=10) # tf.unique does not work
    return x, y

# Define dataset transformations
def to_dataset(x, y):
    """
    Takes NumPy arrays and uses these to create a `tf.data.Dataset`.\n
    Args:\n
    \t- x: The images.\n
    \t- y: The corresponding labels.
    """

    return tf.data.Dataset.from_tensor_slices(
        (x, y)
    ).map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        BATCH_SIZE, drop_remainder=True
    ).cache().repeat().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

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

def evaluate(dataset, max_i, classifier, adv_attack=None, ae=None):
    """
    Calculates the accuracy of the classifier before and after purification.\n
    Args:\n
    \t- dataset: The `tf.data.Dataset` to evaluate on.\n
    \t- max_i: In case of an infinitely repeating dataset, the maximum samples.\n
    \t- classifier: The classifier to evaluate.\n
    \t- ae: The purification model.\n
    \t- adv_attack: The adversarial attack function.\n
    """
    # Initialize score 
    correct = 0
    # Loop over batches and predict
    for i, (imgs, labels) in enumerate(dataset):
        if i==dataset.cardinality or i==max_i:
            break
        # Apply adversarial attack
        if adv_attack:
            imgs = adv_attack(imgs)
        # Apply purification
        if ae:
            imgs = ae(imgs)
        # Calculate correct preds
        pred = classifier(imgs, training=False)
        correct += tf.reduce_sum(tf.cast(tf.argmax(pred, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))

    return float(correct/(i*BATCH_SIZE))