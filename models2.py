# Packages and environment
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda" # Replace with correct location
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
# Constants
LATENT_SHAPE = (None, None, 256) # The architecture is also compatible with (None,None,256)
BATCH_SIZE = 32
FILTERS = [32,64,128,256]
N_CLASSES = 10

# Define a function that creates a Sequential model consisting of deconvolutions
def block(nfilters, size, strides, activation=None, name=None, convtype=None, dtype=None):
    """
    A function that implements a sequence of layers to perform a stable transposed convolution block.

    For arguments, see `tf.keras.layers.Conv2DTranspose`.

    Returns: An instance of `tf.keras.Sequential`
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=.4, dtype=dtype),
        (convtype or tf.keras.layers.Conv2D)(filters=nfilters, kernel_size=size, padding="same", strides=strides, dtype=dtype),
        tf.keras.layers.GroupNormalization(groups=nfilters, dtype=dtype),
        activation or tf.keras.layers.LeakyReLU(dtype=dtype)
    ], name=name)

# Define a function to build the encoder
def build_encoder(channels, name="Encoder"):
    """
    Prepares the encoder model using an architecture based on Inception-ResNet.
    
    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=(None, None, channels), batch_size=BATCH_SIZE, name="Input")
    # Create some downsampling blocks
    for i, nfilters in enumerate(FILTERS):
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(12,12), 
            strides=(2,2), 
            name=f"Downsampler_{i}",
            dtype=tf.float32 if i==(len(FILTERS)-1) else None
        )(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)

# Define a function to build the decoder
def build_decoder(channels, name="Decoder"):
    """
    Prepares the decoder model using an architecture based on Inception-ResNet.

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=LATENT_SHAPE, batch_size=BATCH_SIZE, name="LatentInput")
    # Revert the downsampling
    for i, nfilters in enumerate(FILTERS):
        x = block(
            convtype=tf.keras.layers.Conv2DTranspose,
            nfilters=nfilters, 
            size=(12,12), 
            strides=(2,2), 
            name=f"Downsampler_{i}",
            dtype=tf.float32 if i==(len(FILTERS)-1) else None
        )(x)
    # Reshape channels to image format
    outputs = block(
        nfilters=channels, 
        size=(1,1), 
        strides=(1,1), 
        activation=tf.keras.layers.Activation("sigmoid", dtype="float32"), 
        name=f"Reduce",
        dtype=tf.float32
        )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

# Define a function to build the classifier
def build_classifier(channels, name="Classifier"):
    """
    Prepares the classification model using an architecture based on Inception-ResNet.

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=(None, None, channels), batch_size=BATCH_SIZE, name="Input")
    # Create some downsampling blocks
    for i, nfilters in enumerate(FILTERS):
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(12,12), 
            strides=(2,2), 
            name=f"Downsampler_{i}",
        )(x)
    x = tf.keras.layers.GlobalMaxPooling2D(name="Pooling")(x)
    outputs = tf.keras.layers.Dense(units=N_CLASSES, dtype=tf.float32, activation=tf.keras.activations.softmax, name="Dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)