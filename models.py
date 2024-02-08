# Packages
import tensorflow as tf
# Constants
FILTERS = [64,128,256]
LATENT_SHAPE = (None, None, FILTERS[-1])
BATCH_SIZE = 64

# Define a function that creates a Sequential model consisting of deconvolutions
def block(nfilters, size, strides, activation=None, name=None, convtype=None, dtype=None):
    """
    A function that implements a sequence of layers to perform a stable transposed convolution block.

    For arguments, see `tf.keras.layers.Conv2DTranspose`.

    Returns: An instance of `tf.keras.Sequential`
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=.25, dtype=dtype),
        (convtype or tf.keras.layers.Conv2D)(filters=nfilters, kernel_size=size, padding="same", strides=strides, dtype=dtype, kernel_constraint=tf.keras.constraints.MaxNorm(3)),
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
    inputs = x = tf.keras.layers.Input(shape=(None, None, channels), batch_size=None, name="Input")
    # Create some downsampling blocks
    for i, nfilters in enumerate(FILTERS):
        x_ = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(1,1), 
            strides=(1,1), 
            name=f"Skip_{i}"
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(1,1), 
            name=f"Downsampler_{i*2}",
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(1,1), 
            name=f"Downsampler_{i*2+1}",
        )(x)
        x = tf.keras.layers.Add(name=f"SumSkips_{i}")([x, x_])
        x = tf.keras.layers.MaxPool2D()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)

# Define a function to build the decoder
def build_decoder(channels, name="Decoder"):
    """
    Prepares the decoder model using an architecture based on Inception-ResNet.

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=LATENT_SHAPE, batch_size=None, name="LatentInput")
    # Revert the downsampling
    for i, nfilters in enumerate(reversed(FILTERS)):
        x_ = block(
            convtype=tf.keras.layers.Conv2DTranspose,
            nfilters=nfilters, 
            size=(1,1), 
            strides=(2,2), 
            name=f"Skip_{i}"
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(1,1), 
            name=f"Upsampler_{i*2}",
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2DTranspose,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(2,2) , 
            name=f"Upsampler_{i*2+1}",
        )(x)
        x = tf.keras.layers.Add(name=f"SumSkips_{i}")([x, x_])
    # Reshape channels to image format
    outputs = block(
        nfilters=channels, 
        size=(1,1), 
        strides=(1,1), 
        activation=tf.keras.layers.Activation("sigmoid", dtype=tf.float32), 
        name=f"Reduce",
        dtype=tf.float32
        )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

# Define a function to build the classifier
def build_classifier(channels, nclasses, name="Classifier"):
    """
    Prepares the classification model using an architecture based on Inception-ResNet.

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=(None, None, channels), batch_size=None, name="Input")
    # Create some downsampling blocks
    for i, nfilters in enumerate(FILTERS):
        x_ = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(1,1), 
            strides=(1,1), 
            name=f"Skip_{i}"
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(1,1), 
            name=f"Downsampler_{i*2}",
        )(x)
        x = block(
            convtype=tf.keras.layers.Conv2D,
            nfilters=nfilters, 
            size=(3,3), 
            strides=(1,1), 
            name=f"Downsampler_{i*2+1}",
        )(x)
        x = tf.keras.layers.Add(name=f"SumSkips_{i}")([x, x_])
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalMaxPooling2D(name="Pooling")(x)
    x = tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(), name="Dense_1", kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = tf.keras.layers.Dropout(rate=.25)(x)
    x = tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(), name="Dense_2", kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = tf.keras.layers.Dropout(rate=.25)(x)
    outputs = tf.keras.layers.Dense(units=nclasses, dtype=tf.float32, activation=tf.keras.activations.softmax, name="Outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def build_models(channels, nclasses):
    """
    Build all models and use mixed precision.
    Args:
    \t- channels: int, the number of channels.
    """
    # Encoder
    encoder = build_encoder(channels)
    # Decoder
    decoder = build_decoder(channels)
    # Combine because why not
    ae = tf.keras.Sequential([encoder, decoder], name="Autoencoder")
    # Share an optimizer
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=5e3, decay_rate=.95)
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler, epsilon=1e-4) 
    ae_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(ae_optimizer)

    # Classifier
    classifier = build_classifier(channels, nclasses)
    # Optimizer
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=5e3, decay_rate=.95)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler, epsilon=1e-4) 
    classifier_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(classifier_optimizer)
    return classifier, ae, ae_optimizer, classifier_optimizer