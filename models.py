# Packages 
import tensorflow as tf
# Constants
FILTERS = [64,128,256,512]
LATENT_SHAPE = (None, None, FILTERS[-1])
BATCH_SIZE = 64

# Define a function that creates a Sequential model consisting of deconvolutions
def upsampler(nfilters, size, strides, dilation, activation=None, name=None):
    """
    A function that implements a sequence of layers to perform a stable transposed convolution block.\n

    For arguments, see `tf.keras.layers.Conv2DTranspose`.\n

    Returns: An instance of `tf.keras.Sequential`
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=.4),
        tf.keras.layers.Conv2DTranspose(filters=nfilters, kernel_size=size, padding="same", strides=strides, dilation_rate=dilation),
        tf.keras.layers.GroupNormalization(groups=nfilters),
        activation or tf.keras.layers.LeakyReLU()
    ], name=name)

# Define a function that creates a Sequential model consisting of convolutions
def downsampler(nfilters, size, strides, dilation, activation=None, name=None):
    """
    A function that implements a sequence of layers to perform a stable convolution block.\n

    For arguments, see `tf.keras.layers.Conv2D`.\n

    Returns: An instance of `tf.keras.Sequential`
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=.4),
        tf.keras.layers.Conv2D(filters=nfilters, kernel_size=size, padding="same", strides=strides, dilation_rate=dilation),
        tf.keras.layers.GroupNormalization(groups=nfilters),
        activation or tf.keras.layers.LeakyReLU()
    ], name=name)

# Define a simple inception module 
class Inception(tf.keras.Model):
    """
    A class that implements a block of inception using downsampling modules. Inherits from `tf.keras.Model`.\n
    
    For arguments, see `tf.keras.layers.Conv2D`.
    """
    def __init__(self, nfilters, module=upsampler, strides=2, dilation=1, activation=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.branch1 = module(nfilters=nfilters//2, size=6, strides=strides, dilation=dilation, activation=activation)
        self.branch2 = module(nfilters=nfilters//2, size=12, strides=strides, dilation=dilation, activation=activation)
        self.outputs = tf.keras.layers.Concatenate()
        self.config = ({
            "nfilters": nfilters,
            "module": module,
            "strides": strides,
            "dilation": dilation,
            "activation": activation,
            "name": name
        })
    def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        return self.outputs([branch1, branch2])
    def get_config(self):
        self.config.update(super().get_config())
        return self.config
    @classmethod
    def from_config(cls, config):
        config["module"] = tf.keras.layers.deserialize(config["module"])
        return cls(**config)

# Define a function to build the encoder
def build_encoder(channels, name="Encoder"):
    """
    Prepares the encoder model using an architecture based on Inception-ResNet.\n

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = tf.keras.layers.Input(shape=(None, None, channels), batch_size=None, name="Input")
    # Attend to specific parts using memory-efficient attention
    queries = Inception(nfilters=2, module=downsampler, strides=1, name="Queries")(inputs)
    values = Inception(nfilters=2, module=downsampler, strides=1, name="Values")(inputs)
    keys = Inception(nfilters=2, module=downsampler, strides=1, name="Keys")(inputs)    
    x = tf.keras.layers.Attention()([queries, values, keys])
    # Create some downsampling modules
    for i, nfilters in enumerate(FILTERS):
        # Create identity with specified number of filters
        x_ = downsampler(nfilters, size=(1,1), strides=(2,2), dilation=1,name=f"Downsampler_{i}")(x)
        # Pass input through inception module
        x = Inception(nfilters=nfilters, strides=(2,2), module=downsampler, name=f"Inception_{i*2}")(x)
        x = Inception(nfilters=nfilters, strides=(1,1), dilation=2**i, module=downsampler, name=f"Inception_{i*2+1}")(x)
        # Sum skip and inception
        x = tf.keras.layers.Add(name=f"SumSkips_{i}", dtype=tf.float32 if i==FILTERS[-1]-1 else None)([x_, x])
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)

# Define a function to build the decoder
def build_decoder(channels, name="Decoder"):
    """
    Prepares the decoder model using an architecture based on Inception-ResNet.\n

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = x = tf.keras.layers.Input(shape=LATENT_SHAPE, batch_size=None, name="LatentInput")
    # Revert the downsampling
    for i, nfilters in enumerate(reversed(FILTERS)):
        # Create identity with specified number of filters
        x_ = upsampler(nfilters, size=(1,1), strides=(2,2), dilation=1, name=f"Upsampler_{i}")(x)
        # Pass input through inception module
        x = Inception(nfilters=nfilters, strides=(2,2), name=f"Inception_{i*2}")(x)
        x = Inception(nfilters=nfilters, strides=(1,1), dilation=2**i, module=downsampler, name=f"Inception_{i*2+1}")(x)
        # Sum skip and inception
        x = tf.keras.layers.Add(name=f"SumSkips_{i}")([x_, x])
    # Reshape to image format
    outputs = upsampler(
        channels, 
        size=(1,1), 
        strides=(1,1), 
        dilation=1, 
        activation=tf.keras.layers.Activation("sigmoid", dtype=tf.float32), 
        name=f"Upsampler_{i*2+2}"
        )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

# Define a function to build the classifier
def build_classifier(channels, nclasses, name="Classifier"):
    """
    Prepares the classification model using an architecture based on Inception-ResNet.\n

    Returns: An instance of `tf.keras.Model`.
    """
    # Get input
    inputs = tf.keras.layers.Input(shape=(None, None, channels), batch_size=None, name="Input")
    # Attend to specific parts using memory-efficient attention
    queries = Inception(nfilters=2, module=downsampler, strides=1, name="Queries")(inputs)
    values = Inception(nfilters=2, module=downsampler, strides=1, name="Values")(inputs)
    keys = Inception(nfilters=2, module=downsampler, strides=1, name="Keys")(inputs)    
    x = tf.keras.layers.Attention()([queries, values, keys])
    # Create some downsampling modules
    for i, nfilters in enumerate(FILTERS):
        # Create identity with specified number of filters
        x_ = downsampler(nfilters, size=(1,1), strides=(2,2), dilation=1,name=f"Downsampler_{i}")(x)
        # Pass input through inception module
        x = Inception(nfilters=nfilters, strides=(2,2), module=downsampler, name=f"Inception_{i*2}")(x)
        x = Inception(nfilters=nfilters, strides=(1,1), dilation=2**i, module=downsampler, name=f"Inception_{i*2+1}")(x)
        # Sum skip and inception
        x = tf.keras.layers.Add(name=f"SumSkips_{i}")([x_, x])
    x = tf.keras.layers.GlobalMaxPooling2D(name="Pooling")(x)
    x = tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(), name="Dense1")(x)
    x = tf.keras.layers.Dropout(rate=.25)(x)
    x = tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(), name="Dense2")(x)
    x = tf.keras.layers.Dropout(rate=.25)(x)
    x = tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(), name="Dense3")(x)
    x = tf.keras.layers.Dropout(rate=.25)(x)
    outputs = tf.keras.layers.Dense(units=nclasses, dtype=tf.float32, activation=tf.keras.activations.softmax, name="Outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def build_models(channels, nclasses):
    """
    Build all models and use mixed precision.\n
    Args:\n
    \t- channels: int, the number of channels.
    """
    # Encoder
    encoder = build_encoder(channels)
    # Decoder
    decoder = build_decoder(channels)
    # Combine because why not
    ae = tf.keras.Sequential([encoder, decoder], name="Autoencoder")
    # Share an optimizer
    ae_optimizer = tf.keras.optimizers.Adam(epsilon=1e-3) 
    ae_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(ae_optimizer)

    # Classifier
    classifier = build_classifier(channels, nclasses)
    # Optimizer
    classifier_optimizer = tf.keras.optimizers.Adam(epsilon=1e-3, learning_rate=1e-5) 
    classifier_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(classifier_optimizer)
    return classifier, ae, ae_optimizer, classifier_optimizer