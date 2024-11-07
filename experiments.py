# Load packages
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf, train, json
from models import build_models, BATCH_SIZE
from utils import fgsm, ifgsm, cw, to_dataset, distributor, evaluate
from importlib import reload
from functools import partial
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Loop over parameters
results = {}
for data in [tf.keras.datasets.cifar10, tf.keras.datasets.mnist, tf.keras.datasets.fashion_mnist]:
    # Convert data to tf dataset 
    dataname = data.__name__.split(".")[-1]
    (X_train, y_train), (X_test, y_test) = data.load_data()
    data_train = to_dataset(X_train, y_train) 
    data_test = to_dataset(X_test, y_test)

    # Build the models
    with distributor.scope():
        classifier, ae, ae_optimizer, classifier_optimizer = build_models(
            channels=3 if dataname=="cifar10" else 1, # data.element_spec is undefined
            nclasses=tf.size(tf.unique(tf.squeeze(y_train))[0])
        ) 
    # Train the models and save them
    train.run(
        distributor.experimental_distribute_dataset(data_train), 
        classifier, 
        ae, 
        ae_optimizer, 
        classifier_optimizer
    )

    # Initialize results
    results[dataname] = {}
    results[dataname]["clean"] = evaluate(data_test, len(X_test)//BATCH_SIZE, classifier)
    # Loop over attacks
    for adv_attack in [
        partial(cw, classifier=classifier, epsilon=None), 
        partial(fgsm, classifier=classifier, epsilon=tf.constant(.01)), 
        partial(ifgsm, classifier=classifier, epsilon=tf.constant(.01)), 
        partial(fgsm, classifier=classifier, epsilon=tf.constant(0.06) if dataname=="cifar10" else tf.constant(0.3)), 
        partial(ifgsm, classifier=classifier, epsilon=tf.constant(0.06) if dataname=="cifar10" else tf.constant(0.3)), 
        partial(fgsm, classifier=classifier, epsilon=tf.constant(0.3) if dataname=="cifar10" else tf.constant(0.5)), 
        partial(ifgsm, classifier=classifier, epsilon=tf.constant(0.3) if dataname=="cifar10" else tf.constant(0.5))
    ]:
        # Evaluate
        results[dataname][f"{adv_attack.func.__name__} ({adv_attack.keywords['epsilon']})"] = evaluate(
            data_test, 
            len(X_test)//BATCH_SIZE, 
            classifier, 
            adv_attack
        )
        results[dataname][f"purified {adv_attack.func.__name__} ({adv_attack.keywords['epsilon']})"] = evaluate(
            data_test, 
            len(X_test)//BATCH_SIZE, 
            classifier, 
            adv_attack, 
            ae
        )
    # Save results
    json.dump(results, open("results.json", mode="w"))

    # Reload train module to force reset tf graphs
    tf.keras.backend.clear_session()
    reload(train)