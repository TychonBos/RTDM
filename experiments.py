# Load packages
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf, train, json
from models import build_models, BATCH_SIZE
from utils import fgsm, ifgsm, cw, to_dataset, distributor
from evaluate import evaluate
from importlib import reload
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
        .2, # SHOULD THIS BE VARIABLE? 
        classifier, 
        ae, 
        ae_optimizer, 
        classifier_optimizer
    )

    for adv_attack in [cw, fgsm, ifgsm]:
        for epsilon in  [tf.constant(0.01), tf.constant(0.06) if dataname=="cifar10" else tf.constant(0.3)]: # DO NOT LOOP TWICE FOR CW
            # Evaluate
            accuracies = evaluate(data_test, len(X_test)//BATCH_SIZE, classifier, ae, adv_attack, epsilon) # DO NOt LOOP THRICE FOR CLEAN IMGS
            # Save results
            current = f"{adv_attack.__name__} - {float(epsilon)} - {dataname}"
            results[current] = accuracies
            json.dump(results, open("results.json", mode="w"))

    # Reload train module to force reset tf graphs
    tf.keras.backend.clear_session()
    reload(train)