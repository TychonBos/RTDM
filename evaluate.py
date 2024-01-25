import tensorflow as tf
from models import BATCH_SIZE

def evaluate(dataset, max_i, classifier, ae, adv_attack, epsilon): # MOVE TO utils.py
    """
    Calculates the accuracy of the classifier before and after purification.\n
    Args:\n
    \t- dataset: The `tf.data.Dataset` to evaluate on.\n
    \t- max_i: In case of an infinitely repeating dataset, the maximum samples.\n
    \t- classifier: The classifier to evaluate.\n
    \t- ae: The purification model.\n
    \t- adv_attack: The adversarial attack function.\n
    \t- epsilon: The epsilon value for the adversarial attack.
    """
    # Initialize scores
    correct_original = 0
    correct_reconstructed = 0
    correct_adv = 0
    # Loop over batches and predict
    for i, (imgs, labels) in enumerate(dataset):
        if i==dataset.cardinality or i==max_i:
            break
        # For originals
        original_predictions = classifier(imgs, training=False)
        correct_original += tf.reduce_sum(tf.cast(tf.argmax(original_predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))
        # For adversarials
        adv_imgs = adv_attack(classifier, imgs, epsilon)
        adv_predictions = classifier(adv_imgs, training=False)
        correct_adv += tf.reduce_sum(tf.cast(tf.argmax(adv_predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))
        # For reconstructions
        reconstructed = ae(adv_imgs, training=False)
        predictions = classifier(reconstructed, training=False)
        correct_reconstructed += tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))

    return {
        "acc_original":float(correct_original/(i*BATCH_SIZE)),
        "acc_adv":float(correct_adv/(i*BATCH_SIZE)),
        "acc_reconstructed":float(correct_reconstructed/(i*BATCH_SIZE))
        }