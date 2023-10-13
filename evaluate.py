import tensorflow as tf

def evaluate(dataset, max_i, encoder, decoder, classifier, adv_attack, epsilon):
    # Initialize scores
    correct_original = 0
    correct_reconstructed = 0
    correct_adv = 0
    # Loop over batches and predict
    for i, (imgs, labels) in enumerate(dataset):
        if i==dataset.cardinality or i==max_i:
            break
        # For originals
        original_predictions = classifier(imgs)
        correct_original += tf.reduce_sum(tf.cast(tf.argmax(original_predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))
        # For adversarials
        adv_imgs = adv_attack(classifier, tf.keras.losses.CategoricalCrossentropy(), imgs, labels, epsilon)
        adv_predictions = classifier(adv_imgs)
        correct_adv += tf.reduce_sum(tf.cast(tf.argmax(adv_predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))
        # For reconstructions
        z = encoder(adv_imgs)
        reconstructed = decoder(z)
        predictions = classifier(reconstructed)
        correct_reconstructed += tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1)==tf.argmax(labels, axis=1), dtype=tf.int32))

        return {
            "acc_original":correct_original/i,
            "acc_adv":correct_adv/i,
            "acc_reconstructed":correct_reconstructed/i
            }