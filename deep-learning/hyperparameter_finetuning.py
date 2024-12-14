from kerastuner.tuners import BayesianOptimization
import tensorflow as tf
import keras
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.setup import setup_dataset, generateDatasetArrays

keras.utils.set_random_seed(0)

DOG_LABEL_IDS = [0, 2, 77, 84, 119]
NUM_CLASSES = len(DOG_LABEL_IDS)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
RESIZE_SIZE = (256, 256)
RANDOM_SEED = 42
DEBUG = True

MODEL_FOLDER = 'models/'
MODEL_FILENAME = 'dog_classifier.keras'
TRANSFER_MODEL_FILENAME = 'dog_classifier_transfer_learning.keras'
WEIGHTS_FILENAME = 'model.weights.h5'
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

train_dataset, test_dataset, label_lookup_table, info = setup_dataset('./dataset', labels=DOG_LABEL_IDS)

def preprocess(data):
    img = data["image"]
    data["original_image"] = tf.identity(data["image"])
    img = tf.image.resize(img, RESIZE_SIZE)
    data["image"] = img
    data["label"] = label_lookup_table.lookup(data["label"])
    return data


train_dataset = (
    train_dataset
        .map(preprocess)
        .shuffle(1000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
        .map(preprocess)
        .shuffle(100)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
)

print("Finished loading data")

def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(*RESIZE_SIZE, 3)
    ))

    if hp.Boolean('conv_2'):
        model.add(keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ))

    if hp.Boolean('dropout_1'):
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_1_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    if hp.Boolean('conv_3'):
        model.add(keras.layers.Conv2D(
            filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ))
    
    if hp.Boolean('dropout_2'):
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_2_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(keras.layers.Flatten())

    if hp.Boolean('Dense_1'):
        model.add(keras.layers.Dense(hp.Int('Dense_1_units', min_value=NUM_CLASSES, max_value=256, step=30), activation='relu'))
    
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='bayesian_opt',
    project_name='dog_classifier'
)

images, labels = generateDatasetArrays(train_dataset)
images_test, labels_test = generateDatasetArrays(test_dataset)
tuner.search((images, labels), epochs=EPOCHS, validation_data=(images_test, labels_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]