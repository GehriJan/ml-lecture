from kerastuner.tuners import BayesianOptimization
import tensorflow as tf
import keras
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
from IPython.display import display
import numpy as np

# Load dataset
def setup_dataset(
        data_dir: str = "./dataset",
        show_examples: bool = False,
        labels = [0, 2, 77, 84, 119]
    ):
    """
    This function downloads the dataset and only keeps
    the data specified in `labels`.

    It splitts the dataset in train and test dataset
    """

    datasets, info = tfds.load(
        name="stanford_dogs",
        data_dir=data_dir,
        with_info=True,
        split='train',
    )

    filtered_dataset = datasets.filter(
        lambda item: tf.reduce_any(tf.equal(item['label'], labels))
    )
    total_samples = len(list(filtered_dataset))
    train_size = int(0.75 * total_samples)

    # Split the filtered dataset into train and test
    train_dataset = filtered_dataset.take(train_size)
    test_dataset = filtered_dataset.skip(train_size)

    print("Total Image count: ", total_samples)
    print("Train Dataset Size:", len(list(train_dataset)))
    print("Test Dataset Size:", len(list(test_dataset)))

    # Show examples
    if show_examples:
        fig = tfds.show_examples(train_dataset, info)
        fig.show()
        df = tfds.as_dataframe(train_dataset.take(4), info)
        display(df)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(labels, dtype=tf.int64),
            values=tf.constant(list(range(len(labels))),  dtype=tf.int64),
        ),
        default_value= tf.constant(0,  dtype=tf.int64)
    )

    return train_dataset, test_dataset, table, info

def generateDatasetArrays(dataset):
    combined_dataset = dataset.map(lambda data: (data["image"], data["label"]))
    combined_array = list(combined_dataset)
    images_combined = [item[0] for item in combined_array]
    labels_combined = [item[1] for item in combined_array]
    images = np.array(images_combined)
    labels = np.array(labels_combined)
    return images, labels

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

train_dataset, test_dataset, label_lookup_table, info = setup_dataset('../dataset', labels=DOG_LABEL_IDS)

def preprocess(data):
    data["original_image"] = tf.identity(data["image"])
    data["image"] = tf.image.resize(data["image"], RESIZE_SIZE)
    data["label"] = label_lookup_table.lookup(data["label"])
    return data

train_dataset = (
    train_dataset
        .map(preprocess)
        .shuffle(1000)
)

test_dataset = (
    test_dataset
        .map(preprocess)
        .shuffle(100)
)

images, labels = generateDatasetArrays(train_dataset)
images_test, labels_test = generateDatasetArrays(test_dataset)

train_dataset = (
    train_dataset
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
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


tuner.search((images, labels), epochs=EPOCHS, validation_data=(images_test, labels_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]