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
        split=['train[75%:]', 'test[:25%]'],
    )

    train_dataset, test_dataset = list[tf.data.Dataset]([datasets[0], datasets[1]])

    # Filter for classes
    train_dataset = train_dataset.filter(lambda item: tf.reduce_any(tf.equal(item['label'], labels)))
    test_dataset = test_dataset.filter(lambda item: tf.reduce_any(tf.equal(item['label'], labels)))

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