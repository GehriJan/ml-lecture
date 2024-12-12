import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
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

    datasets, metadata = tfds.load(
        name="stanford_dogs",
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
        split=['train[75%:]', 'test[:25%]'],
    )

    train_dataset, test_dataset = list[tf.data.Dataset]([datasets[0], datasets[1]])

    # Filter for classes
    train_dataset = train_dataset.filter(lambda img, label: tf.reduce_any(tf.equal(label, labels)))
    test_dataset = test_dataset.filter(lambda img, label: tf.reduce_any(tf.equal(label, labels)))

    # Show examples
    if show_examples:
        fig = tfds.show_examples(train_dataset, metadata)
        fig.show()

    return train_dataset, test_dataset, metadata

"""
train_dataset, test_dataset, metadata = setup_dataset()
# Show label values
print({int(y.numpy()) for x, y in train_dataset.concatenate(test_dataset)})
# Show number of occurences
images, labels = tuple(zip(*train_dataset))
labels = np.array(labels)
plt.hist(labels, bins=120)
plt.show()"""
