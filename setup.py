import tensorflow_datasets as tfds
import tensorflow as tf

# Load dataset
def setup_dataset(
        data_dir: str = "./dataset",
        show_examples: bool = False
    ):
    datasets, metadata = tfds.load(
        name="stanford_dogs",
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
        split=['train[75%:]', 'test[:25%]'],
    )

    # Filter for classes
    allowed_labels = [0, 2, 77, 84, 119]
    train_dataset, test_dataset = list[tf.data.Dataset]([datasets[0], datasets[1]])
    train_dataset = train_dataset.filter(lambda img, label: tf.reduce_any(tf.equal(label, allowed_labels)))
    test_dataset = test_dataset.filter(lambda img, label: tf.reduce_any(tf.equal(label, allowed_labels)))

    # Show examples
    if show_examples:
        fig = tfds.show_examples(train_dataset, metadata)
        fig.show()

    return train_dataset, test_dataset, metadata