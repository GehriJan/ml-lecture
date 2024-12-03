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
    train_dataset, test_dataset = list[tf.data.Dataset]([datasets[0], datasets[1]])
    train_dataset = train_dataset.filter(lambda img, label: label==0 or label==2 or label==77 or label==84 or label==119)
    test_dataset = test_dataset.filter(lambda img, label: label==0 or label==2 or label==77 or label==84 or label==119)

    # Show examples
    if show_examples:
        fig = tfds.show_examples(train_dataset, metadata)
        fig.show()

    return train_dataset, test_dataset, metadata