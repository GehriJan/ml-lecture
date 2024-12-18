import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
import random
import math

def visualize_history(history):
    # Loss-Werte
    loss = history['loss']
    val_loss = history['val_loss']

    # Accuracy-Werte (falls verwendet)
    accuracy = history.get('accuracy')
    val_accuracy = history.get('val_accuracy')

    # Epochen erstellen
    epochs = range(1, len(loss) + 1)

    # Plot für den Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Trainings Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validierungs Loss')
    plt.title('Trainings und Validierungs Loss')
    plt.xlabel('Epochen')
    plt.ylabel('Loss')
    plt.legend()

    # Plot für die Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Trainings Genauigkeit')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validierungs Genauigkeit')
    plt.title('Trainings und Validierungs Genauigkeit')
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit')
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_predictions(model_path, dataset, amount=10):
    # Load the model
    model = keras.models.load_model(model_path)

    def predict_image(data):
        filename = data['image/filename'].numpy().decode('utf-8')
        real_dog_name = filename.split('/')[1].split('-', 1)[1]

        predictions = model.predict(tf.expand_dims(data["image"], axis=0))
        predicted_class = np.argmax(predictions, axis=-1)[0]

        return real_dog_name, dogs[predicted_class], data['original_image']

    dataset_size = len(list(dataset))
    num_images_to_show = min(amount, dataset_size)
    sampled_images = random.sample(list(dataset), num_images_to_show)

    dogs = getDognames(dataset)

    cols = 3
    rows = math.ceil(num_images_to_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))

    for i, data in enumerate(sampled_images):
        real_dog_name, predicted_label, image = predict_image(data)

        # Plot image in the grid
        ax = axes[i // cols, i % cols]
        ax.set_title(f"Real: {real_dog_name}\n    Pred: {predicted_label}")
        ax.imshow(image)
        ax.axis("off")

    # Hide unused subplots if the grid is not fully occupied
    for j in range(num_images_to_show, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    plt.tight_layout()
    plt.show()

