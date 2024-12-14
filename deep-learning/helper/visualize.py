import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
import random

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
    print("Model loaded successfully!")

    def predict_image(data):
        filename = data['image/filename'].numpy().decode('utf-8')
        real_dog_name = filename.split('/')[1].split('-', 1)[1]

        predictions = model.predict(tf.expand_dims(data["image"], axis=0))
        predicted_class = np.argmax(predictions, axis=-1)[0]

        plt.title(f"Real: {real_dog_name} Predicted Label: {dogs[predicted_class]}")
        plt.imshow(data['original_image'])
        plt.axis("off")
        plt.show()
        print(data['image/filename'])

    dataset_size = len(list(dataset))
    num_images_to_show = min(amount, dataset_size)
    sampled_images = random.sample(list(dataset), num_images_to_show)

    dogs = getDognames(dataset)

    for data in sampled_images:
        predict_image(data)

def getDognames(dataset):
    dognames = {}

    for data in dataset:
        if data['label'].numpy() in dognames:
            continue
        filename = data['image/filename'].numpy().decode('utf-8')
        dog_name = filename.split('/')[1].split('-', 1)[1]
        dognames[data['label'].numpy()] = dog_name

    return dognames