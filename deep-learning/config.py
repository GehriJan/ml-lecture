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