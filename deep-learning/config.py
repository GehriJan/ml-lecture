DOG_LABEL_IDS = [9, 118, 77, 85, 119]
NUM_CLASSES = len(DOG_LABEL_IDS)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
RESIZE_SIZE = (256, 256)
RANDOM_SEED = 42
DEBUG = False

MODEL_FOLDER = 'models/'
MODEL_FILENAME = 'dog_classifier.keras'
TRANSFER_MODEL_FILENAME = 'dog_classifier_transfer_learning.keras'
WEIGHTS_FILENAME = 'model.weights.h5'