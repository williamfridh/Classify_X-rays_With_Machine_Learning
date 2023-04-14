import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

import matplotlib.pyplot as plt
import numpy

DIRECTORY = 'Covid19-dataset/train'
BATH_SIZE = 8
IMG_WIDTH = 256
IMG_HEIGHT = 256
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
COLOR_MODE = 'grayscale'
CLASS_MODE = 'categorical'

# Training data
training_data_generator = ImageDataGenerator(
    zoom_range = 0.1,
    rotation_range = 25,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    rescale=1/255)

training_iterator = training_data_generator.flow_from_directory(
    DIRECTORY,
    target_size = TARGET_SIZE,
    color_mode = COLOR_MODE,
    batch_size = BATH_SIZE,
    class_mode = CLASS_MODE)
training_iterator.next()    # What does this do? (taken from example)

# Validation data
validation_data_generator = ImageDataGenerator(rescale=1.0/255)

validation_iterator = validation_data_generator.flow_from_directory(
    DIRECTORY,
    target_size = TARGET_SIZE,
    color_mode = COLOR_MODE,
    batch_size = BATH_SIZE,
    class_mode = CLASS_MODE)

# Create model
def create_model():
    new_model = Sequential()

    new_model.add(layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)))

    new_model.add(layers.Conv2D(5, 5, strides=3, activation="relu")) 
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    new_model.add(layers.Dropout(0.1))
    new_model.add(layers.Conv2D(3, 3, strides=1, activation="relu")) 
    new_model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    new_model.add(layers.Dropout(0.2))
    new_model.add(layers.Conv2D(2, 2, strides=2, activation="relu")) 
    new_model.add(layers.Dropout(0.3))

    new_model.add(layers.Flatten())
    new_model.add(layers.Dense(3, activation = 'softmax'))

    # Compile model
    new_model.compile(
        optimizer = optimizers.Adam(learning_rate = 0.003),
        loss = losses.CategoricalCrossentropy(),
        metrics = [metrics.CategoricalAccuracy(), metrics.AUC()])
    
    return new_model

model = create_model()
model.summary()

# Early stopping implementation (taken from example)
es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

# Fit
history = model.fit(
    training_iterator,
    steps_per_epoch = training_iterator.samples / BATH_SIZE,
    epochs = 5,
    validation_data = validation_iterator,
    validation_steps = validation_iterator.samples / BATH_SIZE,
    callbacks = [es])

# Test model
predictions = model.predict(validation_iterator, validation_iterator.samples / BATH_SIZE)
predicted_classes = numpy.argmax(predictions, axis = 1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names = class_labels)
print(report)

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
 
fig.savefig('my_plots.png')