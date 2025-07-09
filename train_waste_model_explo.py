import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

#paths for the dataset
dataset_dir = "./Data"

# Loading data using image dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2, 
    subset="training",
    seed=123,  
    image_size=(256, 256),  # Resize images
    batch_size=32  # Adjust batch size
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

# model architecture
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(256, 256, 3)),  # Rescale
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  #classes of classification
])

# Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
EPOCHS = 15
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Saving
model.save('waste_classification_model.h5')

import matplotlib.pyplot as plt

# Plotting accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plotting loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()