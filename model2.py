import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Ensure TensorFlow logs the error details
tf.debugging.set_log_device_placement(True)

# Directory paths
data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Image parameters
img_width, img_height = 28, 28
num_classes = 5  # Number of shapes 

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1.0/255.0, 
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Function to create, compile and train the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Ensure num_classes matches your dataset
    ])

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Different batch sizes
batch_sizes = [32, 64, 128]
results = {}

# Loop through different batch sizes
for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")

    # Load and preprocess data
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    # Train model with current batch size
    model = create_model()
    history = model.fit(train_generator,
                        epochs=10,
                        validation_data=test_generator)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(test_generator)
    results[batch_size] = val_acc

    print(f"Validation accuracy with batch size {batch_size}: {val_acc:.2f}")

    # Plot training & validation accuracy values for the current batch size
    plt.plot(history.history['accuracy'], label=f'Train Accuracy (Batch Size {batch_size})')
    plt.plot(history.history['val_accuracy'], label=f'Val Accuracy (Batch Size {batch_size})')

# Final plot
plt.title('Model accuracy with different batch sizes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Print final results
print("Final Results (Validation Accuracy):")
for batch_size, accuracy in results.items():
    print(f"Batch Size {batch_size}: {accuracy:.2f}")
