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
batch_size = 32
num_classes = 5  # Number of shapes

# Data augmentation and normalization
train_datagen_augmented = ImageDataGenerator(rescale=1.0/255.0,
                                             rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=True)

train_datagen_no_aug = ImageDataGenerator(rescale=1.0/255.0)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load and preprocess data
train_generator_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                        target_size=(img_width, img_height),
                                                                        batch_size=batch_size,
                                                                        class_mode='categorical')

train_generator_no_aug = train_datagen_no_aug.flow_from_directory(train_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Check class indices
print("Class indices (train augmented):", train_generator_augmented.class_indices)
print("Class indices (train no aug):", train_generator_no_aug.class_indices)
print("Class indices (test):", test_generator.class_indices)

# Model architecture
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

# Train model with data augmentation
model_aug = create_model()
history_aug = model_aug.fit(train_generator_augmented,
                            epochs=10,
                            validation_data=test_generator)

# Train model without data augmentation
model_no_aug = create_model()
history_no_aug = model_no_aug.fit(train_generator_no_aug,
                                  epochs=10,
                                  validation_data=test_generator)

# Evaluate both models
val_loss_aug, val_acc_aug = model_aug.evaluate(test_generator)
val_loss_no_aug, val_acc_no_aug = model_no_aug.evaluate(test_generator)

print(f"Validation accuracy with augmentation: {val_acc_aug}")
print(f"Validation accuracy without augmentation: {val_acc_no_aug}")

# Generate predictions for model with augmentation
Y_pred_aug = model_aug.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred_aug = np.argmax(Y_pred_aug, axis=1)

# Generate predictions for model without augmentation
Y_pred_no_aug = model_no_aug.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred_no_aug = np.argmax(Y_pred_no_aug, axis=1)

# Classification report and confusion matrix for model with augmentation
print('Confusion Matrix (Augmented)')
print(confusion_matrix(test_generator.classes[:len(y_pred_aug)], y_pred_aug))
print('Classification Report (Augmented)')
target_names = list(train_generator_augmented.class_indices.keys())
print(classification_report(test_generator.classes[:len(y_pred_aug)], y_pred_aug, target_names=target_names))

# Classification report and confusion matrix for model without augmentation
print('Confusion Matrix (No Augmentation)')
print(confusion_matrix(test_generator.classes[:len(y_pred_no_aug)], y_pred_no_aug))
print('Classification Report (No Augmentation)')
print(classification_report(test_generator.classes[:len(y_pred_no_aug)], y_pred_no_aug, target_names=target_names))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_aug.history['accuracy'], label='Train Accuracy (Augmented)')
plt.plot(history_aug.history['val_accuracy'], label='Val Accuracy (Augmented)')
plt.plot(history_no_aug.history['accuracy'], label='Train Accuracy (No Augmentation)')
plt.plot(history_no_aug.history['val_accuracy'], label='Val Accuracy (No Augmentation)')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Augmented', 'Val Augmented', 'Train No Aug', 'Val No Aug'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_aug.history['loss'], label='Train Loss (Augmented)')
plt.plot(history_aug.history['val_loss'], label='Val Loss (Augmented)')
plt.plot(history_no_aug.history['loss'], label='Train Loss (No Augmentation)')
plt.plot(history_no_aug.history['val_loss'], label='Val Loss (No Augmentation)')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Augmented', 'Val Augmented', 'Train No Aug', 'Val No Aug'], loc='upper left')

plt.show()


