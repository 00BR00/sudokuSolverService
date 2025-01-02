# prepare data
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_custom_data(data_directory):
    # Create an ImageDataGenerator for data augmentation and normalization
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    # Load training data
    train_generator = datagen.flow_from_directory(
        data_directory,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True
    )

    # Print class indices for verification
    print("Class indices:", train_generator.class_indices)

    return train_generator

def get_data(data_choice):
    data_choice = data_choice.lower()
    if data_choice != 'custom':
        raise ValueError(f"Invalid value for data_choice: {data_choice}. Valid option is: 'custom'")

    # Load custom data
    data_directory = r"D:\PROGRAMACION\REACT NATIVE\sudoku_solver\finally solver sudoku\archive (1)\digits updated\digits updated"
    return load_custom_data(data_directory)

# Example of calling get_data
train_generator = get_data("custom")
