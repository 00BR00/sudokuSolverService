# train
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
import argparse
import prepare_data as prep_data
from tensorflow.keras.callbacks import EarlyStopping

def build_model():
    # Define a CNN model for digit classification
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        
        layers.Dropout(0.5),  # Regularization to prevent overfitting
        layers.Dense(128, activation="relu"),  # Adding a dense layer
        layers.Dropout(0.5),  # Another Dropout layer
        layers.Dense(10, activation="softmax")  # Adjusted for 10 classes (0-9)
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

def main(args):
    data_choice = args['data']
    batch_size = args['batch_size']
    epochs = args['epochs']
    model_save_fpath = args['model_save_fpath']

    # Load data depending on user choice
    train_generator = prep_data.get_data(data_choice=data_choice)

    # Get a model instance
    model = build_model()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting training...")
    model.fit(train_generator,
              validation_data=train_generator,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping])
    print("Training complete")
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_fpath), exist_ok=True)

    if os.path.exists(model_save_fpath):
        now = datetime.now()
        suffix = now.strftime("%d_%m_%Y_%H_%M_%S")
        model_save_fpath = f"D:/PROGRAMACION/REACT NATIVE/sudoku_solver/finally solver sudoku/model{suffix}.h5"

    model.save(model_save_fpath)
    print(f"Model saved at: {model_save_fpath}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="custom", type=str, help="Choose data to use ('custom')")
    ap.add_argument("--model_save_fpath", default="models/model.keras", type=str)
    ap.add_argument("--batch_size", default=32, type=int)  # Puedes ajustar esto si es necesario
    ap.add_argument("--epochs", default=30, type=int)  # Ajustar el número de epochs según sea necesario
    
    args = vars(ap.parse_args())

    main(args)
