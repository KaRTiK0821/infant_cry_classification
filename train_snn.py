import tensorflow as tf
import numpy as np
import librosa
import os

# Function to load and preprocess audio files
def load_audio(file_path, sr=22050, max_length=3):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = librosa.util.fix_length(audio, size=sr * max_length)
    return np.expand_dims(audio, axis=-1)

# Dataset path
dataset_path = "dataset/"
categories = ["belly_pain", "hungry", "tired", "discomfort", "burping"]

# Ensure dataset directories exist
for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        print(f"Warning: {folder} does not exist. Skipping...")

data_pairs = []
labels = []

# Load data pairs for training
for category in categories:
    folder = os.path.join(dataset_path, category)
    if os.path.exists(folder):
        files = os.listdir(folder)
        if len(files) < 2:
            print(f"Warning: Not enough samples in {folder}. Skipping category.")
            continue
        
        for i in range(len(files) - 1):
            sample_1 = load_audio(os.path.join(folder, files[i]))
            sample_2 = load_audio(os.path.join(folder, files[i + 1]))
            data_pairs.append([sample_1, sample_2])
            labels.append(1)  # Similar pairs

# Convert lists to numpy arrays
data_pairs = np.array(data_pairs)
labels = np.array(labels)

# Define the Siamese Neural Network (SNN) model
def build_snn_model(input_shape):
    base_network = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    input_1 = tf.keras.Input(shape=input_shape)
    input_2 = tf.keras.Input(shape=input_shape)
    
    output_1 = base_network(input_1)
    output_2 = base_network(input_2)
    
    distance = tf.keras.layers.Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))([output_1, output_2])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return model

# Create and compile the SNN model
input_shape = (22050 * 3, 1)
model = build_snn_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([data_pairs[:, 0], data_pairs[:, 1]], labels, batch_size=8, epochs=10)

# Save the trained model
model.save("snn_model.h5")
print("Model training completed and saved as snn_model.h5")
