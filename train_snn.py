import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import numpy as np
import librosa
import os

# Custom Layer for Distance Calculation
class CustomDistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

# Define Siamese Network Model
def create_snn(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    shared_dense = tf.keras.Sequential([
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(128, activation="relu")
    ])

    encoded_a = shared_dense(input_a)
    encoded_b = shared_dense(input_b)

    # Use Custom Layer instead of Lambda
    distance = CustomDistanceLayer()([encoded_a, encoded_b])

    output = Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# Load and preprocess audio
def load_audio(file_path, sr=22050, max_length=3):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = librosa.util.fix_length(audio, size=sr * max_length)
    
    return audio.astype(np.float32)

# Prepare dataset
dataset_path = "dataset/"
categories = ["belly_pain", "hungry", "tired", "discomfort", "burping"]

pairs = []
labels = []

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' not found. Please check your dataset location.")

for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Category folder '{folder}' not found. Check dataset structure.")

    files = os.listdir(folder)
    if len(files) < 2:
        continue

    for i in range(len(files) - 1):
        audio_1 = load_audio(os.path.join(folder, files[i]))
        audio_2 = load_audio(os.path.join(folder, files[i + 1]))

        pairs.append((audio_1, audio_2))
        labels.append(1)  # Same category pairs

        # Add negative pairs (different categories)
        negative_category = np.random.choice([c for c in categories if c != category])
        negative_folder = os.path.join(dataset_path, negative_category)
        negative_files = os.listdir(negative_folder)
        if negative_files:
            negative_file = np.random.choice(negative_files)
            audio_negative = load_audio(os.path.join(negative_folder, negative_file))
            pairs.append((audio_1, audio_negative))
            labels.append(0)

# Convert lists to properly structured NumPy arrays
pairs = np.array(pairs, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

# Reshape pairs for training
pairs = pairs.reshape(pairs.shape[0], 2, -1)
input_shape = (pairs.shape[2],)

# Train SNN Model
snn_model = create_snn(input_shape)

snn_model.fit([pairs[:, 0], pairs[:, 1]], labels, batch_size=16, epochs=20, validation_split=0.2)

# Save model in TensorFlow format with custom layer
snn_model.save("snn_model.keras")

print("Model training completed and saved as 'snn_model.keras'")
