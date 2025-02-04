import tensorflow as tf
import numpy as np
import librosa
import os

# Custom Distance Layer for Loading Model
class CustomDistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

# Load audio for classification
def load_audio(file_path, sr=22050, max_length=3):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = librosa.util.fix_length(audio, size=sr * max_length)
    return np.expand_dims(audio, axis=-1)  # Ensure correct shape

# Classify audio using trained model
def classify_audio(model, audio_sample, reference_samples, threshold=0.3):
    min_distance = float('inf')
    best_label = "unknown"
    
    for label, ref_sample in reference_samples.items():
        ref_sample = np.expand_dims(ref_sample, axis=0)
        pred = model.predict([np.expand_dims(audio_sample, axis=0), ref_sample])[0][0]
        
        if pred < min_distance:
            min_distance = pred
            best_label = label
    
    return best_label if min_distance < threshold else "unknown"

# Try loading model with custom layer
try:
    model = tf.keras.models.load_model("snn_model.keras", custom_objects={"CustomDistanceLayer": CustomDistanceLayer})
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load reference samples for comparison
reference_samples = {}
categories = ["belly_pain", "hungry", "tired", "discomfort", "burping"]

dataset_path = "dataset/"
for category in categories:
    folder = os.path.join(dataset_path, category)
    if os.path.exists(folder) and os.listdir(folder):
        file = os.listdir(folder)[0]
        reference_samples[category] = load_audio(os.path.join(folder, file))

# Classify new audio
input_audio = "test_audio.wav"
if not os.path.exists(input_audio):
    print(f"Error: Test audio file '{input_audio}' not found!")
    exit()

audio_sample = load_audio(input_audio)
result = classify_audio(model, audio_sample, reference_samples)
print(f"Predicted Category: {result}")
