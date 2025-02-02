import tensorflow as tf
import numpy as np
import librosa
import os

def load_audio(file_path, sr=22050, max_length=3):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = librosa.util.fix_length(audio, size=sr * max_length)
    return np.expand_dims(audio, axis=-1)

def classify_audio(model, audio_sample, reference_samples, categories, threshold=0.3):
    min_distance = float('inf')
    best_label = "unknown"
    
    for label, ref_sample in reference_samples.items():
        ref_sample = np.expand_dims(ref_sample, axis=0)
        pred = model.predict([np.expand_dims(audio_sample, axis=0), ref_sample])[0][0]
        
        if pred < min_distance:
            min_distance = pred
            best_label = label
    
    return best_label if min_distance < threshold else "unknown"

# Load trained model
model = tf.keras.models.load_model("snn_model.h5", compile=False)

# Load reference samples for comparison
reference_samples = {}
categories = ["belly_pain", "hungry", "tired", "discomfort", "burping"]

dataset_path = "dataset/"
for category in categories:
    folder = os.path.join(dataset_path, category)
    file = os.listdir(folder)[0] if os.listdir(folder) else None
    if file:
        reference_samples[category] = load_audio(os.path.join(folder, file))

# Classify new audio
input_audio = "test_audio.wav"
audio_sample = load_audio(input_audio)
result = classify_audio(model, audio_sample, reference_samples, categories)
print(f"Predicted Category: {result}")
