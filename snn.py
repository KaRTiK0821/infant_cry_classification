import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

# Path to your dataset
DATASET_PATH = "C:/CODE/projects/snn_model/cry/"

# Categories (Ensure these match your actual folder names)
CATEGORIES = ["belly_pain", "burping", "hungry", "discomfort", "tired"]

# Fixed Audio Length (7 sec)
FIXED_DURATION = 7.0
SAMPLE_RATE = 22050  # Standard sampling rate
TIMESTEPS = 300  # Fixed MFCC time frames

# Function to load, trim/pad, and extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Adjust to exactly 7 seconds
    target_length = int(FIXED_DURATION * SAMPLE_RATE)
    if len(y) > target_length:
        y = y[:target_length]  # Trim
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Normalize MFCC values
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Resize MFCC to a fixed shape
    if mfcc.shape[1] < TIMESTEPS:
        mfcc = np.pad(mfcc, ((0, 0), (0, TIMESTEPS - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TIMESTEPS]

    # Reshape to fit CNN (add channel dimension)
    return np.expand_dims(mfcc, axis=-1)  # Shape: (40, TIMESTEPS, 1)

# Function to visualize MFCC spectrogram
def plot_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time", cmap="coolwarm")
    plt.colorbar()
    plt.title("MFCC Visualization")
    plt.show()

# Load and preprocess dataset
def load_dataset():
    X, Y = [], []
    for category in CATEGORIES:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            print(f"⚠️ Warning: Category folder '{category}' not found!")
            continue

        files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
        if not files:
            print(f"⚠️ No .wav files in '{category_path}'!")

        for file in files:
            file_path = os.path.join(category_path, file)
            features = extract_features(file_path)
            X.append(features)
            Y.append(category)

    print(f"✅ Loaded {len(X)} samples from {len(CATEGORIES)} categories.")
    return np.array(X), np.array(Y)

# Define Siamese Network Model
def build_siamese_model(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

# Function to calculate Euclidean distance between two embeddings
def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# Function to classify new audio files using reference dataset
def classify_audio(file_path, reference_data):
    input_feature = extract_features(file_path).flatten()
    
    min_distance = float("inf")
    predicted_label = None

    for ref_label, ref_features_list in reference_data.items():
        for ref_features in ref_features_list:
            distance = euclidean(input_feature, ref_features.flatten())
            if distance < min_distance:
                min_distance = distance
                predicted_label = ref_label

    return predicted_label

# Load and process dataset
X, Y = load_dataset()

# Ensure dataset is loaded
if len(X) == 0:
    print("❌ No data loaded! Check dataset path and files.")
else:
    # Split dataset into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create reference data dictionary
    reference_data = {}
    for label, feature in zip(Y_train, X_train):
        if label not in reference_data:
            reference_data[label] = []
        reference_data[label].append(feature)

    # Build SNN model
    input_shape = (40, TIMESTEPS, 1)  # Shape of MFCC features
    model = build_siamese_model(input_shape)

    # Compile model
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Test with a new audio file
    test_file = "C:/CODE/projects/snn_model/test.wav"
    predicted_label = classify_audio(test_file, reference_data)
    print(f"Predicted Category: {predicted_label}")

    # Visualize MFCC of test file
    plot_mfcc(test_file)
