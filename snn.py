import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
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
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input_layer, x)

# Build full Siamese network
def build_siamese_network(input_shape):
    base_network = build_siamese_model(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Get feature embeddings
    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Compute absolute difference
    diff = Subtract()([encoded_a, encoded_b])
    output = Dense(1, activation='sigmoid')(diff)

    return Model(inputs=[input_a, input_b], outputs=output)

# Generate pairs for contrastive learning
def create_pairs(X, Y):
    pairs, labels = [], []
    class_indices = {label: np.where(Y == label)[0] for label in np.unique(Y)}

    for idx in range(len(X)):
        current_class = Y[idx]
        pos_idx = random.choice(class_indices[current_class])
        neg_class = random.choice([c for c in np.unique(Y) if c != current_class])
        neg_idx = random.choice(class_indices[neg_class])

        pairs.append([X[idx], X[pos_idx]])
        labels.append(1)  # Similar

        pairs.append([X[idx], X[neg_idx]])
        labels.append(0)  # Dissimilar

    return np.array(pairs), np.array(labels)

# Load and process dataset
X, Y = load_dataset()

# Ensure dataset is loaded
if len(X) == 0:
    print("❌ No data loaded! Check dataset path and files.")
else:
    # Encode labels as numerical values
    encoder = LabelEncoder()
    Y_numeric = encoder.fit_transform(Y)

    # Split dataset into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_numeric, test_size=0.2, random_state=42)

    # Generate pairs for training and testing
    train_pairs, train_labels = create_pairs(X_train, Y_train)
    test_pairs, test_labels = create_pairs(X_test, Y_test)

    # Build SNN model
    input_shape = (40, TIMESTEPS, 1)  # Shape of MFCC features
    siamese_model = build_siamese_network(input_shape)

    # Compile model
    siamese_model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    siamese_model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, batch_size=16, epochs=20)

    # Evaluate model
    y_pred = siamese_model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    y_pred = np.round(y_pred)  # Convert probabilities to binary labels

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)

    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print(f"✅ Model Precision: {precision:.4f}")

    # Test with a new audio file
    test_file = "C:/CODE/projects/snn_model/test.wav"
    test_feature = extract_features(test_file)

    # Find the closest match from reference data
    reference_data = {label: [] for label in np.unique(Y_train)}
    for label, feature in zip(Y_train, X_train):
        reference_data[label].append(feature)

    min_distance = float("inf")
    predicted_label = None

    for ref_label, ref_features_list in reference_data.items():
        for ref_features in ref_features_list:
            distance = euclidean(test_feature.flatten(), ref_features.flatten())
            if distance < min_distance:
                min_distance = distance
                predicted_label = ref_label

    print(f"Predicted Category: {encoder.inverse_transform([predicted_label])[0]}")

    # Visualize MFCC of test file
    plot_mfcc(test_file)
