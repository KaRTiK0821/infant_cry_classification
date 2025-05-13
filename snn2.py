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

# Set paths
DATASET_PATH = "C:/CODE/projects/snn_model/baby_chillanto/"
SPECTROGRAM_SAVE_PATH = "C:/CODE/projects/snn_model/spectrograms/"
TEST_AUDIO_PATH = "C:/CODE/projects/snn_model/test_new.wav"

# Create spectrogram folder if not exists
os.makedirs(SPECTROGRAM_SAVE_PATH, exist_ok=True)

# Categories
CATEGORIES = ["asphyxia", "deaf", "hunger", "normal", "pain"]

FIXED_DURATION = 7.0
SAMPLE_RATE = 22050
TIMESTEPS = 300

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = int(FIXED_DURATION * SAMPLE_RATE)
    y = y[:target_length] if len(y) > target_length else np.pad(y, (0, target_length - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, TIMESTEPS - mfcc.shape[1]))) if mfcc.shape[1] < TIMESTEPS else mfcc[:, :TIMESTEPS]
    return np.expand_dims(mfcc, axis=-1)

# Save spectrogram image
def save_mfcc_spectrogram(file_path, category, filename):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"MFCC - {category}")
    save_path = os.path.join(SPECTROGRAM_SAVE_PATH, f"{category}_{filename}.png")
    plt.savefig(save_path)
    plt.close()

# Load dataset
def load_dataset():
    X, Y = [], []
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(path):
            print(f"⚠️ Category '{category}' folder not found!")
            continue
        for idx, file in enumerate([f for f in os.listdir(path) if f.endswith(".wav")]):
            file_path = os.path.join(path, file)
            features = extract_features(file_path)
            X.append(features)
            Y.append(category)
            save_mfcc_spectrogram(file_path, category, os.path.splitext(file)[0])
    print(f"✅ Loaded {len(X)} samples.")
    return np.array(X), np.array(Y)

# Siamese model
def build_siamese_model(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inp, x)

def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def classify_audio(file_path, reference_data):
    input_feature = extract_features(file_path).flatten()
    min_distance = float("inf")
    predicted_label = None
    for ref_label, features_list in reference_data.items():
        for ref_feat in features_list:
            dist = euclidean(input_feature, ref_feat.flatten())
            if dist < min_distance:
                min_distance = dist
                predicted_label = ref_label
    return predicted_label

# Load Data
X, Y = load_dataset()
if len(X) == 0:
    print("❌ No data loaded!")
    exit()

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build reference dictionary
reference_data = {}
for label, feature in zip(Y_train, X_train):
    reference_data.setdefault(label, []).append(feature)

# Build SNN
input_shape = (40, TIMESTEPS, 1)
embedding_model = build_siamese_model(input_shape)

# Create two input layers
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Generate embeddings
embedding_a = embedding_model(input_a)
embedding_b = embedding_model(input_b)

# Compute distance and final prediction
distance = Lambda(euclidean_distance)([embedding_a, embedding_b])
output = Dense(1, activation="sigmoid")(distance)

model = Model(inputs=[input_a, input_b], outputs=output)
model.compile(loss="binary_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])

# Create positive/negative pairs
def create_pairs(X, Y):
    pairs, labels = [], []
    label_to_indices = {label: np.where(Y == label)[0] for label in CATEGORIES}
    for idx_a in range(len(X)):
        x1, label1 = X[idx_a], Y[idx_a]
        # Positive pair
        idx_b = np.random.choice(label_to_indices[label1])
        x2 = X[idx_b]
        pairs.append([x1, x2])
        labels.append(1)
        # Negative pair
        neg_label = np.random.choice([lbl for lbl in CATEGORIES if lbl != label1])
        idx_b = np.random.choice(label_to_indices[neg_label])
        x2 = X[idx_b]
        pairs.append([x1, x2])
        labels.append(0)
    return np.array(pairs), np.array(labels)

train_pairs, train_labels = create_pairs(X_train, Y_train)
test_pairs, test_labels = create_pairs(X_test, Y_test)

# Train the model
history = model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
    validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
    batch_size=32,
    epochs=10
)

# Evaluate
loss, accuracy = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
print(f"\n🔍 Final Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Final Loss: {loss:.4f}")

# Predict category of test audio
if os.path.exists(TEST_AUDIO_PATH):
    predicted = classify_audio(TEST_AUDIO_PATH, reference_data)
    print(f"\n🔊 Test Audio Prediction: {predicted}")
else:
    print("⚠️ Test audio file not found!")
