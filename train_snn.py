import os
import numpy as np
import librosa
import random
import itertools
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

# Extract MFCC features
def extract_mfcc(file_path, sr=16000, n_mfcc=13, max_length=100):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # Ensure shape is (n_mfcc, max_length)
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]

        mfcc = mfcc.T  # ✅ Transpose to (100, 13)

        print(f"📊 MFCC Shape ({file_path}): {mfcc.shape}")  # Debugging
        return mfcc  

    except Exception as e:
        print(f"🚨 Error processing {file_path}: {e}")
        return np.zeros((max_length, n_mfcc))  # Return correct shape even on error


# Load dataset
def load_dataset(dataset_path):
    data, labels = [], []
    label_dict, label_idx = {}, 0
    
    for category in sorted(os.listdir(dataset_path)):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in label_dict:
                label_dict[category] = label_idx
                label_idx += 1
            for file in os.listdir(category_path):
                if file.endswith(".wav"):
                    feature = extract_mfcc(os.path.join(category_path, file))
                    data.append(feature)
                    labels.append(label_dict[category])
    
    return np.array(data), np.array(labels), label_dict

def create_pairs(features, labels):
    pairs, labels_pair = [], []
    label_dict = {}
    
    for idx, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(idx)
    
    for label in label_dict.keys():
        same_class = label_dict[label]
        diff_classes = list(set(label_dict.keys()) - {label})
        
        for (i, j) in itertools.combinations(same_class, 2):
            pairs.append([features[i], features[j]])
            labels_pair.append(1)
        
        for i in same_class:
            j = random.choice(label_dict[random.choice(diff_classes)])
            pairs.append([features[i], features[j]])
            labels_pair.append(0)
    
    return np.array(pairs), np.array(labels_pair)

def build_base_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(64, 3, activation="relu", padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    return Model(input_layer, x)

def euclidean_distance(vectors):
    x, y = vectors
    return tf.math.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def build_siamese_network(input_shape):
    base_model = build_base_model(input_shape)
    input_a, input_b = Input(shape=input_shape), Input(shape=input_shape)
    feat_a, feat_b = base_model(input_a), base_model(input_b)
    distance = Lambda(euclidean_distance)([feat_a, feat_b])
    output = Dense(1, activation="sigmoid")(distance)
    return Model(inputs=[input_a, input_b], outputs=output)

DATASET_PATH = "dataset/"
features, labels, label_map = load_dataset(DATASET_PATH)
print(f"✅ Training Features Shape: {features.shape}")  # Debugging
pairs, pair_labels = create_pairs(features, labels)

input_shape = (100, 13)  # Fixed size (time_steps, features)
siamese_model = build_siamese_network(input_shape)
siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

siamese_model.fit([pairs[:, 0], pairs[:, 1]], pair_labels, batch_size=32, epochs=50, validation_split=0.2)
siamese_model.save("siamese_model.h5")