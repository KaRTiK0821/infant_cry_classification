import tensorflow as tf
import numpy as np
import librosa
import os

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

mfcc_features = extract_mfcc("test_audio.wav")  # Ensure it’s (100, 13)
print("Test MFCC Shape:", mfcc_features.shape)

def load_reference_features(dataset_path):
    reference_features = {}
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                mfcc = extract_mfcc(file_path)
                
                # Ensure the shape is correct
                if mfcc.shape == (13, 100):  
                    reference_features[file_path] = (category, mfcc)

    return reference_features

# Define the custom function again
def euclidean_distance(vectors):
    x, y = vectors
    return tf.math.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# Load the model with the custom function
siamese_model = tf.keras.models.load_model("siamese_model.h5", custom_objects={"euclidean_distance": euclidean_distance}, compile=False)

DATASET_PATH = "dataset/"
reference_features = load_reference_features(DATASET_PATH)

def classify_audio(file_path, siamese_model, reference_features, threshold=0.5):
    new_feature = extract_mfcc(file_path).T  # Transpose to match training shape

    best_match = None
    highest_similarity = -1

    print(f"\n🔍 Processing: {file_path}")  # Debugging

    for ref_path, (category, ref_feature) in reference_features.items():
        ref_feature = ref_feature.T  # Ensure shape consistency
        
        similarity_score = siamese_model.predict([
            np.expand_dims(new_feature, axis=0), 
            np.expand_dims(ref_feature, axis=0)
        ])[0][0]

        print(f"📌 Category: {category}, Similarity Score: {similarity_score}")  # Debugging

        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_match = category

    # Apply threshold
    if highest_similarity >= threshold:
        return best_match
    else:
        return "Unknown"


new_audio_file = "test_audio.wav"
predicted_category = classify_audio(new_audio_file, siamese_model, reference_features)
print(f"Predicted Category: {predicted_category}")