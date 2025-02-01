# Infant Cry Classification using Siamese Neural Network

## 📌 Project Description
Infant Cry Classification is a machine learning project that uses a **Siamese Neural Network (SNN)** to classify infant crying sounds into categories such as:
- **Belly Pain**
- **Hungry**
- **Tired**
- **Discomfort**
- **Burping**

The model is trained on a dataset of **infant cry audio recordings** and extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features to determine the similarity between different cry sounds. This system helps in identifying the reason behind an infant's cry based on audio patterns.

## 🛠 Features
✅ Siamese Neural Network (SNN) for one-shot learning  
✅ Audio feature extraction using **MFCC**  
✅ Dataset processing and **batch audio classification**  
✅ Trained on **infant cry recordings**  
✅ **Real-time classification** for new audio files  

## 📂 Project Structure
```
infant_cry_classification/
│-- dataset/                     # Folder containing infant cry audio files (.wav)
│   │-- belly_pain/
│   │-- hungry/
│   │-- tired/
│   │-- discomfort/
│   │-- burping/
│-- models/                      # Saved trained models
│-- train_snn.py                  # Train Siamese Neural Network
│-- test_snn.py                   # Test the trained model
│-- feature_extraction.py         # Extracts MFCC features from audio files
│-- siamese_model.py              # Defines the SNN architecture
│-- README.md                     # Project documentation
```

## 🏗 Installation & Setup
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/KaRTiK0821/infant_cry_classification.git
cd infant_cry_classification
```

### 2️⃣ **Install Dependencies**
Make sure you have Python 3.10+ installed. Install the required libraries:
```sh
pip install -r requirements.txt
```

### 3️⃣ **Prepare Dataset**
Place the **infant cry audio files** inside the `dataset/` folder. Ensure the structure follows the mentioned format.

### 4️⃣ **Train the Model**
Run the training script to train the Siamese Neural Network:
```sh
python train_snn.py
```
This will generate a trained model in the `models/` directory.

### 5️⃣ **Test the Model**
To classify a new infant cry audio file:
```sh
python test_snn.py --audio path/to/audio.wav
```

## 📈 Model Training & Performance
- **Feature Extraction:** MFCC (Mel-Frequency Cepstral Coefficients)
- **Neural Network:** Siamese Neural Network
- **Loss Function:** Contrastive Loss
- **Optimizer:** Adam
- **Dataset:** Infant Cry Audio Dataset (Manually Labeled)

## 📌 To-Do / Future Improvements
- 🔹 Improve classification accuracy with a larger dataset
- 🔹 Implement a **mobile app** for real-time infant cry detection
- 🔹 Add **spectrogram-based** feature extraction

## 📜 License
This project is open-source and licensed under the **MIT License**.

---
👨‍💻 Developed by **Kartik Samnotra** | [GitHub](https://github.com/KaRTiK0821)  
🚀 *Feel free to contribute, fork, and improve this project!*

