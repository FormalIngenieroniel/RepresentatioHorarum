# 👨‍🔬 Scientific Investigation on CSLR and Temporal Sensitivity 💻

Continuous Sign Language Recognition (CSLR) is a specialized field within computer vision and artificial intelligence that focuses on interpreting and understanding sign language from video sequences. Unlike isolated sign recognition, CSLR deals with continuous signing where multiple signs are connected and form complete phrases or sentences.

---

## 💽 The core components of our intelligent CSLR system include:

- **Autoencoder (AE)**: Learns compressed representations of sign language video sequences

- **3D Convolutional Networks**: Extract spatial and temporal features from video data

- **Bidirectional GRU**: Processes temporal sequences in both directions for better context understanding

- **Multi-dataset Support**: Handles different sign language datasets with unified preprocessing

- This architecture transforms raw video input into meaningful gloss labels, enabling automatic understanding of continuous sign language communication.

---

## 🚀 Features

📹 **Video Sequence Processing**: Handles video input for continuous sign language recognition in a word level

🎯 **Data Preprocessing**: Clear pipeline for video sequence normalization and labeling

🧠 **Deep Learning Architecture**: Combines 3D CNNs with bidirectional GRU networks

🔄 **Autoencoder Framework**: Learns efficient representations of sign language patterns through pretext tasks

📊 **Multi-dataset Support**: Compatible with WLASL, ISL, and SLOVO datasets

🏷️ **Gloss Analisis**: Creates a visual representation of the behavior of each gloss for video sequences

📈 **Temporal Evaluation**: Comprehensive metrics and temporal sensitivity analysis

---

## 🧠 Technical Architecture

### Core Components

- 3D Convolutional Autoencoder

 - Learns compressed representations of video sequences
 - Processes spatial and temporal dimensions simultaneously
 - Captures motion patterns in sign language gestures

- Bidirectional GRU Network

  - Processes temporal sequences in forward and backward directions
  - Maintains context from both past and future states
  - Enhances classification accuracy for continuous signing

- Multi-variant Processing

  - Generates 5 variants of each video sequence for robustness
  - Data augmentation techniques for improved generalization
  - Help organize the sequence in a latent space using triplet loss

## 📋 Dataset Support

### WLASL Dataset


- American Sign Language (ASL) video collection
- Structured JSON metadata with gloss labels
- Multiple video files with standardized naming convention


### ISL Dataset

- Indian Sign Language video sequences
- Custom format requiring preprocessing adaptation
- More videos but less glosses


### SLOVO Dataset

- Slovenian Sign Language recordings
- Russian-origin data requiring translation and adaptation
- Special preprocessing for language-specific features

## 🔧 Data Preprocessing Pipeline

1. Data Restructuring


Converts various dataset formats into unified H5 structure

Standardizes video sequences with consistent frame dimensions

Normalizes temporal sequences to fixed length


2. Label Translation


Automatic translation of non-English gloss labels

Manual verification of translation accuracy

JSON mapping for language-specific terms


3. Sequence Normalization


Resizes frames to standardized dimensions (120x160)

Normalizes sequence length to 7 frames

Handles aspect ratio preservation and quality enhancement


4. Background Processing


Optional background removal for improved feature extraction

Focus on signer silhouette and gesture patterns

Enhanced contrast and clarity adjustments



🏗️ Implementation Structure

Model Architecture

css
copy
download
Input Video Sequence (N x 7 x 120 x 160 x 1)
    ↓
3D Convolutional Autoencoder
    ↓
Compressed Feature Representation
    ↓
Bidirectional GRU Processing
    ↓
Temporal Context Integration
    ↓
Gloss Classification Output

Training Components


Reconstruction Loss: Ensures accurate video sequence reconstruction

Triplet Loss (Inter-gloss): Separates different gloss categories

Triplet Loss (Temporal): Maintains temporal consistency

Combined Loss Function: Weighted combination of all loss terms



📊 Performance Metrics

Training Metrics


Total Loss (combined reconstruction and classification)

Reconstruction Loss (video sequence accuracy)

Inter-gloss Triplet Loss (category separation)

Temporal Triplet Loss (sequence consistency)


Evaluation Metrics


Classification Accuracy per gloss category

Temporal Sensitivity Analysis

Cross-dataset Performance Comparison

Per-sequence Classification Confidence



🛠️ Technical Requirements

Dependencies


PyTorch / TensorFlow for deep learning framework

OpenCV for video processing

NumPy for numerical computations

H5Py for dataset handling

Googletrans for automatic label translation


Hardware Requirements


GPU with CUDA support recommended

Minimum 8GB RAM for processing video sequences

Storage space for multiple dataset formats

Sufficient VRAM for 3D CNN operations


Input Format


Video files in standard formats (MP4, AVI, etc.)

JSON metadata with gloss labels and sequence information

H5 files with preprocessed sequences for training



🔬 Experimental Results

The system has been evaluated across multiple datasets with different configurations:



WLASL Experiments: 2-label to full dataset classification

ISL Dataset Tests: Irish Sign Language recognition performance

SLOVO Evaluations: Slovenian Sign Language processing accuracy

Cross-dataset Analysis: Generalization across different signing styles


Each experiment includes:



Training progression analysis

Temporal sensitivity evaluation

Confusion matrix generation

Performance comparison tables



📈 Future Enhancements


🔄 Real-time Processing: Optimized inference for live sign language recognition

🌐 Multi-language Support: Extended dataset compatibility

📱 Mobile Deployment: Lightweight model variants for mobile devices

🎯 Fine-tuning Capabilities: Domain-specific adaptation mechanisms

📊 Advanced Analytics: Detailed performance breakdown and error analysis



📖 Usage Examples

Dataset Preparation

python
copy
download
# Load and preprocess datasets
from cslr_preprocessing import DataProcessor

processor = DataProcessor()
processed_data = processor.prepare_dataset('WLASL', num_variants=5)

Model Training

python
copy
download
# Initialize and train the CSLR model
from cslr_model import CSLRAutoencoder

model = CSLRAutoencoder(input_shape=(7, 120, 160, 1))
model.train(processed_data, epochs=100)

Inference

python
copy
download
# Predict gloss labels for new sequences
predictions = model.predict(video_sequence)
gloss_labels = processor.decode_predictions(predictions)


👨‍💻 Author

Project created by **Daniel Bernal**
