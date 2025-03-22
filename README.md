# LSTM Architectures with Parameter Fine-Tuning for Enhanced Turkish Sign Language Action Recognition

https://github.com/user-attachments/assets/6d428be7-ceef-40d8-902f-f4712f4e9893
## Overview
This repository contains the implementation of a deep learning model for Turkish Sign Language (TSL) action recognition. The model leverages Long Short-Term Memory (LSTM) networks combined with convolutional layers (Conv1D) and parameter tuning techniques to achieve high accuracy in recognizing 15 different Turkish Sign Language gestures.

## Features
- **Sign Language Recognition**: Recognizes 15 distinct TSL gestures.
- **Deep Learning Model**: Utilizes LSTM and Conv1D layers for feature extraction and classification.
- **Parameter Optimization**: Implements a tuning process using Keras Tuner to optimize model performance.
- **Real-Time Processing**: Designed to run on standard laptop webcams for real-time sign language recognition.
- **High Accuracy**: Achieves up to 98% validation accuracy and 90% test accuracy.

## Dataset
The dataset consists of 3000 videos capturing 15 different TSL gestures. Each video is recorded at 15 FPS, with each frame preprocessed using the MediaPipe holistic feature extraction technique.
- **Training Set**: 2100 videos (70%)
- **Validation Set**: 450 videos (15%)
- **Test Set**: 450 videos (15%)

## Model Architecture
The model consists of:
1. **Feature Extraction**:
   - MediaPipe Holistic for pose, face, and hand landmarks.
   - Normalization of extracted features.
2. **Neural Network Layers**:
   - Conv1D layers with varying kernel sizes (3,5,7) to capture spatial features.
   - Batch normalization and dropout layers to enhance generalization.
   - LSTM layers to capture temporal dependencies.
   - A final softmax layer for classification.
3. **Optimization**:
   - Adam optimizer with a dynamic learning rate.
   - Dropout regularization to prevent overfitting.
   - Random search parameter tuning using Keras Tuner.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TSL-Recognition.git
   cd TSL-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Train the model**:
   ```bash
   python train.py
   ```
2. **Run real-time gesture recognition**:
   ```bash
   python recognize.py
   ```
3. **Evaluate the model**:
   ```bash
   python evaluate.py
   ```

## Dependencies
- Python 3.8+
- TensorFlow 2.8
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Keras Tuner
- CUDA (for GPU acceleration)

## Results
- **Validation Accuracy**: 98%
- **Test Accuracy**: 90%
- **Loss Reduction**: Optimized with early stopping and adaptive learning rates.

## Acknowledgments
This research was conducted as part of a Master of Science thesis at Eastern Mediterranean University under the supervision of Asst. Prof. Dr. Ahmet Ünveren.

## License
This project is licensed under the MIT License.

## Contact
For further inquiries, please contact [your email here].

