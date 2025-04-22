# A-Deep-Le-for-DDoS-Attack-Detection
Overview
This repository contains the implementation of a hybrid deep learning model combining Gated Recurrent Units (GRU) and Bidirectional Long Short-Term Memory (BiLSTM) networks for detecting Distributed Denial of Service (DDoS) attacks in network traffic. The model achieves exceptional accuracy of 99.9954% with 30 epochs of training.
# Dataset
This implementation uses the CICDDoS2019 dataset, which contains benign and DDoS attack traffic data. The dataset includes various DDoS attack types and is widely used for evaluating DDoS detection systems.
# Objectives
Develop a novel hybrid deep learning architecture combining GRU and BiLSTM for efficient DDoS attack detection
Analyze and compare the performance of individual GRU and BiLSTM models with the hybrid approach
Demonstrate the impact of training epochs on detection accuracy and false positive rates
Provide a computationally efficient solution for real-time DDoS attack detection
Create a reproducible framework for future research in network security and DDoS detection
Offer students and researchers a practical implementation to understand deep learning applications for cybersecurity
 
# Requirements

Python 3.8+
TensorFlow 2.x
Keras
NumPy
Pandas
Scikit-learn
Matplotlib

Usage

Clone this repository

git clone https://github.com/yourusername/GRU-BiLSTM-DDoS-Detection.git
cd GRU-BiLSTM-DDoS-Detection

Install the required packages

pip install -r requirements.txt

Download and prepare the CICDDoS2019 dataset

python prepare_data.py

Train the model

python train.py

Evaluate the model

python evaluate.py
# Hyperparameters
The model uses the following hyperparameters:

Epochs: 10, 20, 30
Optimizer: Adam
Loss function: Binary cross-entropy
Activation functions: ReLU, Sigmoid
Learning rate: 0.001
Batch size: 64
