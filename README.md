# Text Classification with LSTM
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Leya-LI/Text-classification-with-LSTM/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Leya-LI/LLM-API-Explorer.svg)](https://github.com/Leya-LI/Text-classification-with-LSTM/issues)
[![GitHub stars](https://img.shields.io/github/stars/Leya-LI/LLM-API-Explorer.svg)](https://github.com/Leya-LI/Text-classification-with-LSTM/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Leya-LI/LLM-API-Explorer.svg)](https://github.com/Leya-LI/Text-classification-with-LSTM/network)

## Overview

This repository contains a text classification project implemented using Long Short-Term Memory (LSTM) networks with PyTorch. 

## Key Features

- **Deep Learning Framework**: Utilizes PyTorch for building and training LSTM models.
- **Data Preprocessing**: Includes a script for preprocessing text data to prepare it for model training.
- **Model Training**: Provides a training script that leverages the preprocessed data to train the LSTM model.
- **Utility Functions**: Offers a collection of utility functions to support the main scripts.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.12
- torch~=2.5.1
- matplotlib~=3.10.0
- pandas~=2.2.3
- nltk~=3.9.1
- scikit-learn~=1.6.0

## Project Structure

```
project_root/
├── data/						# The directory of datasets
│   └── renMinRiBao/
│       ├── test_data.tsv
│       ├── train_data.tsv
├── model/						# The directory to save models
├── LSTM.py					# LSTM model definition
├── preprocess_data.py			# Preprocess the datasets
├── train.py					# The entry point of this project
└── utils.py					# Some useful functions
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Leya-LI/Text-classification-with-LSTM.git
   cd Text-classification-with-LSTM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data**:
   ```bash
   python preprocess_data.py
   ```

4. **Train the model**:
   ```bash
   python train.py
   ```

## Usage

- **LSTM Model**: `LSTM.py` contains the definition of the LSTM model architecture used for text classification.
- **Data Preprocessing**: `preprocess_data.py` is responsible for cleaning and preparing the dataset for training.
- **Training**: `train.py` is the entry point for training the LSTM model on the preprocessed data.
- **Utilities**: `utils.py` contains helper functions used across different scripts.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Leya-LI/Text-classification-with-LSTM/blob/main/LICENSE) file for details.

---

