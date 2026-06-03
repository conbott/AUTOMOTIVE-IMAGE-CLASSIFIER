# Automotive Image Classification

This project is a deep learning-based image classification system designed to recognize and classify 196 different classes of cars using the **Stanford Cars** dataset. It includes a TensorFlow training pipeline and a web-based inference interface built with Streamlit.

## Features
- **Transfer Learning**: Utilizes pre-trained `EfficientNetB0` models via TensorFlow Keras for high-accuracy feature extraction.
- **Data Augmentation**: Robust training using TensorFlow's data augmentation layers.
- **Interactive UI**: A sleek Streamlit application to upload and classify car images in real-time.

## Project Structure
```text
automotive-classification/
├── data/               # The Stanford Cars dataset
│   ├── car_devkit/     # Raw annotations
│   ├── cars_train/     # Raw images
│   └── processed/      # Organized images (Train/Val by class)
├── notebooks/          # EDA and prototyping Jupyter notebooks
├── src/                # Core Python modules
│   ├── prepare_data.py # Data extraction script
│   ├── dataset.py      # TensorFlow Data loading pipeline
│   ├── model.py        # Neural network architecture definition
│   └── train.py        # Model training loop
├── app/                # Streamlit web application
│   └── main.py         # Streamlit UI script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup and Installation

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
If you downloaded the raw Stanford Cars dataset (`cars_train.tgz` and `car_devkit.tgz`), run the data prep script to organize the images into training and validation folders:
```bash
python src/prepare_data.py
```

## Training
To train the model from scratch or fine-tune it:
```bash
python src/train.py
```
This will automatically save the best performing model in `models/car_classifier.keras`.

## Inference Web App
To run the Streamlit app and classify your own images:
```bash
streamlit run app/main.py
```
