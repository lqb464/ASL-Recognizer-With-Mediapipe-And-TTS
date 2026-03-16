# README (English Version)

Vietnamese version: README.md

## Introduction

ASL-TALK is a Machine Learning project for recognizing sign language gestures (American Sign Language - ASL) from webcam video in real time. The system uses hand landmarks to represent hand motion and trains a sequence model to predict the corresponding sign.

The goal of this project is to build a complete ML pipeline including data collection, preprocessing, model training, and real-time inference.

This project is designed as a portfolio project to demonstrate machine learning pipeline design, code organization, and simple inference system development.

---

## Main Features

Collect sign language data from webcam
Import external video datasets
Extract hand landmarks from videos
Build and preprocess training datasets
Train a sequence model for gesture recognition
Run real-time inference using webcam
Organize the project using a standard ML pipeline structure

---

## Project Structure

```
ASL-TALK
│
├─ configs
│
├─ data
│  ├─ raw
│  ├─ interim
│  ├─ processed
│  └─ external
│
├─ models
│  ├─ checkpoints
│  └─ trained
│
├─ src
│  ├─ data
│  ├─ models
│  ├─ pipelines
│  ├─ utils
│  └─ __init__.py
│
├─ tests
│
├─ pyproject.toml
├─ README.md
├─ README-en.md   # README.md in English
├─ LICENSE
└─ .gitignore
```

---

## Machine Learning Pipeline

The data processing and training workflow includes the following steps:

1 Collect data from webcam
2 Store raw data
3 Convert raw data to interim format
4 Build processed training dataset
5 Train sequence model
6 Save model checkpoint
7 Run real-time inference

Pipeline flow:

```
data collection
      ↓
raw dataset
      ↓
dataset preprocessing
      ↓
processed dataset
      ↓
model training
      ↓
model checkpoint
      ↓
real-time inference
```

---

## Installation

Python 3.10 or higher is required.

Install dependencies:

```
pip install -e .
```

---

## Build Dataset

Run dataset pipeline:

```
python -m pipelines.run_dataset
```

This pipeline executes:

raw_to_interim
interim_to_processed

The processed dataset will be stored in:

```
data/processed
```

---

## Train Model

Run the training pipeline:

```
python -m pipelines.run_training
```

After training finishes, model checkpoints will be saved to:

```
models/checkpoints
```

---

## Run Inference

Run webcam inference:

```
python -m src.inference.infer_webcam
```

The system will:

1 Open webcam
2 Detect hands
3 Extract landmarks
4 Predict the corresponding sign

---

## Configuration

System parameters are controlled via YAML configuration files located in:

```
configs
```

Examples include:

```
data.yaml
model.yaml
train.yaml
utils.yaml
```

These configuration files allow adjusting parameters without modifying the source code.

---

## Results

Training and inference outputs are stored in:

```
artifacts
```

Including:

metrics
predictions
figures

---

## Learning Objectives

This project demonstrates:

machine learning pipeline design
dataset pipeline construction
sequence model training
standard ML project structure
simple inference pipeline implementation

---

## License

This project is released under the MIT License.