# ASL-TALK: Real-time Sign Language Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 📌 Introduction

**ASL-TALK** is a real-time American Sign Language (ASL) recognition system. It leverages hand landmark coordinates to represent gestures and utilizes sequence models to predict signs from live webcam feeds.

The system is primarily trained on an external dataset and supports optional real-time data collection via webcam for extending and personalizing the dataset.

This is a **portfolio project** designed to demonstrate:
* End-to-end Machine Learning Pipeline design.
* Professional code organization and modular software engineering.
* Scalable inference system implementation.

---

## ✨ Key Features

* **Data Engineering**: Integration of external datasets from Kaggle as the primary data source, with optional real-time webcam data collection.
* **Feature Extraction**: Mediapipe-based hand landmark extraction for efficient processing of both external and custom data.
* **Preprocessing Pipeline**: Automated workflow converting data from `raw` → `interim` → `processed`.
* **Model Training**: Robust sequence model training (LSTM/GRU) with checkpoint management.
* **Real-time Inference**: High-performance gesture prediction via webcam.

---

## 📊 Dataset

This project uses a combination of external data and optional custom data:

### Primary Dataset
- ASL Citizen Dataset (Kaggle)  
- Source: https://www.kaggle.com/datasets/abd0kamel/asl-citizen

This dataset serves as the main training source, providing diverse sign language samples for better generalization.

Since the dataset does not include hand landmarks, a custom extraction pipeline (`import_external_videos.py`) is used to process raw videos into structured landmark sequences.

### Optional Data Collection
- Real-time data can be collected via webcam using the built-in collection module.
- This allows users to:
  - Extend the dataset with new signs
  - Improve performance on specific users
  - Experiment with custom data distributions

Combining a curated external dataset with real-time collected data helps improve model robustness and flexibility.

---

## 📂 Project Structure

```
ASL-TALK
│
├─ configs/          
├─ data/             
├─ models/           
├─ src/              
├─ tests/            
├─ pyproject.toml    
└─ README.md
```

---

## 🚀 Machine Learning Pipeline

The workflow is strictly organized into automated stages:

1. **Dataset Loading**: Import external data from Kaggle → `data/raw`.
2. **Optional Collection**: Capture additional data via webcam.
3. **Preprocessing**: Extract landmarks → `data/interim` → Normalize → `data/processed`.
4. **Training**: Train sequence model → Save to `models/checkpoints`.
5. **Inference**: Load trained model → Live webcam prediction.

---

## 🛠 Installation

Python 3.10 or higher is required.

```bash
git clone https://github.com/username/ASL-TALK.git
cd ASL-TALK
pip install -e .
```

---

## 📖 Usage Guide

### 1. Data Collection
```bash
python -m src.data.collect_raw_data
```

### 2. Build Dataset
```bash
python -m pipelines.run_dataset
```

### 3. Model Training
```bash
python -m pipelines.run_training
```

### 4. Run Inference
```bash
python -m src.tests.test_infer_webcam
```

---

## ⚙️ Configuration

All system parameters are managed via YAML files in the `configs/` directory.

---

## 📝 License

This project is licensed under the MIT License.