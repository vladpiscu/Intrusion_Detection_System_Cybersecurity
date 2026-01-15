# IoT3 - IoT Cybersecurity Detection System

A machine learning-based system for detecting cybersecurity attacks in IoT MQTT traffic, including bruteforce, DoS, flood, malformed data, and slowite attacks.

## Project Overview

This project implements a complete pipeline for:
- **Data Preprocessing**: Feature extraction, temporal analysis, and normalization
- **Model Training**: Random Forest and Isolation Forest classifiers
- **Fault Injection Testing**: Robustness evaluation with data corruption
- **Streaming Predictions**: Real-time attack detection

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Setup Instructions

### 1. Clone the Repository

```bash
cd /path/to/IoT3
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- tensorflow >= 2.13.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Project Structure

```
IoT3/
├── preprocessing/          # Data preprocessing pipeline
├── training/               # Model training scripts
├── fault_injection/        # Fault injection evaluation
├── streaming/              # Real-time prediction system
├── visualization/          # Visualization tools
├── models/                 # Saved models (gitignored)
├── processed_output/       # Processed train/val/test splits
├── MQTTEEB-D_Final_Dataset/ # Dataset (gitignored)
├── evaluation_results_*/   # Evaluation results (gitignored)
├── results_plots*/         # Plot outputs (gitignored)
└── requirements.txt        # Python dependencies
```

## Quick Start Workflow

### Step 1: Preprocess the Dataset

```bash
python preprocessing/preprocess_dataset.py
```

This will:
- Load data from `MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_dataset_All_Processed.csv`
- Extract features (packet-level, MQTT-level, temporal)
- Create train/validation/test splits (60/20/20)
- Save processed data to `processed_output/`

### Step 2: Train a Model

**Random Forest Classifier:**
```bash
python training/train_random_forest.py
```

**Isolation Forest (Anomaly Detection):**
```bash
python training/train_isolation_forest.py
```

Models are saved to `models/` directory.

### Step 3: Evaluate with Fault Injection (Optional)

Test model robustness with data corruption:

```bash
python fault_injection/evaluate_with_fault_injection.py \
    --model models/random_forest_model.joblib \
    --test processed_output/test.csv \
    --config fault_injection/fault_injection_config.json
```

### Step 4: Run Streaming Predictions (Optional)

For real-time predictions:

```bash
# First, save the preprocessor
python training/save_preprocessor.py \
    --train-data processed_output/train.csv \
    --output models/preprocessor.joblib

# Then run streaming predictions
python streaming/stream_predictions.py \
    --model models/random_forest_model.joblib \
    --input path/to/input_data.csv \
    --output predictions.csv \
    --preprocessor models/preprocessor.joblib
```

## Attack Types

The system detects the following attack types:
- **0**: bruteforce
- **1**: dos (Denial of Service)
- **2**: flood
- **3**: legitimate (normal traffic)
- **4**: malformed
- **5**: slowite

## Documentation

For detailed documentation, see:
- `PROJECT_STRUCTURE.md` - Project organization
- `preprocessing/PREPROCESSING_README.md` - Preprocessing details
- `training/TRAINING_README.md` - Training guide
- `fault_injection/QUICK_START_FAULT_INJECTION.md` - Fault injection guide
- `streaming/STREAMING_PREDICTIONS_README.md` - Streaming guide

## Notes

- The dataset folder (`MQTTEEB-D_Final_Dataset/`) is gitignored
- Model files and evaluation results are gitignored
- Make sure to activate the virtual environment before running any scripts
