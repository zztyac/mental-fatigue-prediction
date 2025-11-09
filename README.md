# 🔬 Metal Multi-Axial Fatigue Life Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

<div align="center">
  <img src="fatigue_prediction/web/static/images/image.png" alt="Metal Fatigue Life Prediction System" width="600"/>
</div>

This system is a deep learning-based metal multi-axial fatigue life prediction platform that integrates multiple advanced deep learning models (CNN, LSTM, Transformer). By analyzing material features and time series data, it achieves accurate prediction of metal material fatigue life. The system provides a user-friendly Web interface supporting data upload, model training, prediction, and result visualization.


## 📑 Table of Contents

- [📋 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [⚙️ Environment Setup](#️-environment-setup)
- [📁 Directory Structure](#-directory-structure)
- [📖 Usage Guide](#-usage-guide)
  - [1. 🚀 Starting the System](#1--starting-the-system)
  - [2. 📊 Data Preparation](#2--data-preparation)
  - [3. 🎯 Model Training](#3--model-training)
  - [4. 🔮 Prediction Usage](#4--prediction-usage)
  - [5. 📈 Model Comparison](#5--model-comparison)
- [📚 API Documentation](#-api-documentation)
- [❓ FAQ](#-faq)
- [🔧 Maintenance and Updates](#-maintenance-and-updates)

## ✨ Features

- 🤖 **Multi-Model Support**: Integrates multiple deep learning models including CNN, LSTM, and Transformer
- 🔄 **Data Processing**: Supports preprocessing and feature engineering for stress and strain data
- 📊 **Model Comparison**: Provides comparative analysis and visualization of different model performances
- 🌐 **Web Interface**: Intuitive user interface supporting interactive operations
- 📈 **Result Visualization**: Detailed prediction result display and performance metric analysis
- 📝 **Logging**: Complete logging of training and prediction processes

## ⚙️ Environment Setup

1. **Create a virtual environment:**
```bash
conda create -n mental python=3.10
conda activate mental
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> 💡 **Note**: For GPU support, install PyTorch with CUDA from [PyTorch official website](https://pytorch.org/get-started/locally/)

## 📁 Directory Structure

```
metal_fatigue/
├── 📦 fatigue_prediction/    # Main prediction module
│   ├── 🌐 web/              # Web application
│   ├── 🤖 models/           # Model definitions
│   ├── 📊 data/             # Data processing
│   ├── ⚙️ configs/          # Configuration files
│   ├── 🎓 training/         # Training modules
│   └── 🛠️ utils/            # Utility functions
├── 📁 dataset/              # Dataset directory
│   ├── All data_Strain/     # Strain data
│   └── All data_Stress/     # Stress data
├── 📈 results/              # Result output
├── 📤 uploads/              # Uploaded files
└── 🚀 start.py             # Startup script
```

## 📖 Usage Guide

### 1. 🚀 Starting the System

```bash
python start.py
```

Default configuration:
- Host: localhost
- Port: 5000
- Debug mode: Enabled

Optional parameters:
- `--host`: Specify host address
- `--port`: Specify port number
- `--debug`: Enable debug mode
- `--log-level`: Set log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

### 2. 📊 Data Preparation

1. Data format requirements:
   - Stress data: CSV format containing time series stress data
   - Strain data: CSV format containing time series strain data
   - Material features: CSV format containing basic material property parameters

2. Data directory structure:
   - Place stress data in `dataset/All data_Stress/` directory
   - Place strain data in `dataset/All data_Strain/` directory

### 3. 🎯 Model Training

1. Upload training data through the Web interface
2. Select the model to use (CNN/LSTM/Transformer)
3. Set training parameters (learning rate, batch size, number of epochs, etc.)
4. Start training and monitor progress

### 4. 🔮 Prediction Usage

1. Upload data to be predicted
2. Select a trained model
3. Execute prediction
4. View prediction results and visualization charts

### 5. 📈 Model Comparison

Use the model comparison tool:
```bash
python model_comparison.py
```

## 📚 API Documentation

The system provides the following main API endpoints:

### 📤 Data Management
- `POST /api/upload` - Upload data files
- `GET /api/data/list` - Get data list

### 🤖 Model Operations
- `POST /api/model/train` - Start model training
- `GET /api/model/status` - Get training status
- `POST /api/model/predict` - Execute prediction

### 📊 Result Queries
- `GET /api/results` - Get prediction results
- `GET /api/results/visualization` - Get visualization data

## ❓ FAQ

### Q: What should I do if the system fails to start?
**A:** Check the following:
- ✅ Ensure all dependencies are correctly installed
- 🔌 Check if the port is occupied
- 📝 View error messages in log files

### Q: What should I do if training is very slow?
**A:** You can:
- 🚀 Use GPU acceleration (if available)
- 📉 Reduce batch size
- ⚙️ Adjust model parameters
- 📊 Reduce training data volume

### Q: What should I do if prediction results are inaccurate?
**A:** Try the following methods:
- 📈 Increase training data volume
- 🎛️ Adjust model hyperparameters
- 🔄 Try different model architectures
- 🔍 Check data preprocessing steps

## 🔧 Maintenance and Updates

### 📝 Log Management
- 📂 Log file location: `fatigue_prediction/logs/`
- 📊 Regularly check log file sizes
- 🔄 Configure log rotation strategy

### 💾 Data Backup
- 📦 Regularly backup training data
- 💿 Backup model checkpoints
- 💾 Save important prediction results

---

<div align="center">

**Made with ❤️ for Metal Fatigue Life Prediction**

⭐ Star this repo if you find it helpful!

</div>
