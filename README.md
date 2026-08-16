# NeuroPulse PH-CT AI Dashboard

A professional, interactive AI diagnostic dashboard for Pulmonary Hypertension CT analysis using DenseNet121 and Hybrid YOLOv8 architectures.

## Features
- Real-time Deep Learning Inference.
- GradCAM++ Attention Heatmaps.
- Interactive Plotly Confusion Matrix.
- Live Chart.js Convergence Curves.
- Multi-architecture Support (DenseNet vs. Hybrid).
- Drag & Drop Patient Imaging.

## Prerequisites
- Python 3.8+
- Recommended: NVIDIA GPU with CUDA for faster inference (though CPU works).

## Installation

1. **Clone or Extract** this project to your local machine.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask Server**:
   ```bash
   python server.py
   ```
2. **Access the Dashboard**:
   Open your browser and navigate to:
   `http://127.0.0.1:5000`

## Project Structure
- `server.py`: Flask Backend & PyTorch Inference Engine.
- `templates/index.html`: Dashboard Frontend.
- `static/css/style.css`: Premium Glassmorphism UI Styles.
- `static/js/main.js`: Interactive Frontend Logic (Chart.js & Plotly).
- `requirements.txt`: Python package dependencies.

## Dataset
This dashboard is optimized for the **PH-CT-V1** dataset available on Kaggle (`turkertuncer/ph-ct-v1`).
