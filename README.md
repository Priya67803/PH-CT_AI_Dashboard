# NeuroPulse PH-CT AI Dashboard

A professional, interactive AI diagnostic dashboard for Pulmonary Hypertension CT analysis using DenseNet121 and Hybrid YOLOv8 architectures.

## Features
- Real-time Deep Learning Inference
- GradCAM++ Attention Heatmaps
- Interactive Plotly Confusion Matrix
- Live Chart.js Convergence Curves
- Multi-architecture Support (DenseNet vs. Hybrid)
- Drag & Drop Patient Imaging

## Prerequisites
- Python 3.12+
- Recommended: NVIDIA GPU with CUDA for faster inference (though CPU works)

## Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask Server**:
   ```bash
   python app.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to: `http://127.0.0.1:7860`

## Deployment (Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo `Priya67803/PH-CT_AI_Dashboard`
4. Render auto-detects `render.yaml` and configures everything

## Project Structure
- `app.py`: Flask Backend & PyTorch Inference Engine
- `templates/index.html`: Dashboard Frontend
- `static/css/style.css`: Premium Glassmorphism UI Styles
- `static/js/main.js`: Interactive Frontend Logic (Chart.js & Plotly)
- `requirements.txt`: Python package dependencies
- `render.yaml`: Render.com deployment configuration

## Dataset
This dashboard is optimized for the **PH-CT-V1** dataset available on Kaggle (`turkertuncer/ph-ct-v1`).
