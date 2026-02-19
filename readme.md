# PlantVision AI - Deployment Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open browser at `http://localhost:8501`

## Model Details
- Ensemble of EfficientNetB3 + ResNet50
- 99.36% test accuracy
- 0.9902 macro F1 score
- 38 plant disease classes