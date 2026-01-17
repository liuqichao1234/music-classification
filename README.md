# Music Genre Classification with CNN-Transformer Hybrid Model

## Project Overview
```
This project implements a high-performance music genre classification model based on a **CNN-Transformer dual-branch fusion architecture**. It combines CNN's ability to extract local spectral features with Transformer's advantage in modeling global temporal dependencies, and integrates the Gated-ECA Attention mechanism to enhance feature discriminability. The model is validated on two public datasets (GTZAN & FMA-Medium) to ensure generalization.


## Environment Requirements
Install required dependencies via pip:
```bash
pip install tensorflow>=2.8.0 librosa>=0.10.0 numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.12.0 scikit-learn>=1.0.2

```

## File Structure
├── attention/          # Implementation of Gated-ECA Attention mechanism
├── dataload/           # Data loading & preprocessing scripts (Mel spectrogram generation, audio chunking)
├── Generative Model/   # Diffusion model for minority class data augmentation
├── model/              # Model architecture definition, training & evaluation scripts
├── data/               # Dataset storage directory (place GTZAN/FMA-Medium here)
└── README.md           # Project documentation

## Model Architecture

The core dual-branch fusion structure:

1.  **Preliminary Feature Extraction**
2.  **Transformer Branch**
3.  **CNN Branch**
4.  **Fusion & Classification**

## Usage Guide

1.  **Clone the Repository**
git clone [your-repo-url]
cd [your-repo-name]

2.   **Prepare Datasets**
    
    Download public music genre datasets and place them in the `data/` directory:
    
    -   GTZAN: [https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
    -   FMA-Medium: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
    
3.  **Data Preprocessing**
    
    Generate Mel spectrogram features (replace STFT) via the preprocessing script:   
    ```
    python dataload/preprocess.py
    ```



