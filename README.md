# TrOCR Table Extraction Project

This project implements a TrOCR-based model for extracting structured tables from scientific document images and converting them to HTML.

## Structure
- `src/data/`: Data loading and preprocessing
- `src/models/`: Model setup and configuration
- `src/training/`: Training pipeline
- `src/inference/`: Inference pipeline
- `src/utils/`: Evaluation metrics and utilities
- `src/config.py`: Configuration and hyperparameters

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare PubTabNet dataset and update config paths.
3. Run training:
   ```bash
   python src/training/train.py
   ```
4. Run inference:
   ```bash
   python src/inference/predict.py --image path/to/image.png
   ```

## Deliverables
- Trained model and processor
- Inference and evaluation scripts
- Documentation and logs
