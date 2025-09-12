# TrOCR Table Extraction Project

A deep learning project that uses Microsoft's TrOCR model to extract structured table information from images of scientific documents and convert them to HTML markup.

## 🎯 Project Overview

This project fine-tunes the TrOCR (Transformer-based OCR) model on the PubTabNet dataset to:
- Extract text from table images
- Preserve table structure information  
- Generate clean HTML markup
- Enable automated document analysis

## 📋 Project Structure

```
├── main.py                     # Main runner script
├── setup.py                    # Environment setup
├── requirements.txt            # Python dependencies
├── DownloadandPreprocess.py    # Data loading and preprocessing
├── train_model.py             # Model training script
├── evaluate_model.py          # Model evaluation
├── inference.py               # Inference on new images
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting and visualization
├── models/                   # Saved model checkpoints
├── data/                     # Dataset cache and info
├── logs/                     # Training logs
└── outputs/                  # Inference results
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or download the project
cd your-project-directory

# Set up environment and install dependencies
python setup.py
```

### 2. Full Pipeline (Recommended)
```bash
# Run the complete pipeline in test mode
python main.py full-pipeline

# For full dataset (slower but better results)
python main.py full-pipeline --full-mode
```

### 3. Step-by-Step Usage

#### Data Preprocessing
```bash
python main.py preprocess              # Test mode (fast)
python main.py preprocess --full-mode  # Full dataset
```

#### Training
```bash
python main.py train                   # Test mode (2 epochs)
python main.py train --full-mode       # Full training
```

#### Evaluation
```bash
python main.py evaluate
```

#### Inference on New Images
```bash
# Single image
python main.py inference --image-path ./sample_table.png

# Directory of images
python main.py inference --image-dir ./test_images/
```

## 📊 Model Performance

### Target Metrics (from project description):
- **BLEU Score**: >85%
- **Exact Match Accuracy**: >70%  
- **Structure Preservation**: High fidelity table structure

### Training Configuration:
- **Base Model**: microsoft/trocr-base-printed
- **Dataset**: PubTabNet (500K+ table images)
- **Image Size**: 384x384 pixels
- **Max Sequence Length**: 512 tokens
- **Training**: 15-20 epochs, AdamW optimizer

## 🛠️ Technical Details

### Model Architecture:
- **Vision Encoder**: DeiT (Data-efficient image Transformer)
- **Text Decoder**: RoBERTa-based language model
- **Fine-tuning**: End-to-end on table structure data

### Data Pipeline:
1. Download PubTabNet from Hugging Face Hub
2. Preprocess images (resize, normalize)
3. Clean HTML annotations (remove styling)
4. Create train/validation/test splits
5. Tokenize HTML for sequence generation

### Evaluation Metrics:
- **BLEU Score**: Text generation quality
- **Exact Match**: Perfect prediction accuracy
- **Structure Metrics**: Table element preservation
- **Character Error Rate**: Character-level accuracy

## 📁 File Descriptions

### Core Scripts:
- **`main.py`**: Central command interface for all operations
- **`DownloadandPreprocess.py`**: Handles PubTabNet dataset loading and preprocessing
- **`train_model.py`**: Implements TrOCR fine-tuning with proper training loop
- **`evaluate_model.py`**: Comprehensive evaluation with multiple metrics
- **`inference.py`**: Production inference pipeline for new images

### Utilities:
- **`utils/metrics.py`**: Evaluation metrics (BLEU, exact match, structure scores)
- **`utils/visualization.py`**: Training curves, result visualization, HTML demos
- **`setup.py`**: Automated environment setup and dependency checking

## 💻 System Requirements

### Minimum Requirements:
- Python 3.8+
- 8GB RAM
- 10GB free disk space
- Internet connection (for downloading dataset)

### Recommended:
- Python 3.9+
- 16GB+ RAM
- GPU with 6GB+ VRAM (for faster training)
- 50GB free disk space (for full dataset)

## 🔧 Configuration

### Training Configuration (train_model.py):
```python
config = {
    'batch_size': 8,           # Adjust based on GPU memory
    'learning_rate': 5e-5,     # Learning rate
    'num_epochs': 3,           # Number of training epochs
    'max_length': 512,         # Maximum sequence length
}
```

### Evaluation Configuration (evaluate_model.py):
```python
config = {
    'eval_batch_size': 4,      # Evaluation batch size
    'model_path': './models/best_model_hf'  # Path to trained model
}
```

## 📈 Usage Examples

### Training from Scratch:
```bash
# Quick test training
python main.py train

# Full training (15-20 epochs)
python main.py train --full-mode
```

### Evaluating a Trained Model:
```bash
python main.py evaluate
```

### Making Predictions:
```bash
# Single image prediction
python inference.py --image_path ./table_image.png

# Batch prediction
python inference.py --image_dir ./images/ --output_file ./results.json
```

## 🐛 Troubleshooting

### Common Issues:

1. **ImportError**: Run `python setup.py` to install dependencies
2. **CUDA out of memory**: Reduce batch_size in training config
3. **Model not found**: Train the model first using `python main.py train`
4. **Dataset download slow**: The dataset is large (11GB), be patient
5. **Low accuracy**: Increase training epochs or check data quality

### Performance Tips:
- Use GPU for faster training (`CUDA_VISIBLE_DEVICES=0`)
- Start with test mode to verify everything works
- Monitor training curves in `./logs/training_log.json`
- Check evaluation metrics in `./evaluation_results/`

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the training logs in `./logs/`
3. Verify all dependencies are installed with `python setup.py`

## 🏆 Expected Results

After successful training, you should achieve:
- Training loss < 1.0
- Validation BLEU score > 0.8
- Good table structure preservation
- Readable HTML output for table images

## 📝 Next Steps

After running the basic pipeline:
1. Experiment with different hyperparameters
2. Add data augmentation for better generalization
3. Implement beam search tuning
4. Add support for different table formats
5. Deploy as a web service or API

---

**Happy coding! 🚀**
