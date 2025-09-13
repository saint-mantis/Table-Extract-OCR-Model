import os
import re
from pathlib import Path
from datasets import load_dataset
from transformers import TrOCRProcessor
from PIL import Image
import torch
from torch.utils.data import Dataset
import json

class PubTabNetDataset(Dataset):
    """Simple dataset class for PubTabNet preprocessing"""
    
    def __init__(self, split="train", max_samples=None):
        print(f"Loading {split} split from PubTabNet...")
        
        # Load dataset from Hugging Face - using dataset with HTML annotations
        self.dataset = load_dataset("apoidea/pubtabnet-html", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Initialize TrOCR processor
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def clean_html(self, html):
        """Clean HTML markup"""
        if not html:
            return html
        
        # Check if HTML is in JSON format (like the PubTabNet structure)
        try:
            import json
            html_data = json.loads(html)
            # If it's JSON, extract text content or convert to simple HTML
            if isinstance(html_data, dict) and 'cells' in html_data:
                # Simple conversion - just extract text tokens
                text_content = []
                for cell in html_data.get('cells', []):
                    tokens = cell.get('tokens', [])
                    text_content.extend(tokens)
                html = ' '.join(text_content)
            elif isinstance(html_data, dict):
                html = str(html_data)
        except:
            # If not JSON, treat as regular HTML
            pass
        
        # Remove style attributes
        html = re.sub(r'\s*style\s*=\s*["\'][^"\']*["\']', '', html)
        html = re.sub(r'\s*class\s*=\s*["\'][^"\']*["\']', '', html)
        
        # Normalize whitespace
        html = re.sub(r'\s+', ' ', html)
        html = re.sub(r'>\s+<', '><', html)
        
        return html.strip()
    
    def __getitem__(self, idx):
        """Get preprocessed sample"""
        sample = self.dataset[idx]
        
        # Get image
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image (resize to 384x384 and normalize)
        pixel_values = self.processor.feature_extractor(
            image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Get and clean HTML
        html = sample.get('html', '')
        cleaned_html = self.clean_html(html)
        
        # Tokenize HTML
        labels = self.processor.tokenizer(
            cleaned_html,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'original_html': html,
            'cleaned_html': cleaned_html
        }

def download_and_preprocess_data(test_mode=True):
    """Download and preprocess PubTabNet dataset"""
    
    print("Starting PubTabNet dataset preprocessing...")
    
    # Create directories
    os.makedirs("./data", exist_ok=True)
    
    # Load datasets - this dataset has train and validation splits
    if test_mode:
        print("Running in test mode - loading 50 samples per split")
        train_dataset = PubTabNetDataset("train", max_samples=50)
        val_dataset = PubTabNetDataset("validation", max_samples=20)
        
        # Create a small test set from validation data
        val_full = PubTabNetDataset("validation", max_samples=40)
        test_dataset = torch.utils.data.Subset(val_full, range(20, 40))
        
    else:
        print("Loading full dataset...")
        train_dataset = PubTabNetDataset("train")
        val_dataset = PubTabNetDataset("validation")
        
        # Create test set from validation (split validation 80/20)
        val_full = PubTabNetDataset("validation")
        val_size = len(val_full)
        val_split_point = int(0.8 * val_size)
        
        val_dataset = torch.utils.data.Subset(val_full, range(0, val_split_point))
        test_dataset = torch.utils.data.Subset(val_full, range(val_split_point, val_size))
    
    datasets = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
    
    # Test preprocessing with first sample from train dataset
    print("\nTesting preprocessing with first sample:")
    if hasattr(train_dataset, 'dataset'):
        # If it's our custom dataset class
        sample = train_dataset[0]
        sample_count = len(train_dataset)
    else:
        # If it's a Subset, get the underlying dataset
        sample = train_dataset.dataset[train_dataset.indices[0]]
        sample_count = len(train_dataset.dataset)
        
    print(f"Image shape: {sample['pixel_values'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Original HTML length: {len(sample['original_html'])}")
    print(f"Cleaned HTML length: {len(sample['cleaned_html'])}")
    print(f"HTML preview: {sample['cleaned_html'][:150]}...")
    
    # Save dataset info
    dataset_info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'image_size': [384, 384],
        'max_sequence_length': 512
    }
    
    with open('./data/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset Info:")
    print(f"Train samples: {dataset_info['train_size']}")
    print(f"Validation samples: {dataset_info['val_size']}")
    print(f"Test samples: {dataset_info['test_size']}")
    print(f"Saved dataset info to ./data/dataset_info.json")
    
    return datasets

if __name__ == "__main__":
    # Download and preprocess data - CHANGED TO FULL DATASET
    datasets = download_and_preprocess_data(test_mode=False)
    
    print("\nPreprocessing completed successfully!")
    print("You can now use these datasets for training your TrOCR model.")
