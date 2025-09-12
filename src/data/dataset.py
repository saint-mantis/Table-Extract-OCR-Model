
from torch.utils.data import Dataset
from datasets import load_dataset
from preprocess import preprocess_image, clean_html
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class PubTabNetDataset(Dataset):
    def __init__(self, split, processor, max_samples=None):
        """
        split: 'train', 'validation', or 'test'
        processor: TrOCR processor for image and text
        max_samples: Optional[int], limit number of samples for quick tests
        """
        self.dataset = load_dataset(config.DATASET_NAME, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = preprocess_image(sample['img'])
        html = clean_html(sample['html'])
        inputs = self.processor(images=image, text=html, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        return {key: val.squeeze(0) for key, val in inputs.items()}
