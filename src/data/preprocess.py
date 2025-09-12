from PIL import Image
import numpy as np

def preprocess_image(image_path, size=(384, 384)):
    """Load and preprocess an image for TrOCR."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    img_np = np.array(img) / 255.0  # Normalize to [0,1]
    return img_np

def clean_html(html):
    """Clean and standardize HTML markup for table extraction."""
    html = html.replace('\n', '').replace('\t', '')
    # Further cleaning as needed
    return html.strip()
