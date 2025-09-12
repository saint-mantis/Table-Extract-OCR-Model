import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import json
import argparse
from pathlib import Path
import os

class TrOCRInference:
    """Inference class for TrOCR table extraction"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and processor
        print(f"Loading model from: {model_path}")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Preprocess image for TrOCR"""
        # Load and convert image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with TrOCR processor (resizes to 384x384)
        pixel_values = self.processor.feature_extractor(
            image, 
            return_tensors="pt"
        ).pixel_values
        
        return pixel_values.to(self.device)
    
    def predict(self, image_path, max_length=512):
        """Predict HTML from table image"""
        print(f"Processing: {image_path}")
        
        # Preprocess image
        pixel_values = self.preprocess_image(image_path)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode prediction
        predicted_html = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_html
    
    def predict_batch(self, image_paths, max_length=512):
        """Predict HTML for multiple images"""
        predictions = []
        
        for image_path in image_paths:
            try:
                prediction = self.predict(image_path, max_length)
                predictions.append({
                    'image_path': str(image_path),
                    'predicted_html': prediction,
                    'status': 'success'
                })
            except Exception as e:
                predictions.append({
                    'image_path': str(image_path),
                    'predicted_html': '',
                    'status': 'error',
                    'error': str(e)
                })
                print(f"❌ Error processing {image_path}: {e}")
        
        return predictions
    
    def save_predictions(self, predictions, output_file):
        """Save predictions to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"💾 Predictions saved to: {output_path}")

def format_html_output(html_text):
    """Format HTML for better readability"""
    # Add line breaks for major table elements
    html_text = html_text.replace('<table>', '<table>\n  ')
    html_text = html_text.replace('</table>', '\n</table>')
    html_text = html_text.replace('<tr>', '\n  <tr>')
    html_text = html_text.replace('</tr>', '</tr>')
    html_text = html_text.replace('<td>', '<td>')
    html_text = html_text.replace('</td>', '</td>')
    
    return html_text

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='TrOCR Table Extraction Inference')
    parser.add_argument('--model_path', type=str, default='./models/best_model_hf',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image file')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output_file', type=str, default='./outputs/predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for generation')
    
    args = parser.parse_args()
    
    print("TrOCR Table Extraction - Inference")
    print("=" * 40)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    # Initialize inference
    inference = TrOCRInference(str(model_path))
    
    # Collect image paths
    image_paths = []
    
    if args.image_path:
        # Single image
        if Path(args.image_path).exists():
            image_paths = [args.image_path]
        else:
            print(f"❌ Image not found: {args.image_path}")
            return
            
    elif args.image_dir:
        # Directory of images
        image_dir = Path(args.image_dir)
        if image_dir.exists():
            # Get all image files
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            for ext in extensions:
                image_paths.extend(image_dir.glob(f"*{ext}"))
                image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        else:
            print(f"❌ Directory not found: {args.image_dir}")
            return
    else:
        print("❌ Please provide either --image_path or --image_dir")
        return
    
    if not image_paths:
        print("❌ No valid images found")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Run predictions
    predictions = inference.predict_batch(image_paths, args.max_length)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    successful_predictions = 0
    for pred in predictions:
        if pred['status'] == 'success':
            successful_predictions += 1
            print(f"\n📁 {Path(pred['image_path']).name}")
            print("HTML Output:")
            formatted_html = format_html_output(pred['predicted_html'])
            print(formatted_html[:300] + "..." if len(formatted_html) > 300 else formatted_html)
        else:
            print(f"\n❌ Failed: {Path(pred['image_path']).name} - {pred['error']}")
    
    print(f"\n📊 Summary: {successful_predictions}/{len(predictions)} predictions successful")
    
    # Save predictions
    inference.save_predictions(predictions, args.output_file)

# Example usage functions
def demo_single_image():
    """Demo function for single image prediction"""
    model_path = "./models/best_model_hf"
    image_path = "./sample_table.png"  # Replace with your image path
    
    if not Path(model_path).exists():
        print("❌ Please train the model first")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Please provide a valid image at {image_path}")
        return
    
    inference = TrOCRInference(model_path)
    html_result = inference.predict(image_path)
    
    print("Predicted HTML:")
    print(format_html_output(html_result))

if __name__ == "__main__":
    main()
