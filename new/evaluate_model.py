import torch
import json
import re
from pathlib import Path
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from torch.utils.data import DataLoader
from DownloadandPreprocess import download_and_preprocess_data
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class TableEvaluator:
    """Evaluation metrics for table structure extraction"""
    
    def __init__(self, processor):
        self.processor = processor
        
    def calculate_bleu_score(self, predicted, target):
        """Calculate BLEU score between predicted and target HTML"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize
            pred_tokens = word_tokenize(predicted.lower())
            target_tokens = word_tokenize(target.lower())
            
            # Calculate BLEU
            score = sentence_bleu([target_tokens], pred_tokens)
            return score
        except ImportError:
            # Simple word-level similarity if NLTK not available
            pred_words = predicted.lower().split()
            target_words = target.lower().split()
            
            if len(target_words) == 0:
                return 1.0 if len(pred_words) == 0 else 0.0
            
            common_words = set(pred_words) & set(target_words)
            return len(common_words) / len(target_words)
    
    def exact_match_accuracy(self, predicted, target):
        """Calculate exact match accuracy"""
        # Normalize whitespace for comparison
        pred_normalized = ' '.join(predicted.split())
        target_normalized = ' '.join(target.split())
        return 1.0 if pred_normalized == target_normalized else 0.0
    
    def table_structure_score(self, predicted, target):
        """Evaluate table structure preservation"""
        scores = {}
        
        # Count table elements
        for tag in ['<table>', '<tr>', '<td>', '<th>']:
            pred_count = predicted.count(tag)
            target_count = target.count(tag)
            
            if target_count == 0:
                scores[f'{tag}_score'] = 1.0 if pred_count == 0 else 0.0
            else:
                scores[f'{tag}_score'] = min(pred_count, target_count) / target_count
        
        # Overall structure score
        scores['structure_score'] = np.mean(list(scores.values()))
        return scores

class ModelEvaluator:
    """Main evaluation class"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Initialize evaluator
        self.table_evaluator = TableEvaluator(self.processor)
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, pixel_values):
        """Generate prediction for a batch of images"""
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device)
            
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.config['max_length'],
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            # Decode predictions
            predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return predictions
    
    def evaluate_dataset(self, data_loader, split_name="test"):
        """Evaluate model on a dataset split"""
        print(f"\nEvaluating on {split_name} split...")
        
        all_predictions = []
        all_targets = []
        all_metrics = defaultdict(list)
        
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            # Get predictions
            predictions = self.predict(batch['pixel_values'])
            targets = [self.processor.tokenizer.decode(labels, skip_special_tokens=True) 
                      for labels in batch['labels']]
            
            # Calculate metrics for each sample in batch
            for pred, target in zip(predictions, targets):
                # BLEU score
                bleu = self.table_evaluator.calculate_bleu_score(pred, target)
                all_metrics['bleu'].append(bleu)
                
                # Exact match
                exact_match = self.table_evaluator.exact_match_accuracy(pred, target)
                all_metrics['exact_match'].append(exact_match)
                
                # Structure scores
                structure_scores = self.table_evaluator.table_structure_score(pred, target)
                for key, score in structure_scores.items():
                    all_metrics[key].append(score)
                
                all_predictions.append(pred)
                all_targets.append(target)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics, all_predictions, all_targets
    
    def run_full_evaluation(self, test_mode=True):
        """Run complete evaluation on all splits"""
        print("Starting Model Evaluation")
        print("=" * 40)
        
        # Load data
        datasets = download_and_preprocess_data(test_mode=test_mode)
        
        results = {}
        
        # Evaluate on validation and test sets
        for split_name in ['validation', 'test']:
            data_loader = DataLoader(
                datasets[split_name],
                batch_size=self.config['eval_batch_size'],
                shuffle=False,
                num_workers=2
            )
            
            metrics, predictions, targets = self.evaluate_dataset(data_loader, split_name)
            results[split_name] = {
                'metrics': metrics,
                'predictions': predictions[:10],  # Save first 10 for inspection
                'targets': targets[:10]
            }
        
        # Print results
        self.print_results(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for split_name, split_results in results.items():
            metrics = split_results['metrics']
            print(f"\n{split_name.upper()} SET:")
            print(f"  BLEU Score: {metrics['bleu']:.4f}")
            print(f"  Exact Match: {metrics['exact_match']:.4f}")
            print(f"  Structure Score: {metrics['structure_score']:.4f}")
            print(f"  Table Tag Score: {metrics['<table>_score']:.4f}")
            print(f"  Row Tag Score: {metrics['<tr>_score']:.4f}")
            print(f"  Cell Tag Score: {metrics['<td>_score']:.4f}")
    
    def save_results(self, results):
        """Save evaluation results to file"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = {}
        for split_name, split_results in results.items():
            summary[split_name] = split_results['metrics']
        
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")

def get_evaluation_config():
    """Get evaluation configuration"""
    return {
        'max_length': 512,
        'eval_batch_size': 4,
        'output_dir': './evaluation_results',
        'model_path': './models/best_model_hf'  # Path to saved model
    }

def main():
    """Main evaluation function"""
    config = get_evaluation_config()
    
    # Check if model exists
    model_path = Path(config['model_path'])
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(str(model_path), config)
    
    # Run evaluation
    print("Running evaluation in TEST mode (small dataset)...")
    results = evaluator.run_full_evaluation(test_mode=True)
    
    print("\n✅ Evaluation completed!")
    print("For full evaluation, change test_mode=False in the script")

if __name__ == "__main__":
    main()
