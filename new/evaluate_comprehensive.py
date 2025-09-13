"""
Enhanced evaluation script with comprehensive accuracy metrics
"""

import torch
import json
import time
import os
from pathlib import Path
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from torch.utils.data import DataLoader
from DownloadandPreprocess import download_and_preprocess_data
from tqdm import tqdm
import numpy as np
import re
from collections import defaultdict

class ComprehensiveEvaluator:
    """Enhanced evaluator with multiple accuracy metrics"""
    
    def __init__(self, model_path="./models/best_model_hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first!")
            return
            
        print(f"Loading model from: {model_path}")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def calculate_bleu_score(self, predicted, target):
        """Calculate BLEU score"""
        pred_tokens = predicted.lower().split()
        target_tokens = target.lower().split()
        
        if len(target_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        # Simple BLEU calculation (unigram precision)
        common_tokens = set(pred_tokens) & set(target_tokens)
        if len(pred_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(set(pred_tokens))
        recall = len(common_tokens) / len(set(target_tokens))
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def exact_match_accuracy(self, predicted, target):
        """Calculate exact match accuracy"""
        pred_clean = ' '.join(predicted.split())
        target_clean = ' '.join(target.split())
        return 1.0 if pred_clean.lower() == target_clean.lower() else 0.0
    
    def partial_match_accuracy(self, predicted, target, threshold=0.5):
        """Calculate partial match accuracy"""
        pred_words = set(predicted.lower().split())
        target_words = set(target.lower().split())
        
        if len(target_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        intersection = pred_words & target_words
        similarity = len(intersection) / len(target_words)
        return 1.0 if similarity >= threshold else 0.0
    
    def table_structure_score(self, predicted, target):
        """Evaluate table structure preservation"""
        table_tags = ['<table>', '<tr>', '<td>', '<th>', '<thead>', '<tbody>']
        scores = {}
        
        for tag in table_tags:
            pred_count = predicted.lower().count(tag.lower())
            target_count = target.lower().count(tag.lower())
            
            if target_count == 0:
                scores[tag] = 1.0 if pred_count == 0 else 0.5
            else:
                scores[tag] = min(pred_count, target_count) / target_count
        
        return np.mean(list(scores.values())), scores
    
    def evaluate_dataset(self, dataset, split_name, max_samples=100):
        """Evaluate on a dataset split"""
        print(f"\n📊 Evaluating {split_name} split...")
        
        # Limit samples for faster evaluation
        num_samples = min(max_samples, len(dataset))
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        metrics = defaultdict(list)
        predictions = []
        targets = []
        inference_times = []
        
        samples_processed = 0
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            if samples_processed >= num_samples:
                break
                
            pixel_values = batch['pixel_values'].to(self.device)
            batch_targets = [self.processor.tokenizer.decode(labels, skip_special_tokens=True) 
                           for labels in batch['labels']]
            
            # Inference with timing
            start_time = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=256,  # Reasonable length
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(batch_targets))
            
            # Decode predictions
            batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Calculate metrics for each sample in batch
            for pred, target in zip(batch_predictions, batch_targets):
                # Basic metrics
                bleu = self.calculate_bleu_score(pred, target)
                exact_match = self.exact_match_accuracy(pred, target)
                partial_match = self.partial_match_accuracy(pred, target)
                
                # Structure metrics
                struct_score, struct_details = self.table_structure_score(pred, target)
                
                # Store metrics
                metrics['bleu'].append(bleu)
                metrics['exact_match'].append(exact_match)
                metrics['partial_match'].append(partial_match)
                metrics['structure_score'].append(struct_score)
                
                # Store predictions and targets
                predictions.append(pred)
                targets.append(target)
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        avg_metrics['avg_inference_time'] = np.mean(inference_times)
        avg_metrics['samples_evaluated'] = samples_processed
        
        return avg_metrics, predictions[:5], targets[:5]  # Return first 5 for inspection
    
    def run_comprehensive_evaluation(self, test_mode=False):
        """Run complete evaluation"""
        print("🚀 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Load datasets
        datasets = download_and_preprocess_data(test_mode=test_mode)
        
        results = {}
        
        # Evaluate on validation and test sets
        for split_name in ['validation', 'test']:
            if split_name in datasets:
                max_samples = 50 if test_mode else 200  # More samples for full evaluation
                
                metrics, predictions, targets = self.evaluate_dataset(
                    datasets[split_name], split_name, max_samples
                )
                
                results[split_name] = {
                    'metrics': metrics,
                    'sample_predictions': predictions,
                    'sample_targets': targets
                }
        
        # Print results
        self.print_results(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def print_results(self, results):
        """Print comprehensive results"""
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 60)
        
        for split_name, split_results in results.items():
            metrics = split_results['metrics']
            
            print(f"\n🎯 {split_name.upper()} SET RESULTS:")
            print(f"  📊 Samples Evaluated: {metrics['samples_evaluated']}")
            print(f"  ⚡ Avg Inference Time: {metrics['avg_inference_time']:.3f}s per sample")
            print(f"  📝 BLEU Score: {metrics['bleu']['mean']:.4f} ± {metrics['bleu']['std']:.4f}")
            print(f"  🎯 Exact Match Accuracy: {metrics['exact_match']['mean']:.4f} ({metrics['exact_match']['mean']*100:.1f}%)")
            print(f"  🔍 Partial Match Accuracy: {metrics['partial_match']['mean']:.4f} ({metrics['partial_match']['mean']*100:.1f}%)")
            print(f"  🏗️  Table Structure Score: {metrics['structure_score']['mean']:.4f}")
            
            print(f"\n  📋 Sample Predictions:")
            for i, (pred, target) in enumerate(zip(split_results['sample_predictions'][:3], 
                                                  split_results['sample_targets'][:3])):
                print(f"    Example {i+1}:")
                print(f"      Predicted: {pred[:100]}...")
                print(f"      Target:    {target[:100]}...")
                print(f"      Match: {'✅' if pred[:50] == target[:50] else '❌'}")
    
    def save_results(self, results):
        """Save detailed results"""
        output_dir = Path("./evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "comprehensive_evaluation.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_used': str(self.device),
            'summary_metrics': {}
        }
        
        for split_name, split_results in results.items():
            metrics = split_results['metrics']
            summary['summary_metrics'][split_name] = {
                'bleu_score': round(metrics['bleu']['mean'], 4),
                'exact_match_accuracy': round(metrics['exact_match']['mean'], 4),
                'partial_match_accuracy': round(metrics['partial_match']['mean'], 4),
                'structure_score': round(metrics['structure_score']['mean'], 4),
                'inference_time_per_sample': round(metrics['avg_inference_time'], 3)
            }
        
        with open(output_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}")

def main():
    """Main evaluation function"""
    
    # Check if model exists
    model_path = "./models/best_model_hf"
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("Please run training first: python train_model.py")
        return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(model_path)
    
    # Run evaluation
    print("Running evaluation on test dataset...")
    print("For full evaluation, the script will automatically use appropriate sample sizes.")
    
    results = evaluator.run_comprehensive_evaluation(test_mode=False)
    
    print("\n🎉 Comprehensive evaluation completed!")
    print("Check './evaluation_results/' for detailed results")

if __name__ == "__main__":
    main()
