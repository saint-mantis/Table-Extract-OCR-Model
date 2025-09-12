"""
Evaluation metrics for TrOCR table extraction
"""

import re
import json
from collections import defaultdict
import difflib
from pathlib import Path

class TableMetrics:
    """Comprehensive metrics for table extraction evaluation"""
    
    @staticmethod
    def bleu_score_simple(predicted, target):
        """Simple BLEU score implementation"""
        pred_words = predicted.lower().split()
        target_words = target.lower().split()
        
        if len(target_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate n-gram precision for n=1,2,3,4
        scores = []
        for n in range(1, 5):
            pred_ngrams = [tuple(pred_words[i:i+n]) for i in range(len(pred_words)-n+1)]
            target_ngrams = [tuple(target_words[i:i+n]) for i in range(len(target_words)-n+1)]
            
            if len(target_ngrams) == 0:
                scores.append(0.0)
                continue
            
            matches = sum(1 for ng in pred_ngrams if ng in target_ngrams)
            precision = matches / max(1, len(pred_ngrams))
            scores.append(precision)
        
        # Geometric mean of n-gram precisions
        if all(s > 0 for s in scores):
            bleu = (scores[0] * scores[1] * scores[2] * scores[3]) ** 0.25
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, len(pred_words) / max(1, len(target_words)))
        return bleu * bp
    
    @staticmethod
    def exact_match(predicted, target):
        """Exact match accuracy"""
        pred_clean = ' '.join(predicted.split())
        target_clean = ' '.join(target.split())
        return 1.0 if pred_clean == target_clean else 0.0
    
    @staticmethod
    def character_error_rate(predicted, target):
        """Calculate character-level error rate"""
        if len(target) == 0:
            return 0.0 if len(predicted) == 0 else 1.0
        
        # Calculate Levenshtein distance
        distance = difflib.SequenceMatcher(None, predicted, target)
        similarity = distance.ratio()
        return 1.0 - similarity
    
    @staticmethod
    def table_structure_metrics(predicted, target):
        """Evaluate table structure preservation"""
        metrics = {}
        
        # Define table elements to check
        table_elements = {
            'table': r'<table[^>]*>',
            'tr': r'<tr[^>]*>',
            'td': r'<td[^>]*>',
            'th': r'<th[^>]*>',
            'thead': r'<thead[^>]*>',
            'tbody': r'<tbody[^>]*>'
        }
        
        for element, pattern in table_elements.items():
            pred_count = len(re.findall(pattern, predicted, re.IGNORECASE))
            target_count = len(re.findall(pattern, target, re.IGNORECASE))
            
            if target_count == 0:
                metrics[f'{element}_precision'] = 1.0 if pred_count == 0 else 0.0
            else:
                metrics[f'{element}_precision'] = min(pred_count, target_count) / target_count
            
            if pred_count == 0:
                metrics[f'{element}_recall'] = 1.0 if target_count == 0 else 0.0
            else:
                metrics[f'{element}_recall'] = min(pred_count, target_count) / pred_count
        
        # Overall structure score
        precision_scores = [v for k, v in metrics.items() if 'precision' in k]
        recall_scores = [v for k, v in metrics.items() if 'recall' in k]
        
        metrics['structure_precision'] = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        metrics['structure_recall'] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        # F1 score for structure
        p, r = metrics['structure_precision'], metrics['structure_recall']
        metrics['structure_f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def calculate_all_metrics(predicted, target):
        """Calculate all metrics for a prediction"""
        metrics = {}
        
        # Basic metrics
        metrics['bleu'] = TableMetrics.bleu_score_simple(predicted, target)
        metrics['exact_match'] = TableMetrics.exact_match(predicted, target)
        metrics['character_error_rate'] = TableMetrics.character_error_rate(predicted, target)
        
        # Structure metrics
        structure_metrics = TableMetrics.table_structure_metrics(predicted, target)
        metrics.update(structure_metrics)
        
        return metrics

class EvaluationReport:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self):
        self.metrics_calculator = TableMetrics()
    
    def generate_report(self, predictions_file, output_file=None):
        """Generate comprehensive evaluation report"""
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        report = {
            'summary': {},
            'detailed_metrics': {},
            'error_analysis': {},
            'sample_predictions': []
        }
        
        all_metrics = defaultdict(list)
        successful_predictions = 0
        
        # Process each prediction
        for i, pred in enumerate(predictions):
            if pred['status'] == 'success':
                successful_predictions += 1
                
                # Calculate metrics (if target available)
                if 'target_html' in pred:
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        pred['predicted_html'], 
                        pred['target_html']
                    )
                    
                    for metric_name, value in metrics.items():
                        all_metrics[metric_name].append(value)
                
                # Save sample predictions for inspection
                if i < 5:  # First 5 samples
                    report['sample_predictions'].append({
                        'image_path': pred['image_path'],
                        'predicted_html': pred['predicted_html'][:300] + "...",
                        'prediction_length': len(pred['predicted_html'])
                    })
        
        # Calculate summary statistics
        report['summary'] = {
            'total_predictions': len(predictions),
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / len(predictions) if predictions else 0.0
        }
        
        # Calculate average metrics
        if all_metrics:
            report['detailed_metrics'] = {
                metric_name: {
                    'mean': float(sum(values) / len(values)),
                    'std': float((sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5),
                    'min': float(min(values)),
                    'max': float(max(values))
                }
                for metric_name, values in all_metrics.items()
            }
        
        # Error analysis
        failed_predictions = [p for p in predictions if p['status'] != 'success']
        report['error_analysis'] = {
            'failed_count': len(failed_predictions),
            'common_errors': [p.get('error', 'Unknown') for p in failed_predictions[:10]]
        }
        
        # Save report
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Evaluation report saved to: {output_file}")
        
        return report
    
    def print_summary(self, report):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        summary = report['summary']
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Successful: {summary['successful_predictions']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        
        if 'detailed_metrics' in report and report['detailed_metrics']:
            print("\nKey Metrics:")
            metrics = report['detailed_metrics']
            
            if 'bleu' in metrics:
                print(f"  BLEU Score: {metrics['bleu']['mean']:.4f} ± {metrics['bleu']['std']:.4f}")
            if 'exact_match' in metrics:
                print(f"  Exact Match: {metrics['exact_match']['mean']:.4f}")
            if 'structure_f1' in metrics:
                print(f"  Structure F1: {metrics['structure_f1']['mean']:.4f}")
        
        if report['error_analysis']['failed_count'] > 0:
            print(f"\nErrors: {report['error_analysis']['failed_count']} failed predictions")

def main():
    """Demo usage of metrics"""
    # Example usage
    predicted = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"
    target = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"
    
    metrics = TableMetrics.calculate_all_metrics(predicted, target)
    
    print("Demo Metrics Calculation:")
    print("="*30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
