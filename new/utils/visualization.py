"""
Utility functions for TrOCR table extraction project
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns
from PIL import Image, ImageDraw
import re

def plot_training_curves(log_file_path):
    """Plot training and validation loss curves"""
    with open(log_file_path, 'r') as f:
        log_data = json.load(f)
    
    train_losses = log_data['train_losses']
    val_losses = log_data['val_losses']
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('TrOCR Training Progress', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best validation loss annotation
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    plt.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                xy=(best_val_idx + 1, best_val_loss),
                xytext=(best_val_idx + 1, best_val_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    return plt

def plot_evaluation_metrics(results_file_path):
    """Plot evaluation metrics comparison"""
    with open(results_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract metrics for plotting
    splits = list(results.keys())
    metrics = ['bleu', 'exact_match', 'structure_score']
    
    data = []
    for split in splits:
        for metric in metrics:
            if metric in results[split]['metrics']:
                data.append({
                    'Split': split.title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': results[split]['metrics'][metric]
                })
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    split_names = [d['Split'] for d in data if d['Metric'] == 'Bleu']
    bleu_scores = [d['Score'] for d in data if d['Metric'] == 'Bleu']
    exact_scores = [d['Score'] for d in data if d['Metric'] == 'Exact Match']
    struct_scores = [d['Score'] for d in data if d['Metric'] == 'Structure Score']
    
    x = np.arange(len(split_names))
    width = 0.25
    
    ax1.bar(x - width, bleu_scores, width, label='BLEU Score')
    ax1.bar(x, exact_scores, width, label='Exact Match')
    ax1.bar(x + width, struct_scores, width, label='Structure Score')
    
    ax1.set_xlabel('Dataset Split')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(split_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Detailed structure metrics
    structure_metrics = ['<table>_score', '<tr>_score', '<td>_score', '<th>_score']
    for split in splits:
        split_data = results[split]['metrics']
        metric_names = [m.replace('_score', '').replace('<', '').replace('>', '') for m in structure_metrics if m in split_data]
        metric_values = [split_data[m] for m in structure_metrics if m in split_data]
        
        ax2.plot(metric_names, metric_values, marker='o', label=f'{split.title()} Split', linewidth=2)
    
    ax2.set_xlabel('HTML Tag Type')
    ax2.set_ylabel('Structure Preservation Score')
    ax2.set_title('Table Structure Preservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_prediction(image_path, predicted_html, target_html=None):
    """Visualize image with predicted HTML"""
    fig, axes = plt.subplots(1, 2 if target_html else 1, figsize=(15, 8))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Load and display image
    image = Image.open(image_path)
    axes[0].imshow(image)
    axes[0].set_title(f"Input Image\n{Path(image_path).name}")
    axes[0].axis('off')
    
    # Display prediction
    pred_text = f"Predicted HTML:\n{predicted_html[:200]}..."
    axes[0].text(0, -50, pred_text, fontsize=8, wrap=True, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Display target if provided
    if target_html and len(axes) > 1:
        target_text = f"Target HTML:\n{target_html[:200]}..."
        axes[1].text(0.1, 0.5, target_text, fontsize=10, wrap=True,
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1].set_title("Target HTML")
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_html_complexity(html_text):
    """Analyze HTML structure complexity"""
    analysis = {}
    
    # Count table elements
    table_tags = ['<table>', '<tr>', '<td>', '<th>', '<thead>', '<tbody>']
    for tag in table_tags:
        analysis[f'{tag}_count'] = html_text.count(tag)
    
    # Calculate table dimensions (approximate)
    tr_count = analysis['<tr>_count']
    td_count = analysis['<td>_count']
    th_count = analysis['<th>_count']
    
    analysis['estimated_rows'] = tr_count
    analysis['estimated_cols'] = max(1, (td_count + th_count) // max(1, tr_count))
    analysis['total_cells'] = td_count + th_count
    analysis['has_header'] = th_count > 0
    analysis['html_length'] = len(html_text)
    
    return analysis

def compare_predictions(predictions_file):
    """Compare predictions with targets and generate analysis"""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'total_predictions': len(data),
        'successful_predictions': 0,
        'complexity_analysis': [],
        'common_errors': []
    }
    
    for item in data:
        if item['status'] == 'success':
            analysis['successful_predictions'] += 1
            
            # Analyze complexity
            complexity = analyze_html_complexity(item['predicted_html'])
            analysis['complexity_analysis'].append(complexity)
    
    # Calculate average complexity
    if analysis['complexity_analysis']:
        avg_complexity = {}
        for key in analysis['complexity_analysis'][0].keys():
            if isinstance(analysis['complexity_analysis'][0][key], (int, float)):
                avg_complexity[f'avg_{key}'] = np.mean([c[key] for c in analysis['complexity_analysis']])
        
        analysis['average_complexity'] = avg_complexity
    
    return analysis

def create_demo_html_file(predictions, output_path="./outputs/demo_results.html"):
    """Create an HTML demo file showing predictions"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TrOCR Table Extraction Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
            .image { max-width: 400px; margin: 10px 0; }
            .html-output { background: #f5f5f5; padding: 10px; border-radius: 5px; }
            .success { border-left: 5px solid #4CAF50; }
            .error { border-left: 5px solid #f44336; }
        </style>
    </head>
    <body>
        <h1>TrOCR Table Extraction Results</h1>
    """
    
    for i, pred in enumerate(predictions):
        status_class = "success" if pred['status'] == 'success' else "error"
        
        html_content += f"""
        <div class="result {status_class}">
            <h3>Result {i+1}: {Path(pred['image_path']).name}</h3>
            <p><strong>Status:</strong> {pred['status'].title()}</p>
        """
        
        if pred['status'] == 'success':
            html_content += f"""
            <div class="html-output">
                <strong>Predicted HTML:</strong><br>
                <pre>{pred['predicted_html']}</pre>
            </div>
            """
        else:
            html_content += f"<p><strong>Error:</strong> {pred.get('error', 'Unknown error')}</p>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Demo HTML created: {output_path}")

if __name__ == "__main__":
    # Demo usage
    print("TrOCR Utilities Demo")
    print("=" * 20)
    print("Available functions:")
    print("- plot_training_curves(log_file)")
    print("- plot_evaluation_metrics(results_file)")
    print("- visualize_prediction(image_path, predicted_html)")
    print("- analyze_html_complexity(html_text)")
    print("- compare_predictions(predictions_file)")
    print("- create_demo_html_file(predictions)")
