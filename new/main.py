"""
Main runner script for TrOCR Table Extraction Project
This script provides a simple interface to run all components
"""

import argparse
import sys
from pathlib import Path

def run_setup():
    """Run project setup"""
    print("🔧 Running project setup...")
    import setup
    setup.main()

def run_preprocessing(test_mode=True):
    """Run data preprocessing"""
    print("📊 Running data preprocessing...")
    from DownloadandPreprocess import download_and_preprocess_data
    
    datasets = download_and_preprocess_data(test_mode=test_mode)
    print(f"✅ Preprocessing completed!")
    return datasets

def run_training(test_mode=True):
    """Run model training"""
    print("🚀 Running model training...")
    try:
        from train_model import TrOCRTrainer, get_training_config
        
        config = get_training_config()
        if test_mode:
            config['num_epochs'] = 2  # Shorter training for testing
            config['batch_size'] = 4  # Smaller batch size
        
        trainer = TrOCRTrainer(config)
        model = trainer.train(test_mode=test_mode)
        print("✅ Training completed!")
        return model
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def run_evaluation():
    """Run model evaluation"""
    print("📈 Running model evaluation...")
    try:
        from evaluate_model import ModelEvaluator, get_evaluation_config
        
        config = get_evaluation_config()
        evaluator = ModelEvaluator(config['model_path'], config)
        results = evaluator.run_full_evaluation(test_mode=True)
        print("✅ Evaluation completed!")
        return results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def run_inference(image_path=None, image_dir=None):
    """Run inference on images"""
    print("🔮 Running inference...")
    try:
        from inference import TrOCRInference
        
        model_path = "./models/best_model_hf"
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first!")
            return None
        
        inference = TrOCRInference(model_path)
        
        if image_path:
            result = inference.predict(image_path)
            print(f"Prediction for {image_path}:")
            print(result)
        elif image_dir:
            image_paths = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
            results = inference.predict_batch(image_paths)
            print(f"Processed {len(results)} images")
        
        print("✅ Inference completed!")
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='TrOCR Table Extraction Pipeline')
    parser.add_argument('command', choices=[
        'setup', 'preprocess', 'train', 'evaluate', 'inference', 'full-pipeline'
    ], help='Command to run')
    
    parser.add_argument('--test-mode', action='store_true', default=True,
                       help='Run in test mode with smaller dataset')
    parser.add_argument('--full-mode', action='store_true', 
                       help='Run with full dataset (overrides test-mode)')
    parser.add_argument('--image-path', type=str, help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, help='Directory of images for inference')
    
    args = parser.parse_args()
    
    # Determine mode
    test_mode = args.test_mode and not args.full_mode
    
    print("TrOCR Table Extraction Pipeline")
    print("=" * 40)
    print(f"Mode: {'TEST' if test_mode else 'FULL'} dataset")
    print(f"Command: {args.command}")
    print()
    
    # Run commands
    if args.command == 'setup':
        run_setup()
        
    elif args.command == 'preprocess':
        run_preprocessing(test_mode=test_mode)
        
    elif args.command == 'train':
        run_training(test_mode=test_mode)
        
    elif args.command == 'evaluate':
        run_evaluation()
        
    elif args.command == 'inference':
        run_inference(args.image_path, args.image_dir)
        
    elif args.command == 'full-pipeline':
        print("🔄 Running full pipeline...")
        
        # 1. Setup (if needed)
        print("\n1️⃣ Setup check...")
        try:
            import torch, transformers, datasets
            print("✅ Dependencies already installed")
        except ImportError:
            run_setup()
        
        # 2. Preprocessing
        print("\n2️⃣ Data preprocessing...")
        datasets = run_preprocessing(test_mode=test_mode)
        if not datasets:
            print("❌ Preprocessing failed, stopping pipeline")
            return
        
        # 3. Training
        print("\n3️⃣ Model training...")
        model = run_training(test_mode=test_mode)
        if not model:
            print("❌ Training failed, stopping pipeline")
            return
        
        # 4. Evaluation
        print("\n4️⃣ Model evaluation...")
        results = run_evaluation()
        
        print("\n🎉 Full pipeline completed!")
        print("\nNext steps:")
        print("- Check training results in ./logs/")
        print("- Check evaluation results in ./evaluation_results/")
        print("- Use inference.py to test on new images")

def show_project_status():
    """Show current project status"""
    print("📋 Project Status Check")
    print("=" * 30)
    
    components = {
        'Data preprocessing': './DownloadandPreprocess.py',
        'Training script': './train_model.py', 
        'Evaluation script': './evaluate_model.py',
        'Inference script': './inference.py',
        'Setup script': './setup.py',
        'Requirements': './requirements.txt'
    }
    
    for name, path in components.items():
        status = "✅" if Path(path).exists() else "❌"
        print(f"{status} {name}: {path}")
    
    # Check for trained model
    model_path = Path('./models/best_model_hf')
    model_status = "✅ Trained" if model_path.exists() else "❌ Not trained"
    print(f"{model_status} Model: {model_path}")
    
    print("\n📚 Usage Examples:")
    print("python main.py setup                    # Set up environment")
    print("python main.py preprocess               # Download and preprocess data")
    print("python main.py train                    # Train model")
    print("python main.py evaluate                 # Evaluate model")
    print("python main.py inference --image-path ./sample.png")
    print("python main.py full-pipeline            # Run everything")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_project_status()
    else:
        main()
