import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import get_linear_schedule_with_warmup
import json
from tqdm import tqdm
import logging
from pathlib import Path
from DownloadandPreprocess import download_and_preprocess_data, PubTabNetDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrOCRTrainer:
    """Training class for TrOCR table extraction"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        Path(config['model_save_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize model and processor
        self.processor = TrOCRProcessor.from_pretrained(config['model_name'])
        self.model = VisionEncoderDecoderModel.from_pretrained(config['model_name']).to(self.device)
        
        # Set decoder configuration for proper training
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, test_mode=True):
        """Prepare training data"""
        logger.info("Preparing datasets...")
        
        # Load preprocessed datasets
        datasets = download_and_preprocess_data(test_mode=test_mode)
        
        # Create data loaders
        train_loader = DataLoader(
            datasets['train'], 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            datasets['validation'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            datasets['test'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_model(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        save_dir = Path(self.config['model_save_dir'])
        
        # Save regular checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Also save the model and processor for easy loading
            self.model.save_pretrained(save_dir / "best_model_hf")
            self.processor.save_pretrained(save_dir / "best_model_hf")
            
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, test_mode=True):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(test_mode=test_mode)
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = len(train_loader) * self.config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {val_loss:.4f}")
            
            self.save_model(epoch, val_loss, is_best=is_best)
            
            # Save training progress
            self.save_training_log()
        
        logger.info("Training completed!")
        return self.model
    
    def save_training_log(self):
        """Save training progress"""
        log_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'best_val_loss': min(self.val_losses) if self.val_losses else float('inf')
        }
        
        log_file = Path(self.config['log_dir']) / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

def get_training_config():
    """Get training configuration"""
    return {
        # Model settings
        'model_name': 'microsoft/trocr-base-printed',
        
        # Training hyperparameters
        'batch_size': 8,  # Adjust based on GPU memory
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 3,  # Start small for testing
        
        # Directories
        'model_save_dir': './models',
        'log_dir': './logs',
        
        # Other settings
        'max_length': 512,
        'warmup_ratio': 0.1,
        'save_every_epoch': True
    }

def main():
    """Main training function"""
    print("TrOCR Table Extraction - Training")
    print("=" * 40)
    
    # Get configuration
    config = get_training_config()
    
    # Display configuration
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = TrOCRTrainer(config)
    
    # Start training (test mode for quick validation)
    print(f"\nStarting training in TEST mode (small dataset)...")
    print("Change test_mode=False for full dataset training")
    
    try:
        model = trainer.train(test_mode=True)
        print("\n✅ Training completed successfully!")
        print(f"Best model saved to: {config['model_save_dir']}/best_model_hf/")
        print(f"Training logs saved to: {config['log_dir']}/training_log.json")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return None
    
    return model

if __name__ == "__main__":
    model = main()
