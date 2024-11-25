import os
import random
import pandas as pd
import numpy as np
import torch
import gc
import wandb
import re
import nltk
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from nltk.tokenize import sent_tokenize

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# to off this warning what to do

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def print_gpu_utilization():
    """Print current GPU memory utilization"""
    if torch.cuda.is_available():
        print("Memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Memory cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

class LegalTextDataset(Dataset):
    """Dataset for legal text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[ds\]', '', text)
        text = re.sub(r'###.*?:', '', text)
        text = re.sub(r'\d+\.', '', text)
        return text.strip()

    def truncate_text(self, text):
        """Smartly truncate text to fit max_length"""
        sentences = sent_tokenize(text)
        truncated = []
        current_length = 0
        
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if current_length + len(tokens) + 2 <= self.max_length:
                truncated.append(sentence)
                current_length += len(tokens)
            else:
                break
        
        return ' '.join(truncated)

    def __getitem__(self, idx):
        text = self.clean_text(str(self.texts[idx]))
        text = self.truncate_text(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class LegalTextTrainer:
    """Trainer for legal text classification model"""
    def __init__(
        self,
        model_name='roberta-base',
        output_dir='saved_models/legal',
        gradient_accumulation_steps=2
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # Initialize mixed precision training
        self.scaler = GradScaler()

    def create_dataloaders(self, train_df, val_df=None, batch_size=8):
        """Create training and validation dataloaders"""
        train_dataset = LegalTextDataset(
            train_df['Input'].values,
            train_df['Label'].values,
            self.tokenizer
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = None
        if val_df is not None:
            val_dataset = LegalTextDataset(
                val_df['Input'].values,
                val_df['Label'].values,
                self.tokenizer
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=SequentialSampler(val_dataset),
                num_workers=4,
                pin_memory=True
            )
        
        return train_dataloader, val_dataloader

    def train(self, train_df, val_df=None, epochs=5, batch_size=8, learning_rate=2e-5):
        """Train the model"""
        # Initialize wandb
        wandb.init(project="legal-text-classification")
        
        # Create dataloaders
        train_dataloader, val_dataloader = self.create_dataloaders(
            train_df, val_df, batch_size
        )
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        print_gpu_utilization()
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_dataloader, desc='Training')
            
            for batch_idx, batch in enumerate(train_pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                total_train_loss += loss.item() * self.gradient_accumulation_steps
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
                
                # Clean memory after each batch
                del outputs, loss
                clear_gpu_memory()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f'\nAverage training loss: {avg_train_loss:.4f}')
            
            # Validation phase
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                print(f"Validation Loss: {val_metrics['loss']:.4f}")
                print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Validation F1: {val_metrics['f1']:.4f}")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_model('best_model')
                
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1']
                })
            
            # Clean memory after each epoch
            clear_gpu_memory()
        
        wandb.finish()

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        eval_pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch in eval_pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            del outputs, preds
            clear_gpu_memory()
        
        metrics = classification_report(all_labels, all_preds, output_dict=True)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': metrics['accuracy'],
            'f1': metrics['weighted avg']['f1-score'],
            'precision': metrics['weighted avg']['precision'],
            'recall': metrics['weighted avg']['recall']
        }

    def save_model(self, model_name):
        """Save model and tokenizer"""
        save_dir = self.output_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        
        self.tokenizer.save_pretrained(save_dir)
        print(f"\nModel saved to {save_dir}")

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Load data
    train_df = pd.read_csv('datasets/train_ft.csv')
    val_df = pd.read_csv('datasets/val_ft.csv')
    
    # use only 100 samples for training
    train_df = train_df.sample(100)
    val_df = val_df.sample(100)
    # Initialize trainer
    trainer = LegalTextTrainer(
        model_name='roberta-base',
        output_dir='saved_models/legal',
        gradient_accumulation_steps=2
    )
    
    # Train model with optimized parameters
    trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5
    )

if __name__ == "__main__":
    main()