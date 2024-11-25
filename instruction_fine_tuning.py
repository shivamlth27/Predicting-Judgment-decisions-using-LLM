import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report
import time
import gc

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def load_instructions(file_path="datasets/instruction_decision.csv"):
    """
    Load the instruction templates from CSV file
    """
    instructions_df = pd.read_csv(file_path)
    return instructions_df['Instructions'].tolist()

def format_instruction_input(row, instruction_templates):
    """
    Format the input with a randomly sampled instruction for each example
    """
    # Randomly sample an instruction template
    instruction = random.choice(instruction_templates)
    
    # Combine case name and details
    case_text = (
        f"Case Name: {row['Case Name']}\n\n"
        f"Case Details Input: {row['Input']}\n\n"
        f"Case Details Output: {row['Output']}"
    )
    
    # Format with instruction
    formatted_input = f"{instruction}\n\nInput: {case_text}"
    return formatted_input

def prepare_data(df, tokenizer, instruction_templates, max_length=512):
    """
    Prepare data with dynamic instructions for training
    """
    # Format inputs with randomly sampled instructions
    formatted_inputs = [
        format_instruction_input(row, instruction_templates) 
        for _, row in df.iterrows()
    ]
    
    # Tokenize inputs
    encodings = tokenizer(
        formatted_inputs,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Convert labels to tensor
    labels = torch.tensor(df['Label'].values)
    
    return encodings, labels

from tqdm import tqdm  # Import tqdm for progress bars

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Training for one epoch with instruction-based inputs, with a progress bar
    """
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    # Add a progress bar for the training loop
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    """
    Evaluate the model with a progress bar
    """
    model.eval()
    predictions = []
    true_labels = []
    
    # Add a progress bar for the evaluation loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    
    return predictions, true_labels

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data and instructions
    train_df = pd.read_csv('datasets/train.csv')
    # train_df = train_df.head(100)  # Smaller subset for testing
    instruction_templates = load_instructions()
    print(f"Training examples: {len(train_df)}")
    print(f"Loaded {len(instruction_templates)} instruction templates")
    
    # Initialize tokenizer and model
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Add instruction-specific tokens
    special_tokens = {
        'additional_special_tokens': ['Input:', 'Case Name:', 'Case Details Input:', 'Case Details Output:']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Resize token embeddings for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Prepare training data
    train_encodings, train_labels = prepare_data(train_df, tokenizer, instruction_templates)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        train_labels
    )
    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    # Training parameters
    epochs = 2
    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting instruction fine-tuning...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        
        # Re-prepare data with dynamic instructions
        train_encodings, train_labels = prepare_data(train_df, tokenizer, instruction_templates)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        
        # Train and evaluate
        avg_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        print(f'Average loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = "saved_models/roberta_instruction/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")
        
        predictions, true_labels = evaluate(
            model=model,
            dataloader=train_dataloader,
            device=device
        )
        print("\nTraining Metrics:")
        print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    main()
