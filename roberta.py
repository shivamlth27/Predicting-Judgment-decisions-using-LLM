import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import f1_score, confusion_matrix
import tqdm
import progressbar
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import time
import datetime
import gc

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def check_gpu():
    """Check GPU availability and print detailed information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        for i in range(gpu_count):
            print(f"\nGPU {i} details:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.0f}MB")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.0f}MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.0f}MB")
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("No GPU available, using CPU")

def optimize_memory():
    """Optimize GPU memory usage and provide memory stats."""
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Print memory stats after optimization
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} memory after optimization:")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.0f}MB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.0f}MB")

def load_data():
    """Load and prepare the dataset."""
    train_set = pd.read_csv('datasets/train.csv')
    test_set = pd.read_csv("datasets/test.csv")
    return train_set, test_set

def att_masking(input_ids):
    """Create attention masks for input sequences."""
    return [[int(token_id > 0) for token_id in sent] for sent in input_ids]

def grouped_input_ids(tokenizer, all_toks):
    """Process tokens into grouped input IDs."""
    splitted_toks = []
    l = 0
    r = 510
    while l < len(all_toks):
        splitted_toks.append(all_toks[l:min(r, len(all_toks))])
        l += 410
        r += 410

    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    e_sents = []
    for l_t in splitted_toks:
        l_t = [CLS] + l_t + [SEP]
        encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
        e_sents.append(encoded_sent)

    e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="post", truncating="pre")
    att_masks = att_masking(e_sents)
    return e_sents, att_masks

@torch.no_grad()
def generate_training_data(dataf, tokenizer):
    """Generate training data from dataframe with GPU optimization."""
    all_input_ids, all_att_masks, all_labels = [], [], []
    
    for i in progressbar.progressbar(range(len(dataf['Input']))):
        text = dataf['Input'].iloc[i]
        toks = tokenizer.tokenize(text)
        if len(toks) > 10000:
            toks = toks[len(toks)-10000:]

        splitted_input_ids, splitted_att_masks = grouped_input_ids(tokenizer, toks)
        doc_label = dataf['Label'].iloc[i]
        
        for j in range(len(splitted_input_ids)):
            all_input_ids.append(splitted_input_ids[j])
            all_att_masks.append(splitted_att_masks[j])
            all_labels.append(doc_label)

        # Periodically clear GPU cache during processing
        if i % 1000 == 0:
            optimize_memory()

    return all_input_ids, all_att_masks, all_labels

def flat_accuracy(preds, labels):
    """Calculate accuracy from predictions and labels."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def metrics_calculator(preds, test_labels):
    """Calculate various metrics for model evaluation."""
    cm = confusion_matrix(test_labels, preds)
    num_classes = 2
    TP = []
    FP = []
    FN = []
    
    for i in range(num_classes):
        FN.append(sum(cm[i][j] for j in range(num_classes) if i != j))
        FP.append(sum(cm[j][i] for j in range(num_classes) if i != j))
        TP.append(cm[i][i])

    precision = [TP[i]/(TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0 for i in range(num_classes)]
    recall = [TP[i]/(TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0 for i in range(num_classes)]

    macro_precision = sum(precision)/num_classes
    macro_recall = sum(recall)/num_classes
    micro_precision = sum(TP)/(sum(TP) + sum(FP)) if (sum(TP) + sum(FP)) > 0 else 0
    micro_recall = sum(TP)/(sum(TP) + sum(FN)) if (sum(TP) + sum(FN)) > 0 else 0
    
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

def main():
    # Check GPU and print detailed information
    print("Checking GPU availability and details...")
    check_gpu()
    
    # Set device and optimize CUDA operations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
        torch.backends.cudnn.allow_tf32 = True
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_set, test_set = load_data()
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    # model_name = 'roberta-large'
    model_name = 'distilroberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Move model to GPU and enable parallel processing if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Generate training data
    print("Generating training data...")
    train_input_ids, train_att_masks, train_labels = generate_training_data(train_set, tokenizer)

    # Convert to tensors and move to GPU
    train_inputs = torch.tensor(train_input_ids).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    train_masks = torch.tensor(train_att_masks).to(device)

    # Optimize memory before creating DataLoader
    optimize_memory()

    # Create DataLoader with appropriate batch size based on GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    batch_size = 32  # Use larger batch size for GPUs with 16GB+ memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Training parameters
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=total_steps
    )

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        print(f'\n======== Epoch {epoch + 1} / {epochs} ========')
        print('Training...')
        
        model.train()
        total_loss = 0
        
        # Track time for performance monitoring
        epoch_start_time = time.time()
        gradient_accumulation_steps = 4  
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = time.time() - epoch_start_time
                print(f'  Batch {step}  of  {len(train_dataloader)}.')
                print(f'  Average loss: {total_loss/step:.4f}')
                print(f'  Elapsed time: {elapsed:.2f}s')
                print(f'  Batches per second: {step/elapsed:.2f}')
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            # Use torch.cuda.amp for mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
            
            total_loss += loss.item()
            loss = loss / gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Periodically optimize memory
            if step % 100 == 0:
                optimize_memory()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Epoch completed in: {time.time() - epoch_start_time:.2f}s")

    # Save model
    output_dir = "saved_models/Roberta_15k/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving model to {output_dir}")
    
    # Save model state
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)

    # Testing
    print("\nRunning testing...")
    model.eval()
    
    # Prepare test data
    test_input_ids, test_lengths = [], []
    for i in progressbar.progressbar(range(len(test_set['Input']))):
        sen = test_set['Input'].iloc[i]
        sen = tokenizer.tokenize(sen)
        if len(sen) > 510:
            sen = sen[len(sen)-510:]
        sen = [tokenizer.cls_token] + sen + [tokenizer.sep_token]
        encoded_sent = tokenizer.convert_tokens_to_ids(sen)
        test_input_ids.append(encoded_sent)
        test_lengths.append(len(encoded_sent))

    test_input_ids = pad_sequences(test_input_ids, maxlen=512, value=0, 
                                 dtype="long", truncating="post", padding="post")
    test_attention_masks = att_masking(test_input_ids)

    # Convert to tensors and move to GPU
    prediction_inputs = torch.tensor(test_input_ids).to(device)
    prediction_masks = torch.tensor(test_attention_masks).to(device)
    prediction_labels = torch.tensor(test_set.Label.to_numpy().astype(int)).to(device)

    # Create test DataLoader
    test_batch_size = batch_size * 2  # Can use larger batch size for inference
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, 
                                     batch_size=test_batch_size)

    # Test predictions
    predictions, true_labels = [], []
    test_start_time = time.time()
    
    with torch.no_grad():
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids, token_type_ids=None, 
                              attention_mask=b_input_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)

    print(f"\nTesting completed in: {time.time() - test_start_time:.2f}s")

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Convert logits to predicted labels
    predicted_labels = np.argmax(predictions, axis=1)

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    main()