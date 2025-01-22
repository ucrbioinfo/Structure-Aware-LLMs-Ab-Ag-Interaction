import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataset import ProteinMLMDataset 
from tokenizer import ProteinTokenizer 

MAX_LENGTH = 170 
MLM_MASK_RATE = 0.3 

def train_epoch(model, train_sequences, epoch_id, batch_size, learning_rate, 
                max_grad_norm=1.0, device='cuda'):
    # Move model to the device (GPU/CPU)
    model.to(device) 
    # Prepare Data for this epoch 
    prot_tok = ProteinTokenizer()
    train_dataset = ProteinMLMDataset(train_sequences, prot_tok, MAX_LENGTH, MLM_MASK_RATE)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader)  
    warmup_steps = int(0.2*total_steps) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps) 
    # Loss function for MLM (ignoring padded tokens)
    loss_fn = CrossEntropyLoss(ignore_index=-100) 
    # Set model to training mode 
    model.train()  

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id + 1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # Ground truth tokens for MLM
        optimizer.zero_grad()   
        # Forward pass
        logits = model(input_ids, attention_mask)
        # Compute loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()  # Update model parameters
        scheduler.step()  # Adjust learning rate
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    #print(f"Epoch {epoch_id + 1} finished with average loss: {avg_loss:.4f}", flush=True)
    return model, avg_loss 


def val_epoch(model, val_sequences, device='cuda', batch_size=32):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    prot_tok = ProteinTokenizer()
    val_dataset = ProteinMLMDataset(val_sequences, prot_tok, MAX_LENGTH, MLM_MASK_RATE)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = CrossEntropyLoss(ignore_index=-100)  # Assuming 0 is the pad token id

    with torch.no_grad():  # No need to track gradients during validation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # Forward pass
            logits = model(input_ids, attention_mask)
            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(logits, dim=-1) 
            # Ignore pad tokens in accuracy calculation
            mask = (labels != -100)  # Assuming 0 is the pad token id
            correct_predictions += (predicted[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item() 
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0 
    #print(f"Val loss: {avg_loss:0.4f}, Val_accuracy: {accuracy:0.4f}", flush=True)
    return avg_loss, accuracy 

 
 
def train(model, 
    train_seqs, 
    val_seqs,
    max_length, mlm_mask_rate,
    num_epochs, 
    save_path,
    batch_size=32, 
    learning_rate=5e-5, 
    max_grad_norm=1.0, 
    device='cpu'):
    
    global MAX_LENGTH 
    global MLM_MASK_RATE 
    MAX_LENGTH = max_length 
    MLM_MASK_RATE = mlm_mask_rate  
    
    max_val_acc = 0.0 
    for epoch_id in range(num_epochs):
        model, train_loss = train_epoch( model, train_seqs, epoch_id, batch_size, learning_rate, max_grad_norm, device) 
        val_loss, val_accuracy = val_epoch(model, val_seqs, device, batch_size)
        print(f"Epoch {epoch_id} finished with average loss: {train_loss:.4f}", flush=True)
        print(f"Val loss: {val_loss:0.4f}, Val_accuracy: {val_accuracy:0.4f}", flush=True) 
        if val_accuracy > max_val_acc:
            max_val_acc = val_accuracy 
            torch.save(model.state_dict(), f"{save_path}.pth")
            print(f"Model saved with accuracy {val_accuracy}") 
