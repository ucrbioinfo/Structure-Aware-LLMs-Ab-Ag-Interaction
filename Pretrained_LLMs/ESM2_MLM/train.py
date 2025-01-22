import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataset import ESM2_MLM_Dataset 

MAX_LENGTH = 170 
MLM_MASK_RATE = 0.3

def train_model(model, train_sequences, epoch_id, batch_size, learning_rate, max_grad_norm, device):
    # Move model to the device (GPU/CPU)
    model.to(device)
    # creating new dataset, which will create different masked inputs than previous run 
    train_dataset = ESM2_MLM_Dataset(train_sequences, MAX_LENGTH, MLM_MASK_RATE)  
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) 
    warmup_steps = int(0.2 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss function for MLM (ignoring padded tokens)
    loss_fn = CrossEntropyLoss(ignore_index=-100) 

    model.train()  # Set model to training mode

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)  # Ground truth tokens for MLM

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        logits = model(input_ids)
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
    #print(f"Epoch {epoch_id} finished with average loss: {avg_loss:.4f}", flush=True)
    return model, avg_loss 


def validate_model(model, valid_sequences, device, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    # creating new dataset, which will create different masked inputs than previous run 
    valid_dataset = ESM2_MLM_Dataset(valid_sequences, MAX_LENGTH, MLM_MASK_RATE)
    dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = CrossEntropyLoss(ignore_index=-100)  # Assuming 0 is the pad token id

    with torch.no_grad():  # No need to track gradients during validation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, dim=-1)
            # Ignore unmasked tokens in accuracy calculation
            mask = (labels != -100)  # Assuming -100 is the unmasked token id 
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
        model, train_loss = train_model( model, train_seqs, epoch_id, batch_size, learning_rate, max_grad_norm, device) 
        val_loss, val_accuracy = validate_model(model, val_seqs, device, batch_size)
        print(f"Epoch {epoch_id} finished with average loss: {train_loss:.4f}", flush=True)
        print(f"Val loss: {val_loss:0.4f}, Val_accuracy: {val_accuracy:0.4f}", flush=True) 
        if val_accuracy > max_val_acc:
            max_val_acc = val_accuracy 
            torch.save(model.state_dict(), f"{save_path}.pth")
            print(f"Model saved with accuracy {val_accuracy}") 
    
    
    