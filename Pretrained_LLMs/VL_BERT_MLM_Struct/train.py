import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss 
from tqdm import tqdm

def train_model(
    model, 
    dataloader,
    epochs,
    curr_epoch, 
    learning_rate=5e-5, 
    max_grad_norm=1.0, 
    device='cuda'
):
    """
    Train the ProteinBERT model with the Masked Language Modeling (MLM) objective.

    Args:
        model (torch.nn.Module): The ProteinBERTForMLM model.
        dataset (torch.utils.data.Dataset): The dataset class for protein sequences.
        tokenizer (object): The custom tokenizer that tokenizes protein sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for AdamW optimizer.
        warmup_steps (int): Number of steps for warmup in the learning rate scheduler.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        model (torch.nn.Module): The trained model.
    """
    # Move model to the device (GPU/CPU)
    model.to(device)


    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader)  
    warmup_steps = int(total_steps*0.25)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps)

    # Loss function for MLM (ignoring padded tokens)
    mlm_loss_fn = CrossEntropyLoss(ignore_index=-100) 
    binary_loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([2])).to(device)  # For binary label matrix prediction 
    
    model.train()  # Set model to training mode

    #for epoch in range(epochs):
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {curr_epoch + 1}/{epochs}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # Ground truth tokens for MLM
        binary_labels = batch['contacts'].to(device)  # Ground truth binary label matrix

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        logits, binary_matrix = model(input_ids, attention_mask) 
        # Compute MLM loss
        #print(logits.view(-1, logits.size(-1)), labels.view(-1), flush=True) 
        mlm_loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) 
        # Compute binary label matrix loss
        #print(binary_matrix.view(-1), binary_labels.view(-1), flush=True)  
        binary_loss = binary_loss_fn(binary_matrix.view(-1), binary_labels.view(-1))
        
        # Combine losses with a weighted sum
        loss = mlm_loss + binary_loss 
        total_loss += loss.item() 

        # Backward pass and optimization
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()  # Update model parameters
        scheduler.step()  # Adjust learning rate

        progress_bar.set_postfix(loss=loss.item(), mlm_loss=mlm_loss.item(), 
                                    binary_loss=binary_loss.item())
        

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {curr_epoch + 1} finished with average loss: {avg_loss:.4f}", flush=True)
    return model, avg_loss 


def validate_model(model, dataloader, device='cuda'):
    """
    Validate the ProteinBERT model on the validation dataset.

    Args:
        model (torch.nn.Module): The ProteinBERTForMLM model.
        dataset (torch.utils.data.Dataset): Dataset for validation.
        device (str): Device to validate on ('cuda' or 'cpu').
        batch_size (int): Size of each validation batch.

    Returns:
        avg_loss (float): Average loss on the validation dataset.
        mlm_loss_avg (float): Average MLM loss on the validation dataset.
        binary_loss_avg (float): Average binary label matrix loss.
        accuracy (float): Accuracy of the model on the validation dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_binary_loss = 0.0
    correct_predictions = 0
    total_predictions = 0 
    model.to(device) 

    # Loss functions
    mlm_loss_fn = CrossEntropyLoss(ignore_index=-100)  # For MLM loss
    binary_loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([2])).to(device)  # For binary label matrix loss

    with torch.no_grad():  # No need to track gradients during validation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            binary_labels = batch['contacts'].to(device)  

            # Forward pass
            logits, binary_matrix = model(input_ids, attention_mask)

            # Compute MLM loss
            mlm_loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_mlm_loss += mlm_loss.item() 
            
            # Compute binary matrix loss
            binary_loss = binary_loss_fn(binary_matrix.view(-1), binary_labels.view(-1))
            total_binary_loss += binary_loss.item()
            
            # Combined loss for this batch
            total_loss += mlm_loss.item() + binary_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, dim=-1)
            # Ignore pad tokens in accuracy calculation
            mask = (labels != -100)  # Assuming 0 is the pad token id
            correct_predictions += (predicted[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    mlm_loss_avg = total_mlm_loss / len(dataloader)
    binary_loss_avg = total_binary_loss / len(dataloader) 
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f"Validation Loss: {avg_loss:.4f}, MLM Loss: {mlm_loss_avg:.4f}, "
          f"Binary Loss: {binary_loss_avg:.4f}, Accuracy: {accuracy:.4f}", flush=True)
    return avg_loss, mlm_loss_avg, binary_loss_avg, accuracy 


def train(model, train_loader, val_loader, epochs, learning_rate, max_grad_norm, device): 
    min_val_loss = 1000.0 
    best_val_acc = 0.0 
    for i in range(epochs):
        model, train_loss = train_model(
            model=model,
            dataloader=train_loader,   # Dataset class instance
            epochs=epochs,
            curr_epoch = i, 
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            device=device
        )
        avg_loss, mlm_loss_avg, binary_loss_avg, accuracy  = validate_model(model, val_loader, device) 
        
        if avg_loss < min_val_loss:
            min_val_loss = avg_loss 
            model_to_save = model.module if torch.cuda.device_count() > 1 else model 
            saved_model_name = "CovBert_light_min_loss.pt"   
            torch.save(model_to_save.state_dict(), saved_model_name) 
            print(f"Model saved from epoch {i}, with accuracy {accuracy}." , flush=True) 
        
    return  model 