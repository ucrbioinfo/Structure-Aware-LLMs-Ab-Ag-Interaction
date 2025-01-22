import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss 
from tqdm import tqdm 



def train_epoch(model, dataloader, epochs, curr_epoch, learning_rate=5e-5, max_grad_norm=1.0, device='cuda'):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader)  
    warmup_steps = int(total_steps*0.25) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps) 

    mlm_loss_fn = CrossEntropyLoss(ignore_index=-100) 
    binary_loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([2])).to(device) 
    
    model.train()   

    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {curr_epoch + 1}/{epochs}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        #attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # Ground truth tokens for MLM
        binary_labels = batch['contacts'].to(device)  # Ground truth binary label matrix

        optimizer.zero_grad()   

        # Forward pass
        logits, binary_matrix = model(input_ids) 
        # Loss calculation 
        mlm_loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))  
        binary_loss = binary_loss_fn(binary_matrix.view(-1), binary_labels.view(-1))
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


def val_epoch(model, dataloader, device='cuda'):
    model.eval()   
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_binary_loss = 0.0
    correct_predictions = 0
    total_predictions = 0 
    model.to(device) 

    # Loss functions
    mlm_loss_fn = CrossEntropyLoss(ignore_index=-100)  # For MLM loss
    binary_loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([2])).to(device)  # For binary label matrix loss

    with torch.no_grad():  
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            #attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            binary_labels = batch['contacts'].to(device)  

            # Forward pass
            logits, binary_matrix = model(input_ids)
            # Compute MLM loss
            mlm_loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_mlm_loss += mlm_loss.item() 
            binary_loss = binary_loss_fn(binary_matrix.view(-1), binary_labels.view(-1))
            total_binary_loss += binary_loss.item()
            total_loss += mlm_loss.item() + binary_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, dim=-1)
            mask = (labels != -100)  # ignore non-masked positions 
            correct_predictions += (predicted[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    mlm_loss_avg = total_mlm_loss / len(dataloader)
    binary_loss_avg = total_binary_loss / len(dataloader) 
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f"Validation Loss: {avg_loss:.4f}, MLM Loss: {mlm_loss_avg:.4f}, "
          f"Binary Loss: {binary_loss_avg:.4f}, Accuracy: {accuracy:.4f}", flush=True)
    return avg_loss, mlm_loss_avg, binary_loss_avg, accuracy 


def train_model(model, train_loader, val_loader, epochs, learning_rate, max_grad_norm, device, savepath): 
    min_val_loss = 1000.0 
    best_val_acc = 0.001 
    for i in range(epochs):
        model, train_loss = train_epoch(
            model=model,
            dataloader=train_loader,    
            epochs=epochs,
            curr_epoch = i, 
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            device=device
        )
        val_epoch_loss, mlm_loss_avg, binary_loss_avg, val_acc  = val_epoch(model, val_loader, device) 
        
        if val_epoch_loss+(1/val_acc) < min_val_loss+(1/best_val_acc):
            min_val_loss = val_epoch_loss 
            best_val_acc = val_acc 
            model_to_save = model.module if torch.cuda.device_count() > 1 else model 
            saved_model_name = "ESM_Struct_heavy.pt"   
            torch.save(model_to_save.state_dict(), savepath) 
            print(f"Model saved from epoch {i}, with accuracy {val_acc}." , flush=True) 
        
    return  