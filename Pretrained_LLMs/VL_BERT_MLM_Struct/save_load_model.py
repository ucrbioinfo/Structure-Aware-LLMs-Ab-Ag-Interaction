import json 
import torch 
from .model import CovAbLight

def save_model(model, save_path, model_name):
    """Save the model's state_dict and configuration, removing 'module.' prefix if present."""
    # If the model is wrapped in DataParallel, access the underlying model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Remove 'module.' prefix if the model was trained using DataParallel
    state_dict = model.state_dict()
    if 'module.' in next(iter(state_dict)):  # Check if 'module.' is a prefix in the state_dict
        # Remove the 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Save the model's state_dict without 'module.' prefix
    torch.save(state_dict, f"{save_path}/{model_name}.pth")
    
    # Save the model configuration
    config = {
        'vocab_size': model.config.vocab_size,
        'hidden_size': model.config.hidden_size,
        'num_layers': model.config.num_hidden_layers,
        'max_length': model.config.max_position_embeddings
    }
    with open(f"{save_path}/{model_name}.json", 'w') as f:
        json.dump(config, f)


def load_vl_bert_struct_model(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # Initialize the model with the loaded configuration
    model = CovAbLight(
        vocab_size=25,
        hidden_size=768,
        num_layers=12,
        max_length=217
    ).to(device) 
    
    # Load the model's state_dict
    model.load_state_dict(torch.load("VL_BERT_MLM_Struct/CovBert_light_min_loss.pt", map_location=device), strict=False)
    model.eval()  # Set the model to evaluation mode
    return model

