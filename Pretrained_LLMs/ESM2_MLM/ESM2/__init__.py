from .esm2 import ESM2  
from .create_dataset import convert
from .tokenizer import ESM2_Tokenizer 
import torch 
import numpy as np

def load_model(model, saved_file, device):
    state_dict = torch.load(saved_file, map_location=device) 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict) 
    return model 

