import pandas as pd
import torch
import random 

from dataset import ESM2_MLM_Struct_Dataset  
from model import ESM2_MLM_Struct  
from train import train_model  

from torch.utils.data import DataLoader, random_split


if __name__=="__main__":
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    MASK_RATE = 0.3 
    BATCH_SIZE = 64 
    NUM_EPOCH = 30 
    LEARNING_RATE = 5e-5

    
    ###############  FOR HEAVY CHAIN ###############
    # max_length = 228  
    # dataset = ESM2_MLM_Struct_Dataset('../data/vh_structures/', max_length=max_length, mask_prob=MASK_RATE) 
    # train_ratio = 0.8
    # val_ratio = 1 - train_ratio
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # batch_size = BATCH_SIZE
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

    # model = ESM2_MLM_Struct(hidden_size=640, max_length=max_length+2, device=device) 
    # model = train_model(model, train_loader, val_loader, 
    #               epochs=NUM_EPOCH, learning_rate=LEARNING_RATE,  max_grad_norm=1.0, device=device,
    #               savepath='saved_model/esm_struct_heavy.pth')
    
    
    ###############  FOR LIGHT CHAIN  ############### 
    max_length = 217 
    dataset = ESM2_MLM_Struct_Dataset('../data/vl_structures/', max_length=max_length, mask_prob=MASK_RATE) 
    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

    model = ESM2_MLM_Struct(hidden_size=640, max_length=max_length+2, device=device) 
    model = train_model(model, train_loader, val_loader, 
                  epochs=NUM_EPOCH, learning_rate=LEARNING_RATE,  max_grad_norm=1.0, device=device,
                  savepath='saved_model/esm_struct_light.pth')
     

