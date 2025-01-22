import torch
import torch.nn as nn
from Pretrained.ESM2 import ESM2, load_model, convert  

class ESM2_MLM(nn.Module):
    def __init__(self,device='cpu'):
        super(ESM2_MLM, self).__init__()
        self.device = device
        self.bert = ESM2() 


    def load_esm2_weights(self):
        self.bert = load_model(self.bert, 'ESM2/esm2_t30_150M_UR50D.pt',self.device) 


    def forward(self, input_ids, output_attentions=False, output_representation=False):
        outputs = self.bert(input_ids, repr_layers=[30], return_attention=True)  
        sequence_output = outputs['representations'][30] 
        logits = outputs['logits']
        
        # Return logits and attention matrices
        if output_attentions:
            all_attentions = outputs['attentions'] 
            return logits, sequence_output, all_attentions 
        elif output_representation:
            return logits, sequence_output  
        else:
            return logits
        
        
        
    # def get_embeddings(self, sequences, mode='residue', batch_size=32):
    #     embeddings = [] 
    #     for i in range(0, len(sequences), batch_size):
    #         batch_sequences = sequences[i:i+batch_size]
    #         ids = [] 
    #         masks = [] 
    #         for seq in batch_sequences:
    #             id = torch.tensor(convert(seq, max_length=170)) 
    #             ids.append(id)  
    #         ids = torch.stack(ids) 
    #         masks = torch.stack(masks)
    #         logits, embs = self.forward(ids, output_representation=True)
    #         if mode=='sequence':
    #             for emb in embs: embeddings.append(emb[:,0,:])
    #         else:
    #             embeddings.extend(embs)
    #     return torch.stack(embeddings) 
    
    
    # def get_logits(self, sequences, batch_size=32):
    #     logits = [] 
    #     for i in range(0, len(sequences), batch_size):
    #         batch_sequences = sequences[i:i+batch_size]
    #         ids = [] 
    #         for seq in batch_sequences:
    #             id = torch.tensor(convert(seq, max_length=170))
    #             ids.append(id)
    #         ids = torch.stack(ids) 

    #         logits = self.forward(ids)
    #     return torch.stack(logits)  