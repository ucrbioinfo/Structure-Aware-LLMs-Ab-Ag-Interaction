import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 

# Create dataset from sequences 
esm_alphabet = ['<cls>', '<pad>', '<eos>', '<unk>', 'L',  'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

token2idx_dict = dict(zip(esm_alphabet, list(range(len(esm_alphabet)))))

def token2idx(token):
    if token in token2idx_dict:
        token = token2idx_dict[token]
    else:
        token = token2idx_dict['<unk>']
    return token

def idx2token(idx):
    for token, token_idx in token2idx_dict.items():
        if idx == token_idx:
            return token
    
def convert(seq, length=256):
    tokens = [token2idx('<cls>')] + [1] * length + [token2idx('<eos>')]
    if len(seq) > length:
        start = np.random.randint(len(seq)-length)
        seq = seq[start: start+length]
    for i, tok in enumerate(seq):
        tokens[i+1] = token2idx(tok)
    return np.array(tokens, dtype=int)


# class SeqDataset(torch.utils.data.Dataset):
#     def __init__(self, ab_seq, pred_affinity) -> None:
#         super().__init__()
#         self.ab_seq = ab_seq
#         self.pred_affinity = pred_affinity 
        
#     def __getitem__(self, idx):
#         selected_ab = self.ab_seq[idx]
#         affinity_val = self.pred_affinity[idx]
#         return convert(selected_ab), float(affinity_val) 
    
#     def __len__(self):
#         return len(self.ab_seq) 


