import torch 
import numpy as np 

class ESM2_Tokenizer():
    def __init__(self):
        self.esm_tokens = ['<cls>', '<pad>', '<eos>', '<unk>', 'L',  'A', 'G', 'V', 'S', 'E', 'R', 'T', 
                            'I', 'D','P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z',
                            'O', '.', '-', '<null_1>', '<mask>']

        self.vocab = {token: idx for idx, token in enumerate(self.esm_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}  
        self.pad_token_id = self.vocab['<pad>']
        self.cls_token_id = self.vocab['<cls>'] 
        self.eos_toekn_id = self.vocab['<eos>']
        self.mask_token_id = self.vocab['<mask>'] 
        self.vocab_size = len(self.vocab) 
    
    
    def encode(self, seq, max_length=256):
        tokens = [self.vocab['<cls>']] + [self.vocab['<pad>']] * max_length + [self.vocab['<eos>']]
        if len(seq) > max_length:
            start = np.random.randint(len(seq)-max_length)
            seq = seq[start: start+max_length]
        for i, tok in enumerate(seq):
            tokens[i+1] = self.vocab[tok] 
        return torch.tensor(tokens, dtype=int)
    