from torch.utils.data import Dataset
import torch 
import numpy as np 


# Dataset class
class ESM2_MLM_Dataset(Dataset):
    def __init__(self, sequences, max_length=512, mask_prob=0.15):
        """
        Dataset class for protein sequences with MLM (Masked Language Model) task.
        Args:
            sequences (list): List of protein sequences.
            max_length (int): Maximum length for sequences (including [CLS] and [SEP]).
            mask_prob (float): Probability of masking tokens for MLM.
        """
        self.sequences = sequences
        self.max_length = max_length
        self.mask_prob = mask_prob
        
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
        

    def __len__(self):
        return len(self.sequences)
    
    
    def convert(self, seq, max_length=256):
        tokens = [self.vocab['<cls>']] + [self.vocab['<pad>']] * max_length + [self.vocab['<eos>']]
        if len(seq) > max_length:
            start = np.random.randint(len(seq)-max_length)
            seq = seq[start: start+max_length]
        for i, tok in enumerate(seq):
            tokens[i+1] = self.vocab[tok] 
        return torch.tensor(tokens, dtype=int)
    
    
    def mask_tokens(self, inputs, mask_prob=0.15):
        """
        Randomly masks tokens for the MLM objective.
        The labels contain -100 for tokens that are not masked (ignored in loss computation).
        """
        labels = inputs.clone()
        #probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = torch.isin(labels, torch.tensor([self.cls_token_id, self.eos_toekn_id, self.pad_token_id]))
        
        num_tokens = labels.numel() - special_tokens_mask.sum().item()  # Total tokens excluding special tokens
        num_to_mask = int(mask_prob * num_tokens)  # 15% of tokens
        maskable_indices = torch.nonzero(~special_tokens_mask, as_tuple=False).view(-1)
        masked_indices = maskable_indices[torch.randperm(len(maskable_indices))[:num_to_mask]]
        
        #probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        #masked_indices = torch.bernoulli(probability_matrix).bool()
        for i in range(len(labels)): 
            if i not in masked_indices: labels[i] = -100  # Only compute loss on masked tokens
        inputs[masked_indices] = self.mask_token_id  # Replace selected tokens with <mask> 
        return inputs, labels


    def __getitem__(self, idx, mlm_masking=True):
        sequence = self.sequences[idx]
        input_ids = self.convert(sequence, self.max_length) 
        if mlm_masking:
            masked_input_ids, labels = self.mask_tokens(input_ids, mask_prob=self.mask_prob)
            return {
                'input_ids': masked_input_ids,
                'labels': labels  # Labels for MLM task (-100 for non-masked tokens)
            }
        else:
            return {
                'input_ids': input_ids 
            }

