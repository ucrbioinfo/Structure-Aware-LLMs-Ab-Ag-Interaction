import torch

class ProteinTokenizer:
    def __init__(self):
        # Define the vocabulary for protein sequences, including 'X' as unknown token
        self.amino_acids = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", 
            "P", "Q", "R", "S", "T", "V", "W", "Y", "X"  # 'X' represents an unknown amino acid
        ]
        # Add special tokens for [PAD], [CLS], [SEP], and [MASK]
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        
        # Create a token-to-id mapping (vocabulary)
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens + self.amino_acids)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}  # Reverse lookup table
        self.pad_token_id = self.vocab["[PAD]"]
        self.cls_token_id = self.vocab["[CLS]"]
        self.sep_token_id = self.vocab["[SEP]"]
        self.mask_token_id = self.vocab["[MASK]"]
        self.vocab_size = len(self.vocab) 
        
    def tokenize(self, sequence):
        """Tokenizes a protein sequence into amino acid tokens."""
        return list(sequence)
    
    def convert_tokens_to_ids(self, tokens):
        """Converts a list of tokens into corresponding IDs."""
        return [self.vocab.get(token, self.vocab["X"]) for token in tokens]  # Replace unknown amino acids with 'X'
    
    def convert_ids_to_tokens(self, ids):
        """Converts a list of IDs back into tokens."""
        return [self.id_to_token.get(id_, "[UNK]") for id_ in ids]  # Use "[UNK]" for unknown ids
    
    def encode(self, sequence, max_length=512):
        """
        Encodes a sequence into token IDs with special tokens and padding.
        Also returns attention mask where non-padded tokens are marked with 1.
        """
        tokens = ["[CLS]"] + self.tokenize(sequence)[:max_length - 2] + ["[SEP]"]  # Truncate if needed and add special tokens
        token_ids = self.convert_tokens_to_ids(tokens)
        
        # Pad to max_length
        attention_mask = [1] * len(token_ids)  # Mask for non-padding tokens
        padding_length = max_length - len(token_ids)
        if padding_length > 0:
            token_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        
        return torch.tensor(token_ids), torch.tensor(attention_mask)
    
    def decode(self, token_ids):
        """
        Decodes a list of token IDs back into a sequence, ignoring padding and special tokens.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        # Remove special tokens and join the sequence
        return ''.join([token for token in tokens if token not in self.special_tokens])
    
    def mask_tokens(self, inputs, mask_prob=0.15):
        """
        Randomly masks tokens for the MLM objective.
        The labels contain -100 for tokens that are not masked (ignored in loss computation).
        """
        labels = inputs.clone()
        #probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = torch.isin(labels, torch.tensor([self.cls_token_id, self.sep_token_id, self.pad_token_id]))
        
        num_tokens = labels.numel() - special_tokens_mask.sum().item()  # Total tokens excluding special tokens
        num_to_mask = int(mask_prob * num_tokens)  # 15% of tokens
        # Randomly choose which tokens to mask, ignoring special tokens
        maskable_indices = torch.nonzero(~special_tokens_mask, as_tuple=False).view(-1)
        masked_indices = maskable_indices[torch.randperm(len(maskable_indices))[:num_to_mask]]
        
        #probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        #masked_indices = torch.bernoulli(probability_matrix).bool()
        for i in range(len(labels)): 
            if i not in masked_indices: labels[i] = -100 # Only compute loss on masked tokens 
        
        inputs[masked_indices] = self.mask_token_id  # Replace selected tokens with [MASK]
        return inputs, labels
    
