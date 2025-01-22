from torch.utils.data import Dataset
import os 
import pdb_utilities as myPDB 
import numpy as np

# Dataset class
class StructureDataset(Dataset):
    def __init__(self, pdb_paths_dir, tokenizer, max_length=200, mask_prob=0.15, contact_threshold=4.0):
        """
        Dataset class for protein sequences with MLM (Masked Language Model) task.
        Args:
            sequences (list): List of protein sequences.
            tokenizer (ProteinTokenizer): The tokenizer to convert sequences to token IDs.
            max_length (int): Maximum length for sequences (including [CLS] and [SEP]).
            mask_prob (float): Probability of masking tokens for MLM.
        """
        self.pdb_paths = [pdb_paths_dir+f for f in os.listdir(pdb_paths_dir) if f.endswith('.pdb')] 
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.contact_threshold = contact_threshold 

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.pdb_paths)

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        # Tokenize and encode the sequence with attention mask 
        dist, seq = myPDB.get_residue_distance_matrix(pdb_path, 'A') 
        contacts = np.where(dist < self.contact_threshold, 1, 0) 
        new_matrix = np.zeros((self.max_length, self.max_length))
        l = contacts.shape[0]  
        new_matrix[1:l+1, 1:l+1] = contacts  
        input_ids, attention_mask = self.tokenizer.encode(seq, max_length=self.max_length)
        # Apply masking to input_ids for MLM
        masked_input_ids, labels = self.tokenizer.mask_tokens(input_ids, mask_prob=self.mask_prob)
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'contacts': new_matrix, 
            'labels': labels  # Labels for MLM task (-100 for non-masked tokens)
        }

