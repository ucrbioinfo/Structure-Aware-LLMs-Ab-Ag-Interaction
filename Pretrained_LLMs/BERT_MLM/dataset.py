from torch.utils.data import Dataset

# Dataset class
class ProteinMLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=512, mask_prob=0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids, attention_mask = self.tokenizer.encode(sequence, max_length=self.max_length)
        masked_input_ids, labels = self.tokenizer.mask_tokens(input_ids, mask_prob=self.mask_prob)
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels  # Labels for MLM task (-100 for non-masked tokens)
        }

