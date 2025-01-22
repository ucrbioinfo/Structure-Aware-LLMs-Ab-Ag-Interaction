import torch
import torch.nn as nn
from transformers import BertModel, BertConfig 

class ProteinBERTForMLM(nn.Module):
    def __init__(self, vocab_size=25, hidden_size=768, num_layers=12, max_length=170):
        super(ProteinBERTForMLM, self).__init__()
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            type_vocab_size=1,  # We don't have token type embeddings for protein sequences
            pad_token_id=0
        )
        self.bert = BertModel(self.config)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_ids, attention_mask, output_attentions=False, output_representation=False):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        sequence_output = outputs.last_hidden_state
        logits = self.mlm_head(sequence_output)
        
        # Return logits and attention matrices
        if output_attentions:
            all_attentions = outputs.attentions
            return logits, sequence_output, all_attentions 
        elif output_representation:
            return logits, sequence_output  
        else:
            return logits
        
        
        
        
    # def get_embeddings(self, sequences, mode='residue', batch_size=32):
    #     tok = ProteinTokenizer()
    #     embeddings = [] 
    #     for i in range(0, len(sequences), batch_size):
    #         batch_sequences = sequences[i:i+batch_size]
    #         ids = [] 
    #         masks = [] 
    #         for seq in batch_sequences:
    #             id, mask = tok.encode(seq, max_length=170)
    #             ids.append(id)
    #             masks.append(mask)
    #         ids = torch.stack(ids) 
    #         masks = torch.stack(masks)
    #         logits, embs = self.forward(ids, masks, output_representation=True)
    #         if mode=='sequence':
    #             for emb in embs: embeddings.append(emb.mean(dim=0))
    #         else:
    #             embeddings.extend(embs)
    #     return torch.stack(embeddings) 
    
    # def get_logits(self, sequences, batch_size=32):
    #     tok = ProteinTokenizer()
    #     logits = [] 
    #     for i in range(0, len(sequences), batch_size):
    #         batch_sequences = sequences[i:i+batch_size]
    #         ids = [] 
    #         masks = [] 
    #         for seq in batch_sequences:
    #             id, mask = tok.encode(seq, max_length=170)
    #             ids.append(id)
    #             masks.append(mask)
    #         ids = torch.stack(ids) 
    #         masks = torch.stack(masks)
    #         logits = self.forward(ids, masks)
    #     return torch.stack(logits)  