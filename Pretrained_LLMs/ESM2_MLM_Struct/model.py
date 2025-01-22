import torch
import torch.nn as nn
import torch.nn.functional as F  
from .ESM2 import ESM2, load_model, convert  

class ESM2_MLM_Struct(nn.Module):
    def __init__(self, hidden_size=640, max_length=512, device='cpu'):
        super(ESM2_MLM_Struct, self).__init__()
        self.device = device
        self.bert = ESM2() 
        self.binary_matrix_head = ContactPrediction(hidden_size, max_length) 


    def load_esm2_weights(self):
        self.bert = load_model(self.bert, 'ESM2/esm2_t30_150M_UR50D.pt',self.device) 


    def forward(self, input_ids, output_attentions=False, output_representation=False):
        outputs = self.bert(input_ids, repr_layers=[30], return_attention=True)  
        sequence_output = outputs['representations'][30] 
        logits = outputs['logits'] 
        last_layer_attention = torch.stack([att[29] for att in outputs['attentions'] ]) 
        
        binary_matrix = self.binary_matrix_head(sequence_output, last_layer_attention)
        
        # Return logits and attention matrices
        if output_attentions:
            all_attentions = outputs['attentions'] 
            return logits, sequence_output, all_attentions 
        elif output_representation:
            return logits, sequence_output  
        else:
            return logits, binary_matrix 
        
        
        

    
    
class ContactPrediction(nn.Module):
    def __init__(self, hidden_size, input_len):
        super(ContactPrediction, self).__init__()
        self.embedding_fc = nn.Linear(hidden_size, input_len)
        self.input_len = input_len

        # Row-wise linear transformation layer for each row i
        self.row_fc = nn.Linear(input_len, input_len)  # Acts on each row independently
        
        # 2D convolutional layers for refining 2D matrix prediction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, embeddings, attentions): 
        """
        Args:
            embeddings (torch.Tensor): Last hidden layer embeddings (batch_size, input_len, hidden_size).
            attentions (tuple): Tuple of attention matrices (num_layers, batch_size, num_heads, input_len, input_len).
        Returns:
            binary_matrix (torch.Tensor): Predicted binary matrix of shape (batch_size, input_len, input_len).
        """
        # Use the last layer's attention, averaged over heads
        attentions = attentions.mean(dim=1)  # Shape: (batch_size, input_len, input_len)

        # Adjust embeddings to match the input shape for row-wise transformation
        embedding_features = self.embedding_fc(embeddings)  # Shape: (batch_size, input_len, input_len)
        embedding_features = torch.softmax(embedding_features, dim=-1)
        # Combine embeddings with attention features
        combined_features = torch.relu(embedding_features + attentions) / 2  # Shape: (batch_size, input_len, input_len)

        # Apply row-wise linear transformation to each row
        row_transformed = self.row_fc(combined_features)  # Shape: (batch_size, input_len, input_len)

        # Reshape for convolutional layers
        row_transformed = row_transformed.unsqueeze(1)  # Shape: (batch_size, 1, input_len, input_len)

        # Apply convolutional layers to refine 2D matrix prediction
        x = F.relu(self.conv1(row_transformed))
        x = F.relu(self.conv2(x))
        binary_logits = self.conv3(x)  # Shape: (batch_size, 1, input_len, input_len)
        
        return binary_logits.squeeze(1)  # Remove channel dimension for final output 
    