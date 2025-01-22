import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import BertModel, BertConfig 


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
        """Initialize weights of the model using Xavier initialization."""
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
        attentions = attentions[-1].mean(dim=1)  # Shape: (batch_size, input_len, input_len)

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
    

class CovAbHeavy(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, max_length=512):
        """ 
        BERT model for the Masked Language Modeling (MLM) task, with an additional
        contact prediction head to predict an l x l binary matrix.

        Args:
            vocab_size (int): Size of the vocabulary (number of protein tokens).
            hidden_size (int): Size of hidden representations (embedding size).
            num_layers (int): Number of BERT layers to use.
            max_length (int): Maximum sequence length for the input.
        """
        super(CovAbHeavy, self).__init__()
        
        # Define BERT model configuration
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            type_vocab_size=1,  # We don't have token type embeddings for sequences
            pad_token_id=0
        )

        # Embedding layers: token embeddings and position embeddings
        self.bert = BertModel(self.config)

        # MLM prediction head (linear layer)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        # Contact prediction module
        self.binary_matrix_head = ContactPrediction(hidden_size, max_length)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights of the model using Xavier initialization."""
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Use Xavier uniform initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero


    def forward(self, input_ids, attention_mask, output_attentions=False, 
                output_representation=False):
        """
        Args:
            input_ids (torch.Tensor): Tensor of token IDs (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask to ignore padding tokens.
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                            output_attentions=True) 
        
        # Sequence outputs from the last hidden layer
        sequence_output = outputs.last_hidden_state 
        
        # MLM predictions (logits for each token in the vocabulary)
        logits = self.mlm_head(sequence_output)
        
        
        binary_matrix = self.binary_matrix_head(sequence_output, outputs.attentions)
        
        
        # Return logits and attention matrices
        if output_attentions:
            all_attentions = outputs.attentions
            return logits, sequence_output, torch.stack(all_attentions).permute(1, 0, 2, 3, 4)  
        elif output_representation:
            return logits, sequence_output  
        else:
            return logits, binary_matrix   
        
