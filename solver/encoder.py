import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model, max_len):
        
        super(PositionEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(start=0, end=max_len, dtype=torch.float).unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) 

        
    def forward(self, word_embeddings):
        
        return word_embeddings + self.pe[:word_embeddings.size(0), :]
    
class BinaryModel(nn.Module):
    def __init__(self):
        super(BinaryModel, self).__init__()
        
        self.sign_weight = nn.Linear(in_features=1, out_features=1, bias=False)
        self.exponent_weight = nn.Linear(in_features=11, out_features=1, bias=False)
        self.mantissa_weight = nn.Linear(in_features=52, out_features=1, bias=False)
        
        with torch.no_grad():
            # Weights for the sign bit
            self.sign_weight.weight.copy_(torch.tensor([[-1.0]], dtype=torch.float32))
            
            # Weights for the exponent (2^(e - 1023))
            exponent_powers = torch.tensor([2.0 ** i for i in range(10, -1, -1)], dtype=torch.float32)
            self.exponent_weight.weight.copy_(exponent_powers.unsqueeze(0))
            
            # Weights for the mantissa (2^(-1) to 2^(-52))
            mantissa_powers = torch.tensor([2.0 ** (-i) for i in range(1, 53)], dtype=torch.float32)
            self.mantissa_weight.weight.copy_(mantissa_powers.unsqueeze(0))
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        # Split input into sign, exponent, and mantissa
        sign = input[:, :, :1]  # First bit
        exponent = input[:, :, 1:12]  # Next 11 bits
        mantissa = input[:, :, 12:]  # Remaining 52 bits
        
        # Convert sign
        sign_value = self.sign_weight(sign)
        
        # Convert exponent (subtract 1023 for bias)
        exponent_value = self.exponent_weight(exponent) - 1023
        
        # Convert mantissa (add implicit leading 1)
        mantissa_value = self.mantissa_weight(mantissa) + 1.0
        
        # Compute the final value: (-1)^sign * 2^exponent * mantissa
        decimal_value = torch.pow(2.0, exponent_value) * mantissa_value
        decimal_value = torch.where(sign == 1, -decimal_value, decimal_value)  # Apply sign
        
        return decimal_value

class MathEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super(MathEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionEncoding(embed_dim, max_seq_len)

        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=6)

        self.classifier = nn.Linear(embed_dim, 1)
        self.bit_layer = nn.Linear(embed_dim, 64)
        self.numeric_layer = BinaryModel()
    
    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        positional_embeddings = self.positional_encoding(embeddings)

        context_vectors = self.encoder(positional_embeddings)

        word_number_probs = self.classifier(context_vectors)
        bit_probs = self.bit_layer(context_vectors)
        numeric_values = self.numeric_layer(bit_probs)

        return context_vectors, word_number_probs, numeric_values