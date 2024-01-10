import torch
import math
import torch.nn as nn 
import torch.nn.functional as F

# Model Definition 
class RNNExpertModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNExpertModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
        
    def init_state(self, batch_size):
        return (torch.zeros(self.lstm.num_layers,batch_size, self.lstm.hidden_size),
                torch.zeros(self.lstm.num_layers,batch_size, self.lstm.hidden_size))

# Stacked Decoder with learnable position embedding
class TransformerExpertModel(nn.Module):
    def __init__(self, vocab_size, embed_size=768, nhead=12, num_layers=3):
        super(TransformerExpertModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(512, embed_size)  # Assuming maximum sequence length of 512
        self.transformer_decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_size, nhead) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.positional_embedding(positions)
        
        for layer in self.transformer_decoders:
            x = layer(x)

        output = self.fc(x)
        return output


# Stacked Decoder with sin/cos positional embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class GPT2StackedDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=768, nhead=12, num_layers=12):
        super(GPT2StackedDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.transformer_decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_size, nhead) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)

        for layer in self.transformer_decoders:
            x = layer(x)

        output = self.fc(x)
        return output

