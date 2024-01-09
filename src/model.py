import torch.nn as nn 

# Model Definition 
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLanguageModel, self).__init__()
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
    