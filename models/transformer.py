import torch
import torch.nn as nn


class TimeBenderTransformer(nn.Module):
    def __init__(self, config):
        super(TimeBenderTransformer, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']

        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x
