from components.custmized import ProbAttEncoder, ProbAttDecoder
from components.embed import DataEmbedding
from torch import nn


class Reconstruct(nn.Module):
    def __init__(self, h_params):
        super().__init__()

        mel_dim = h_params['mel_dim']
        mel_hidden_dim = h_params['mel_hidden_dim']

        factor = h_params['factor']
        dropout = h_params['dropout']
        n_heads = h_params['n_heads']
        d_ff = h_params['d_ff']
        activation = h_params['activation']
        e_layers = h_params['e_layers']
        d_layers = h_params['d_layers']

        compressed_dim = h_params['compressed_dim']

        self.enc_embedding = DataEmbedding(mel_dim, mel_hidden_dim)
        self.encoder = ProbAttEncoder(factor, dropout, n_heads, mel_hidden_dim, d_ff, activation, e_layers)
        self.compress = nn.Sequential(nn.Linear(mel_hidden_dim, compressed_dim, bias=True), nn.ReLU())

        self.expand = nn.Sequential(nn.Linear(compressed_dim, mel_hidden_dim, bias=True), nn.ReLU())
        self.dec_embedding = DataEmbedding(mel_dim, mel_hidden_dim)
        self.decoder = ProbAttDecoder(factor, dropout, n_heads, mel_hidden_dim, d_ff, activation, d_layers)
        self.projection = nn.Linear(mel_hidden_dim, mel_dim, bias=True)

    def forward(self, e_in, e_mark, d_in, d_mark):
        e_in = self.enc_embedding(e_in, e_mark)
        e_out, _ = self.encoder(e_in)
        hidden = self.compress(e_out)

        expanded = self.expand(hidden)
        d_in = self.dec_embedding(d_in, d_mark)
        d_out = self.decoder(d_in, expanded)
        d_out = self.projection(d_out)
        return hidden, d_out