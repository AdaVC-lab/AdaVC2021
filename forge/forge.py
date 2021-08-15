# encoding: utf-8

from torch import nn, cat

from components.custmized import ProbAttEncoder, ProbAttDecoder
from components.embed import DataEmbedding


class Forge(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        self.enc_embedding = DataEmbedding(h_params['enc_in'], h_params['enc_out_dim'])
        self.encoder = ProbAttEncoder(h_params['factor'],
                                      h_params['dropout'],
                                      h_params['n_heads'],
                                      h_params['enc_out_dim'],
                                      h_params['d_ff'],
                                      h_params['activation'],
                                      h_params['enc_layers']
                                      )

        self.dec_embedding = DataEmbedding(h_params['dec_in'], h_params['enc_out_dim'])
        self.decoder = ProbAttDecoder(h_params['factor'],
                                      h_params['dropout'],
                                      h_params['n_heads'],
                                      h_params['enc_out_dim'],
                                      h_params['d_ff'],
                                      h_params['activation'],
                                      h_params['dec_layers'])
        self.projection = nn.Linear(h_params['enc_out_dim'], h_params['dec_out_dim'], bias=True)

    def forward(self, cont_codes, enc_mask, dec_in, dec_mask, tim_codes, emo_codes=None):
        if emo_codes is not None:
            emo_codes = emo_codes.unsqueeze(1).expand(-1, cont_codes.size(1), -1)
            enc_in = cat((cont_codes, tim_codes, emo_codes), dim=-1)
        else:
            enc_in = cat((cont_codes, tim_codes), dim=-1)

        enc_in = self.enc_embedding(enc_in, enc_mask)
        enc_out, _ = self.encoder(enc_in)

        dec_in = self.dec_embedding(dec_in, dec_mask)
        dec_out = self.decoder(dec_in, enc_out)
        return self.projection(dec_out)
