# encoding: utf-8
import torch
from torch import nn
from .reconstruct import Reconstruct
from .constraints import TimbreConstraint, ProsodyConstraint, ContentConstraint
from components.custmized import ProbAttEncoder


class Conversion(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        self.reconstruct = Reconstruct(h_params)
        
        self.timbre_con = TimbreConstraint(h_params)
        self.prosody_con = ProsodyConstraint(h_params)
        self.content_con = ContentConstraint(h_params)

        self.tim_hidden_dim = h_params['tim_hidden_dim']
        self.pro_hidden_dim = h_params['pro_hidden_dim']
        self.cont_hidden_dim = h_params['cont_hidden_dim']

        self.tim2common = nn.Linear(self.tim_hidden_dim, h_params['common_dim'])
        self.pro2common = nn.Linear(self.pro_hidden_dim, h_params['common_dim'])
        self.cont2common = nn.Linear(h_params['compressed_dim']-self.tim_hidden_dim-self.pro_hidden_dim,
                                     h_params['common_dim'])

        self.discriminator = Discriminator(h_params)

    def forward(self, e_in, e_mark, d_in, d_mark, f0_e_in, f0_e_mark, f0_d_in, f0_d_mark):
        hidden, mel_out = self.reconstruct(e_in, e_mark, d_in, d_mark)
        tim_hidden = hidden[:, :, :self.tim_hidden_dim]
        pro_hidden = hidden[:, :, self.tim_hidden_dim: (self.tim_hidden_dim+self.pro_hidden_dim)]
        cont_hidden = hidden[:, :, -self.cont_hidden_dim:]

        t_common = self.tim2common(tim_hidden)
        p_common = self.pro2common(pro_hidden)
        c_common = self.cont2common(cont_hidden)
        common = torch.cat((t_common, p_common, c_common), dim=0)
        class_prob = self.discriminator(common)

        return self.timbre_con(tim_hidden), self.prosody_con(pro_hidden), self.content_con(cont_hidden), mel_out, class_prob


class Discriminator(nn.Module):
    def __init__(self, h_params):
        super().__init__()

        factor = h_params['factor']
        dropout = h_params['dropout']
        n_heads = 6
        d_ff = 32
        activation = h_params['activation']
        common_dim = h_params['common_dim']

        self.encoder = ProbAttEncoder(factor, dropout, n_heads, common_dim, d_ff, activation, 2)
        self.output = nn.Sequential(nn.Linear(common_dim, 3, bias=True), nn.Softmax(dim=-1))

    def forward(self, x):
        hidden = self.encoder(x)[0].mean(dim=1)
        return self.output(hidden)
