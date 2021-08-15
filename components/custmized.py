from torch import nn

from .attn import AttentionLayer, ProbAttention, FullAttention
from .decoder import Decoder, DecoderLayer
from .encoder import Encoder, EncoderLayer
# from .encoder import ConvLayer


class ProbAttEncoder(nn.Module):
    def __init__(self, factor, dropout, n_heads, model_dim, d_ff, activation, layers):
        super().__init__()
        prob_att = ProbAttention(False, factor, attention_dropout=dropout)
        prob_att = AttentionLayer(prob_att, model_dim, n_heads, mix=False)
        prob_attentions = [EncoderLayer(prob_att, model_dim, d_ff, dropout=dropout, activation=activation)
                           for _ in range(layers)]
        # distills = [ConvLayer(model_dim) for _ in range(layers - 1)]
        distills = None
        self.prob_att = Encoder(prob_attentions, distills, norm_layer=nn.LayerNorm(model_dim))

    def forward(self, x):
        return self.prob_att(x)


class ProbAttDecoder(nn.Module):
    def __init__(self, factor, dropout, n_heads, model_dim, d_ff, activation, layers):
        super().__init__()
        prob_att = ProbAttention(True, factor, attention_dropout=dropout,
                                 output_attention=False)
        prob_att = AttentionLayer(prob_att, model_dim, n_heads, mix=True)
        full_att = FullAttention(False, factor, attention_dropout=dropout,
                                 output_attention=False)
        full_att = AttentionLayer(full_att, model_dim, n_heads, mix=False)

        self.decoder = Decoder([DecoderLayer(prob_att, full_att, model_dim, d_ff,
                                             dropout=dropout, activation=activation)
                                for _ in range(layers)],
                               norm_layer=nn.LayerNorm(model_dim)
                               )

    def forward(self, decoder_in, encoder_out):
        return self.decoder(decoder_in, encoder_out, x_mask=None, cross_mask=None)
