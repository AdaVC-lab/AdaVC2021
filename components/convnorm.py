# encoding: utf-8
import torch
from torch import nn
from torch.nn.functional import relu, pad


class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNormLayer, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def gen_conv_norm(input_dim, hidden_dim, layers):
    convolutions = []
    for i in range(layers):
        conv_layer = nn.Sequential(
            ConvNormLayer(input_dim if i is 0 else hidden_dim,
                          hidden_dim, kernel_size=5, stride=1,
                          padding=2, dilation=1, w_init_gain='relu'),
            nn.GroupNorm(hidden_dim // 16, hidden_dim))
        convolutions.append(conv_layer)
    convolutions = nn.ModuleList(convolutions)
    return convolutions


class ConvNorm4Rhy(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(ConvNorm4Rhy, self).__init__()
        self.convolutions = gen_conv_norm(input_dim, hidden_dim, layers)

    def forward(self, rhy):
        rhy = rhy.permute(0, 2, 1)
        for conv in self.convolutions:
            rhy = relu(conv(rhy))
        return rhy.transpose(1, 2)


class RandomResample(nn.Module):
    def __init__(self, max_pad_len, max_seq_len):
        super(RandomResample, self).__init__()
        self.max_pad_len = max_pad_len
        self.min_seg_len = 19
        self.max_seg_len = 32
        self.max_num_seg = max_seq_len // self.min_seg_len + 1

    def pad_sequences(self, sequences):
        out_dims = (len(sequences), self.max_pad_len, sequences[0].size(-1))
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_pad_len]
        return out_tensor

    def forward(self, x, seq_len):
        device = x.device
        batch_size = x.size(0)

        # indices of each sub segment
        indices = torch.arange(self.max_seg_len * 2, device=device) \
            .unsqueeze(0).expand(batch_size * self.max_num_seg, -1)
        # scales of each sub segment
        scales = torch.rand(batch_size * self.max_num_seg,
                            device=device) + 0.5

        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl

        len_seg = torch.randint(low=self.min_seg_len,
                                high=self.max_seg_len,
                                size=(batch_size * self.max_num_seg, 1),
                                device=device)

        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = pad(offset[:, :-1], (1, 0), value=0).view(-1, 1)

        idx_scaled_org = idx_scaled_fl + offset

        len_seq_rp = torch.repeat_interleave(seq_len, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)

        idx_mask_final = idx_mask & idx_mask_org

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)

        index_1 = torch.repeat_interleave(torch.arange(batch_size,
                                                       device=device), counts)

        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1

        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = torch.split(y, counts.tolist(), dim=0)
        seq_padded = self.pad_sequences(sequences)
        return seq_padded


class ConvNorm4ContPit(nn.Module):
    def __init__(self, cont_dim, cont_hidden_dim, pit_dim, pit_hidden_dim, layers, max_pad_len, max_seq_len):
        super(ConvNorm4ContPit, self).__init__()
        self.cont_conv_layers = gen_conv_norm(cont_dim, cont_hidden_dim, layers)
        self.pit_conv_layers = gen_conv_norm(pit_dim, pit_hidden_dim, layers)
        self.cont_hidden_dim = cont_hidden_dim

        self.rand_res = RandomResample(max_pad_len, max_seq_len)
        self.register_buffer('max_pad_len', torch.tensor(max_pad_len))

    def forward(self, mel, f0):
        mel = mel.transpose(2, 1)
        f0 = f0.transpose(2, 1)

        for cont_conv, pit_conv in zip(self.cont_conv_layers, self.pit_conv_layers):
            # group norm
            mel_hidden = relu(cont_conv(mel))
            f0_hidden = relu(pit_conv(f0))

            # random resampling
            mel_f0 = torch.cat((mel_hidden, f0_hidden), dim=1).transpose(1, 2)
            mel_f0 = self.rand_res(mel_f0, self.max_pad_len.expand(mel_hidden.size(0)))

            mel = mel_f0[:, :, :self.cont_hidden_dim]
            mel = mel.transpose(2, 1)
            f0 = mel_f0[:, :, self.cont_hidden_dim:]
            f0 = f0.transpose(2, 1)

        mel = mel.transpose(1, 2)
        f0 = f0.transpose(1, 2)
        return mel, f0
