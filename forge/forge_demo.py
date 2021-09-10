# encoding: utf-8

from utils.preprocess import calc_spec_f0, gen_timestamp
from conversion.data_loader import padding

import torch
import scipy.io.wavfile as wav2

device = 'cuda:0'

AdaVC = torch.load('./model.pkl')

def forge_to_wavefrom(bad_path, seq_len, pad_len):
    mel_spe_good, f0_norm_good = calc_spec_f0('./sbf02.wav')
    if mel_spe_good.shape[0] < seq_len:
        mel_spe_good = padding(mel_spe_good, seq_len).astype('float32')
        f0_norm_good = padding(f0_norm_good, seq_len).astype('float32')
    else:
        mel_spe_good = mel_spe_good[:seq_len, :]
        f0_norm_good = f0_norm_good[:seq_len, :]
        
    DEC_INP_good = torch.zeros(size=(pad_len, 80), dtype=torch.float32).to(device)

    #DEC_INP_F0_good = torch.zeros(size=(pad_len, 257), dtype=torch.float32).to(device)
    dec_inp_good = torch.cat([torch.from_numpy(mel_spe_good[:(seq_len - pad_len), :]), DEC_INP_good], dim=0).to(device)
    #dec_inp_f0_good = torch.cat([torch.from_numpy(f0_norm_good[:(seq_len - pad_len), :]), DEC_INP_F0_good], dim=0).to(device)
    
    mel_spe_good = torch.from_numpy(mel_spe_good).view((1, seq_len, 80))
    timestamp_good = gen_timestamp(f0_norm_good)
    timestamp_good = torch.from_numpy(timestamp_good).view(1, seq_len, 3).to(device)
    
    hidden_good, _ = AdaVC.reconstruct(mel_spe_good.to(device), timestamp_good, dec_inp_good.view((1, seq_len, 80)), timestamp_good)
                                       
    pro_hidden_good = hidden_good[:, :, 35:70]
    
    mel_spe, f0_norm = calc_spec_f0(bad_path)
    if mel_spe.shape[0] < seq_len:
        mel_spe = padding(mel_spe, seq_len).astype('float32')
        f0_norm = padding(f0_norm, seq_len).astype('float32')
    else:
        mel_spe = mel_spe[:seq_len, :]
        f0_norm = f0_norm[:seq_len, :]
        
    DEC_INP = torch.zeros(size=(pad_len, 80), dtype=torch.float32).to(device)

    #DEC_INP_F0 = torch.zeros(size=(pad_len, 257), dtype=torch.float32).to(device)
    dec_inp = torch.cat([torch.from_numpy(mel_spe[:(seq_len - pad_len), :]), DEC_INP], dim=0).to(device)
    #dec_inp_f0 = torch.cat([torch.from_numpy(f0_norm[:(seq_len - pad_len), :]), DEC_INP_F0], dim=0).to(device)
    
    mel_spe = torch.from_numpy(mel_spe).view((1, seq_len, 80))
    timestamp = gen_timestamp(f0_norm)
    timestamp = torch.from_numpy(timestamp).view(1, seq_len, 3).to(device)
    
    hidden, _ = AdaVC.reconstruct(mel_spe.to(device), timestamp, dec_inp.view((1, seq_len, 80)), timestamp)
        
    tim_hidden = hidden[:, :, :35]
    pro_hidden = hidden[:, :, 35:70]
    
    transform_direction = pro_hidden_good.mean(dim=1) - pro_hidden.mean(dim=1)
    pro_hidden_good = pro_hidden + 3.5 * transform_direction. # 3.5 is the hyperparameter $\beta$ in this paper
    cont_hidden = hidden[:, :, 70:]
    
    new_hidden = torch.cat([tim_hidden, pro_hidden_good, cont_hidden], dim=-1)
    new_expand_hidden = AdaVC.reconstruct.expand(new_hidden)
    
    d_in = AdaVC.reconstruct.dec_embedding(dec_inp.view(1, seq_len, 80), timestamp)
    d_out = AdaVC.reconstruct.decoder(d_in, new_expand_hidden)
    d_out = AdaVC.reconstruct.projection(d_out)
    return d_out

d_out = forge_to_wavefrom(bad_path, seq_len, pad_len)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
