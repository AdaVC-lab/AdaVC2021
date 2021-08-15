# encoding: utf-8

from os.path import join
from pickle import dump
from time import clock

import torch
from settings import CONV_PARAMS
from torch import optim, nn
from .conversion import Conversion
from .data_loader import get_data

DEVICE = torch.device(CONV_PARAMS['device'])


CONV = Conversion(CONV_PARAMS)
CONV.to(DEVICE)

OPTIMIZER = optim.Adam(filter(lambda x: x.requires_grad, CONV.parameters()), lr=0.0001)
CRITERION = nn.MSELoss()
SEP_CRITERION = nn.CrossEntropyLoss()

DATA_LOADER, TRAIN_STEPS = get_data(CONV_PARAMS)
PSEUDO_LABEL = torch.range(0, 2, dtype=torch.long).repeat_interleave(CONV_PARAMS['batch_size']).to(DEVICE)
DEC_INP = torch.zeros(size=(CONV_PARAMS['batch_size'],
                            CONV_PARAMS['pred_len'],
                            CONV_PARAMS['mel_dim']), dtype=torch.float32)

DEC_INP_F0 = torch.zeros(size=(CONV_PARAMS['batch_size'],
                            CONV_PARAMS['pred_len'],
                            CONV_PARAMS['f0_dim']), dtype=torch.float32)

train_loss = []
start = clock()


for epoch in range(CONV_PARAMS['epochs']):
    if epoch + 1 == 201:
        for param in CONV.discriminator.parameters():
            param.requires_grad = False
        for param in CONV.timbre.parameters():
            param.requires_grad = False
        for param in CONV.prosody.parameters():
            param.requires_grad = False
        
        OPTIMIZER = optim.Adam(filter(lambda x: x.requires_grad, CONV.parameters()), lr=0.0001)
    
    for i, (batch_mel, batch_f0, batch_timestamp, batch_timbre) in enumerate(DATA_LOADER):
        OPTIMIZER.zero_grad()

        batch_timestamp = batch_timestamp.to(DEVICE)
        batch_timbre = batch_timbre.to(DEVICE).squeeze()
        batch_timbre = batch_timbre.repeat_interleave(CONV_PARAMS['max_pad_len'], dim=1)

        dec_inp = torch.cat([batch_mel[:, :CONV_PARAMS['label_len'], :], DEC_INP], dim=1).to(DEVICE)
        dec_inp_f0 = torch.cat([batch_f0[:, :CONV_PARAMS['label_len'], :], DEC_INP_F0], dim=1).to(DEVICE)

        batch_f0 = batch_f0.to(DEVICE)
        batch_mel = batch_mel.to(DEVICE)
        tim_out, pro_out, cont_out, mel_out, class_prob = CONV(batch_mel, batch_timestamp, dec_inp, batch_timestamp, 
                                                              batch_f0, batch_timestamp, dec_inp_f0, batch_timestamp)

        tim_loss = CRITERION(tim_out, batch_timbre)
        pro_loss = CRITERION(pro_out, batch_f0)
        
        mel_loss = CRITERION(mel_out, batch_mel)

        # discriminate prosody, content and timbre
        separate_loss = SEP_CRITERION(class_prob, PSEUDO_LABEL)

        loss = tim_loss + pro_loss + mel_loss + separate_loss

        if (i + 1) % 20 == 0:
            print(f"epoch: {epoch + 1} | iter: {i + 1} | loss: {round(loss.item(), 6)} | "
                  f"tim_loss: {round(tim_loss.item(), 6)} | pro_loss: {round(pro_loss.item(), 6)} | "
                  f"mel_loss: {round(mel_loss.item(), 6)} | "
                  f"separate_loss: {round(separate_loss.item(), 6)}")
        loss.backward()
        OPTIMIZER.step()

    train_loss.append(loss.item())
    if (epoch + 1) % 100 == 0:
        train_info = f'pro-35_tim-35_cont-90_epoch-{epoch + 1}_time-{round(clock() - start, 2)}_loss-{round(loss.item(), 6)}'
        print(f'saving model: {train_info}')
        filename = f'conversion_{train_info}.pkl'
        torch.save(CONV, join(CONV_PARAMS['check_point'], filename))
        with open(join(CONV_PARAMS['check_point'], 'reconstruct_loss.pkl'), 'wb') as handle:
            dump(train_loss, handle)
