# encoding: utf-8

from os.path import join
from pickle import dump, load
from time import clock

import torch
from data_loader import get_data
from settings import FORGE_PARAMS
from torch import optim, nn

from forge import Forge

DEVICE = torch.device(FORGE_PARAMS['device'])
FORGE = Forge(FORGE_PARAMS)
FORGE.to(DEVICE)
OPTIMIZER = optim.Adam(FORGE.parameters(), lr=0.0001)
CRITERION = nn.MSELoss()
DATA_LOADER, TRAIN_STEPS = get_data(FORGE_PARAMS)

DEC_INP = torch.zeros(size=(FORGE_PARAMS['batch_size'],
                            FORGE_PARAMS['pred_len'],
                            FORGE_PARAMS['dec_out_dim']), dtype=torch.float32).to(DEVICE)

TEST_DEC_INP = torch.zeros(size=(30,
                            FORGE_PARAMS['pred_len'],
                            FORGE_PARAMS['dec_out_dim']), dtype=torch.float32).to(DEVICE)

with open('/home/hz-liuben/AdaVC-all/AdaVC_20210728_final/check_points/extracted_features_for_forge_testset.pkl', 'rb') as handle:
    extracted_features = load(handle)
            
test_timestamp = []
test_tim_codes = []
test_pro_codes = []
test_cont_codes = []
test_emo_codes = []

for audio_id in extracted_features:
    mask, tim_hidden, pro_hidden, cont_hidden, emotion = extracted_features[audio_id]
    

    test_timestamp.append(mask)
    test_tim_codes.append(tim_hidden)
    test_pro_codes.append(pro_hidden)
    test_cont_codes.append(cont_hidden)
    test_emo_codes.append(torch.from_numpy(emotion.astype('float32')).unsqueeze(0))
    
test_timestamp = torch.cat(test_timestamp, dim=0).to(DEVICE)
test_tim_codes = torch.cat(test_tim_codes, dim=0).to(DEVICE)
test_pro_codes = torch.cat(test_pro_codes, dim=0).to(DEVICE)
test_cont_codes = torch.cat(test_cont_codes, dim=0).to(DEVICE)
test_emo_codes = torch.cat(test_emo_codes, dim=0).to(DEVICE)


train_loss = []
start = clock()
for epoch in range(FORGE_PARAMS['epochs']):
    for i, (batch_mask, batch_tim, batch_pro, batch_cont, batch_emo) in enumerate(DATA_LOADER):
        OPTIMIZER.zero_grad()

        batch_mask = batch_mask.to(DEVICE)
        batch_tim = batch_tim.to(DEVICE)
        batch_pro = batch_pro.rto(DEVICE)
        batch_cont = batch_cont.to(DEVICE)
        batch_emo = batch_emo.to(DEVICE)

        dec_inp = torch.cat([batch_pro[:, :FORGE_PARAMS['label_len', :], :], DEC_INP], dim=1)

        outputs = FORGE(batch_cont, batch_mask, dec_inp, batch_mask, batch_tim,
                        batch_emo, batch_mask)

        loss = CRITERION(outputs, batch_pro)

        if (i + 1) % 20 == 0:
            test_dec_inp = torch.cat([test_pro_codes[:, :FORGE_PARAMS['label_len'], :], TEST_DEC_INP], dim=1)
            test_outputs = FORGE(test_cont_codes, test_timestamp, test_dec_inp, test_timestamp, test_tim_codes, test_emo_codes)
            test_loss = CRITERION(test_outputs, test_pro_codes)
            
            print(f"epoch: {epoch + 1} | iter: {i + 1} | loss: {round(loss.item(), 6)} | test_loss: {round(test_loss.item(), 6)}")
        loss.backward()
        OPTIMIZER.step()

    train_loss.append((loss.item(), test_loss.item()))
    if (epoch + 1) % 100 == 0:
        train_info = f'epoch-{epoch + 1}_time-{round(clock() - start, 2)}_loss-{round(loss.item(), 6)}'
        print(f'saving model: {train_info}')
        filename = f'forge_{train_info}.pkl'
        torch.save(FORGE, join(FORGE_PARAMS['check_point'], filename))

        with open(join(FORGE_PARAMS['check_point'], 'forge_loss.pkl'), 'wb') as handle:
            dump(train_loss, handle)
