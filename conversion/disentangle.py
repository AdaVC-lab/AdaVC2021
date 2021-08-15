# encoding: utf-8

from os import listdir
from os.path import join

import torch
from settings import CONV_PARAMS

from .conversion import Conversion

CONV_MODEL = Conversion(CONV_PARAMS)
CONV_MODEL.to(CONV_PARAMS['device'])

model_path = join(CONV_PARAMS['check_point'], 'conv_.pkl')
CONV_MODEL.load_state_dict(torch.load(model_path))

root_dir = CONV_PARAMS['root_dir']
for collector in listdir(root_dir):
    collector_path = join(root_dir, collector, 'diarized')

    for wav_name in listdir(collector_path):
        if 'xw' not in wav_name:
            pass
        else:
            pass

