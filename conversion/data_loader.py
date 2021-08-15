# encoding: utf-8
from torch.utils.data import DataLoader, Dataset
from utils.preprocess import calc_spec_f0, gen_timestamp
from os import listdir
from os.path import join, isdir
# from sklearn.preprocessing import OneHotEncoder
import numpy as np
from multiprocessing import Queue, Process, Manager
from queue import Empty
import pickle

with open('/home/hz-liuben/AdaVC-all/AdaVC_20210707/check_points/coll_emb.pkl', 'rb') as handle:
    COLLECTORS_EMB = pickle.load(handle)


def producer(root_dir):
    queue = Queue(-1)

    for collector in listdir(root_dir):
        if not isdir(join(root_dir, collector)): continue
        
        collector_path = join(root_dir, collector, 'diarized')

        for wav_name in listdir(collector_path):
            if 'xw' not in wav_name:
                continue
            wav_path = join(collector_path, wav_name)
            queue.put_nowait((collector, wav_path))
    return queue


def padding(seq, max_pad_len):
    nrow = seq.shape[0]
    if nrow >= max_pad_len:
        return seq[:max_pad_len, :]
    return np.pad(seq, ((0, max_pad_len - nrow), (0, 0)))


def preprocess_wav(queue, max_pad_len, result):
    while not queue.empty():
        try:
            collector, wav_path = queue.get_nowait()
        except Empty:
            break
        
        mel, f0 = calc_spec_f0(wav_path)
        timestamp = gen_timestamp(mel)

        mel = padding(mel, max_pad_len).astype('float32')
        
        f0 = padding(f0, max_pad_len).astype('float32')
        timestamp = padding(timestamp, max_pad_len).astype('float32')
        timestamp = timestamp
        
        collector_onehot = COLLECTORS_EMB[collector].reshape(1, -1)
        result.append((mel.astype('float32'), f0.astype('float32'), timestamp.astype('float32'), collector_onehot.astype('float32')))


def consumer(num_workers=6, root_dir='/', max_pad_len=300):
    queue = producer(root_dir)
    mgr = Manager()
    result = mgr.list()

    pools = []
    for _ in range(num_workers):
        process = Process(target=preprocess_wav, args=(queue, max_pad_len, result))
        process.start()
        print(f'starting subprocess-{process.ident}...')
        pools.append(process)
    
    for p in pools:
        p.join()

    print('finishing all subprocesses!')

    mel_set = []
    f0_set = []
    timestamp_set = []
    collector_set = []
    for mel, f0, timestamp, collector in result:
        mel_set.append(mel)
        f0_set.append(f0)
        timestamp_set.append(timestamp)
        collector_set.append(collector)

    return mel_set, f0_set, timestamp_set, collector_set


class WavData(Dataset):
    def __init__(self, hyper_params):
        self.root_dir = hyper_params['root_dir']
        self.max_pad_len = hyper_params['max_pad_len']
        self.num_workers = hyper_params['num_workers']
        self.__read_data__()

    def __read_data__(self):
        self.mel_set, self.f0_set, self.timestamp_set, self.collector_set = consumer(self.num_workers, self.root_dir,
                                                                                     self.max_pad_len)

    def __getitem__(self, index):
        return self.mel_set[index], self.f0_set[index], self.timestamp_set[index], self.collector_set[index]

    def __len__(self):
        return len(self.mel_set)


def get_data(hyper_params):
    wav_data = WavData(hyper_params)
    data_loader = DataLoader(wav_data,
                             batch_size=hyper_params['batch_size'],
                             num_workers=0,
                             drop_last=True, shuffle=True)
    nrow = len(wav_data)
    return data_loader, nrow
