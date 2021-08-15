from torch.utils.data import DataLoader, Dataset
from pickle import load


class DisentangledFeat(Dataset):
    def __init__(self):
        super().__init__()
        self.__read__()

    def __read__(self):
        with open('/home/hz-liuben/AdaVC-all/AdaVC_20210728_final/check_points/extracted_features_for_forge.pkl', 'rb') as handle:
            extracted_features = load(handle)
        
        self.timestamp = []
        self.tim_codes = []
        self.pro_codes = []
        self.cont_codes = []
        self.emo_codes = []

        for audio_id in extracted_features:
            mask, tim_hidden, pro_hidden, cont_hidden, emotion = extracted_features[audio_id]
            

            self.timestamp.append(mask.squeeze())
            self.tim_codes.append(tim_hidden.squeeze())
            self.pro_codes.append(pro_hidden.squeeze())
            self.cont_codes.append(cont_hidden.squeeze())
            self.emo_codes.append(emotion.astype('float32'))

    def __getitem__(self, index):
        return (self.timestamp[index], self.tim_codes[index], 
                self.pro_codes[index], self.cont_codes[index], 
                self.emo_codes[index])

    def __len__(self):
        return len(self.timestamp)


def get_data(hyper_params):
    dis_feat = DisentangledFeat()
    data_loader = DataLoader(dis_feat,
                             batch_size=hyper_params['batch_size'],
                             num_workers=0,
                             drop_last=True,
                             shuffle=True)
    return data_loader, len(dis_feat)
