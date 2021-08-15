# encoding: utf-8

DEVICE = 'cuda:0'
CHECK_POINT = '/home/hz-liuben/AdaVC-all/AdaVC_20210725/check_points/ada_vc_model'


CONV_PARAMS = dict(
    device=DEVICE,
    epochs=1500,
    batch_size=32,

    max_seq_len=400,
    max_pad_len=480,
    label_len=20,  # label_len + pred_len = max_pad_len
    pred_len=460,
    
    factor=5,  # prob_sparse attn factor
    dropout=0.05,
    n_heads=4,  # num of heads
    activation='gelu',
    d_ff=512,  # dimension of fcn

    mel_hidden_dim=256,
    mel_dim=80,
    f0_dim=257,
    compressed_dim=160,
    common_dim=32,

    tim_hidden_dim=35,
    tim_layers=2,
    tim_dim=23,

    pro_hidden_dim=35,
    pro_layers=3,
    pro_dim=257,
    
    cont_hidden_dim=90,
    cont_layers=2,
    cont_dim=16,

    e_layers=2,
    d_layers=2,
    root_dir='/home/hz-liuben/0610/grouped_wav/',
    num_workers=15,
    check_point=CHECK_POINT,
)

FORGE_PARAMS = dict(
    device=DEVICE,
    epochs=10000,
    batch_size=32,

    enc_in=90+8+35,  # tim_enc_dim + cont_enc_dim + emotion of borrower
    enc_out_dim=128,
    enc_layers=2,

    dec_in=35,
    dec_out_dim=35,  # pit_enc_dim
    dec_layers=3,

    factor=5,  # prob_sparse attn factor
    dropout=0.05,
    n_heads=4,  # num of heads
    activation='gelu',
    d_ff=128,  # dimension of fcn

    pred_len=460,
    label_len=20,
    check_point=CHECK_POINT,
)