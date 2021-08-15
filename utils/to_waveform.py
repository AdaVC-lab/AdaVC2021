import torch
from tqdm import tqdm
from wavenet_vocoder import builder

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def build_model():
    model = getattr(builder, 'wavenet')(
        out_channels=30,
        layers=24,
        stacks=4,
        residual_channels=512,
        gate_channels=512,
        skip_out_channels=256,
        cin_channels=80,
        gin_channels=-1,
        weight_normalization=True,
        n_speakers=-1,
        dropout=0.05,
        kernel_size=3,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 4],
        freq_axis_kernel_size=3,
        scalar_input=True,
        legacy=True,
    )
    return model


def gen_wave(model, c=None):
    """Generate waveform samples by WaveNet.

    """
    model.eval()
    model.make_generation_fast_()

    tc = c.shape[0]
    upsample_factor = 256
    # Overwrite length according to feature size
    length = tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=-32.23619130191664)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


