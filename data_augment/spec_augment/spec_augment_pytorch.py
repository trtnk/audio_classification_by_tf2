# Thanks to https://github.com/zcaceres/spec_augment
import numpy as np
import torch
import random
from data_augment.spec_augment.sparse_image_warp_pytorch import sparse_image_warp

DEFAULT_CONFIG = {
    "TimeWarp": {
        "time_warping_para": 10
    },
    "FrequencyMask": {
        "frequency_masking_para": 20,
        "frequency_mask_num": 1,
        "replace_with_zero": False
    },
    "TimeMask": {
        "time_masking_para": 40,
        "time_mask_num": 1,
        "replace_with_zero": False
    }
}

def time_warp(spec, W=10):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows//2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    #point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    point_to_warp = np.random.randint(W, spec_len-W)
    #assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def frequency_masking(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def time_masking(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned


def spec_augment(mel_spectrogram, config=DEFAULT_CONFIG):

    if mel_spectrogram.ndim != 2:
        raise ValueError("input mel_spectrogram dimension must be 2.")
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram.reshape(1, v, tau)).clone()
    mel_spectrogram_tensor = spec_augment_from_tensor(mel_spectrogram_tensor, config=config)

    # tensor (1, R, W) to numpy.ndarray (R, W)
    ret = mel_spectrogram_tensor.squeeze().to("cpu").detach().numpy().copy()

    return ret


# mel_spectrogram_tensor.shape: (1, R, W)
def spec_augment_from_tensor(mel_spectrogram_tensor, config=DEFAULT_CONFIG):
    if "TimeWarp" in config:
        mel_spectrogram_tensor = time_warp(mel_spectrogram_tensor,
                                           W=config["TimeWarp"]["time_warping_para"])

    if "FrequencyMask" in config:
        mel_spectrogram_tensor = frequency_masking(mel_spectrogram_tensor,
                                                   F=config["FrequencyMask"]["frequency_masking_para"],
                                                   num_masks=config["FrequencyMask"]["frequency_mask_num"],
                                                   replace_with_zero=config["FrequencyMask"]["replace_with_zero"])

    if "TimeMask" in config:
        mel_spectrogram_tensor = time_masking(mel_spectrogram_tensor,
                                              T=config["TimeMask"]["time_masking_para"],
                                              num_masks=config["TimeMask"]["time_mask_num"],
                                              replace_with_zero=config["TimeMask"]["replace_with_zero"])
    return mel_spectrogram_tensor
