"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
"""

from data_augment.spec_augment.sparse_image_warp_tensorflow import sparse_image_warp
import tensorflow as tf
import numpy as np

DEFAULT_CONFIG = {
    "TimeWarp": {
        "W": 10
    },
    "FrequencyMask": {
        "F": 30,
        "num_masks": 1,
        "replace_with_zero": False
    },
    "TimeMask": {
        "T": 30,
        "num_masks": 1,
        "replace_with_zero": False
    }
}


def time_warp(mel_spectrogram, W=80):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
      W(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.

    # Returns
      mel_spectrogram(Tensor): warped and masked mel spectrogram. (1, row, col, 1)
    """

    fbank_size = tf.shape(mel_spectrogram)
    num_rows, spec_len = fbank_size[1], fbank_size[2]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    # TODO: Symptomatic treatment.
    #       If we do not take a margin, the behavior of sparse_image_warp will be unstable, so take an extra margin.
    time_margin = spec_len // 5 + W
    pt = tf.random.uniform([], time_margin, spec_len - time_margin, tf.int32) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(num_rows // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_freq, src_ctr_pt_time), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -W, W, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_freq, dest_ctr_pt_time), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def time_warp_old(mel_spectrogram, W=80):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
      W(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.

    # Returns
      mel_spectrogram(Tensor): warped and masked mel spectrogram. (1, row, col, 1)
    """

    num_rows = mel_spectrogram.shape[1]
    spec_len = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    pt = np.random.randint(W, spec_len-W)

    # Destination
    w = np.random.randint(-W, W)

    # warp
    y = num_rows // 2
    source_control_point_locations = tf.constant([[[y, pt]]], dtype=tf.float32)
    dest_control_point_locations = tf.constant([[[y, pt + w]]], dtype=tf.float32)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def frequency_masking(mel_spectrogram, v, F=27, num_masks=2, replace_with_zero=False):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
      F(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      num_masks(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(mel_spectrogram)
    num_rows, spec_len = fbank_size[1], fbank_size[2]
    spec_mean = tf.math.reduce_mean(mel_spectrogram)

    for i in range(num_masks):
        f = tf.random.uniform([], minval=0, maxval=F, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.cast(tf.ones(shape=(1, num_rows - f0 - f, spec_len, 1)), tf.bool),
                          tf.cast(tf.zeros(shape=(1, f, spec_len, 1)), tf.bool),
                          tf.cast(tf.ones(shape=(1, f0, spec_len, 1)), tf.bool),
                          ), 1)
        if replace_with_zero:
            padding_val = tf.cast(0, dtype=float)
        else:
            padding_val = spec_mean
        mel_spectrogram = tf.where(mask, mel_spectrogram, tf.fill([1, num_rows, spec_len, 1], padding_val))
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, T=100, num_masks=2, replace_with_zero=False):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
      T(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      num_masks(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(tensor): (1, n_fft, spec_length, 1): dimension is 4
    """
    fbank_size = tf.shape(mel_spectrogram)
    num_rows, spec_len = fbank_size[1], fbank_size[2]
    spec_mean = tf.math.reduce_mean(mel_spectrogram)

    # Step 3 : Time masking
    for i in range(num_masks):
        t = tf.random.uniform([], minval=0, maxval=T, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.cast(tf.ones(shape=(1, num_rows, spec_len - t0 - t, 1)), tf.bool),
                          tf.cast(tf.zeros(shape=(1, num_rows, t, 1)), tf.bool),
                          tf.cast(tf.ones(shape=(1, num_rows, t0, 1)), tf.bool),
                          ), 2)
        if replace_with_zero:
            padding_val = tf.cast(0, dtype=float)
        else:
            padding_val = spec_mean
        mel_spectrogram = tf.where(mask, mel_spectrogram, tf.fill([1, num_rows, spec_len, 1], padding_val))
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram, config=DEFAULT_CONFIG):

    if mel_spectrogram.ndim != 2:
        raise ValueError("input mel_spectrogram dimension must be 2.")
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    mel_spectrogram_tensor = tf.cast(mel_spectrogram.reshape(v, tau, 1), dtype=tf.float32)
    mel_spectrogram_tensor = spec_augment_from_tensor(mel_spectrogram_tensor, config=config)

    # tensor (R, W, 1) to numpy.ndarray (R, W)
    ret = mel_spectrogram_tensor.numpy().reshape(v, tau)

    return ret


# mel_spectrogram_tensor.shape: (R, W, 1)
def spec_augment_from_tensor(mel_spectrogram_tensor, config=DEFAULT_CONFIG):
    mel_spectrogram_tensor = tf.expand_dims(mel_spectrogram_tensor, 0)  # (R, W, 1) -> (1, R, W, 1)
    v = mel_spectrogram_tensor.shape[1]
    tau = mel_spectrogram_tensor.shape[2]

    if "TimeWarp" in config:
        mel_spectrogram_tensor = time_warp(mel_spectrogram_tensor,
                                           W=config["TimeWarp"]["W"])

    if "FrequencyMask" in config:
        mel_spectrogram_tensor = frequency_masking(mel_spectrogram_tensor, v=v,
                                                   F=config["FrequencyMask"]["F"],
                                                   num_masks=config["FrequencyMask"]["num_masks"],
                                                   replace_with_zero=config["FrequencyMask"]["replace_with_zero"])

    if "TimeMask" in config:
        mel_spectrogram_tensor = time_masking(mel_spectrogram_tensor, tau=tau,
                                              T=config["TimeMask"]["T"],
                                              num_masks=config["TimeMask"]["num_masks"],
                                              replace_with_zero=config["TimeMask"]["replace_with_zero"])

    # tensor (1, R, W, 1) to tensor (R, W, 1)
    ret = tf.squeeze(mel_spectrogram_tensor, 0)

    return ret
