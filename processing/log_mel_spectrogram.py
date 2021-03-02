"""
Thanks to https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
"""
import tensorflow as tf


class LogMelSpectrogram:
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min=0.0, f_max=None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max
        )

    def __call__(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.n_fft,
                                      frame_step=self.hop_length,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = self.power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def power_to_db(self, magnitude, amin=1e-16, top_db=80.0):
        """
        https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
        """
        ref_value = tf.reduce_max(magnitude)
        log_spec = 10.0 * self._tf_log10(tf.maximum(amin, magnitude))
        log_spec -= 10.0 * self._tf_log10(tf.maximum(amin, ref_value))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    @staticmethod
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

