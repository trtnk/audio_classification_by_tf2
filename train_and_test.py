from util.create_cvset import get_cv_df_from_yaml
from util.checkpoint_tools import CheckpointTools
from util import common
from processing.log_mel_spectrogram import LogMelSpectrogram
from models.MinimumCNN import *
import argparse

import pandas as pd
import numpy as np
import scipy as sp
import os
import glob

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# データ拡張用
from data_augment.spec_augment.spec_augment_tensorflow import spec_augment, spec_augment_from_tensor
from data_augment.mixup.mixup_tensorflow import mixup_from_tensor

# GPUのメモリに制限をかけないとエラーを吐くことがあるため、以下でメモリ使用量を制限
# 参考: [How to limit GPU Memory in TensorFlow 2.0 (and 1.x) | by Jun-young, Cha | Medium](https://starriet.medium.com/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument("-yp", "--yaml_path", help="config yaml file path")
parser.add_argument("-op", "--output_dir", help="output csv file path")
parser.add_argument("-sr", "--sample_rate", default=2000, help="sampling rate [Hz]")
parser.add_argument("-nf", "--n_fft", default=256, help="sampling rate [Hz]")
parser.add_argument("-hl", "--hop_length", default=64, help="hop length")
parser.add_argument("-nm", "--n_mels", default=128, help="n_mels")
parser.add_argument("-ws", "--window_sec", default=4, help="Window sec [sec]")
parser.add_argument("-ss", "--stride_sec", default=2, help="Window stride sec [sec]")
parser.add_argument("-en", "--extract_num", default=4, help="Number of signals to extract from one wav file.")
parser.add_argument("-sa", "--spec_augment", action="store_true")
parser.add_argument("-ma", "--mixup", action="store_true")
parser.add_argument("-bs", "--batch_size", default=32, help="Batch size")
parser.add_argument("-e", "--epoch", default=30, help="Epoch num")
args = parser.parse_args()

if not os.path.exists(args.yaml_path):
    raise Exception(f"yaml file is not exist. {args.yaml_path}")
yaml_path = args.yaml_path

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
output_dir = args.output_dir

batch_size = args.batch_size
EPOCHS = args.epoch

sample_rate = args.sample_rate
n_fft = args.n_fft
hop_length = args.hop_length
n_mels = args.n_mels
log_mel_spectrogram = LogMelSpectrogram(sample_rate, n_fft, hop_length, n_mels)

window_sec = args.window_sec
stride_sec = args.stride_sec
extract_num = args.extract_num

spec_augment_flag = args.spec_augment
mixup_flag = args.mixup

# tf settings
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Spec augment setting
config_TimeWarp = {
    "TimeWarp": {
        "W": 5 #移動量の最大値
    }
}
config_FreqMask = {
    "FrequencyMask": {
        "F": 20, # マスクの幅の最大値
        "num_masks": 1, # マスクをかける数
        "replace_with_zero": True # ゼロ埋めするか否か
    }
}
config_TimeMask = {
    "TimeMask": {
        "T": 20, # マスクの幅の最大値
        "num_masks": 1, # マスクをかける数
        "replace_with_zero": True # ゼロ埋めするか否か
    }
}


# for SpecAugment
def augment_time_warp(melspec_tensor, label_tensor):
    return spec_augment_from_tensor(melspec_tensor, config=config_TimeWarp), label_tensor


def augment_freq_mask(melspec_tensor, label_tensor):
    return spec_augment_from_tensor(melspec_tensor, config=config_FreqMask), label_tensor


def augment_time_mask(melspec_tensor, label_tensor):
    return spec_augment_from_tensor(melspec_tensor, config=config_TimeMask), label_tensor


alpha = 0.3 # SpecAugmentの各処理を走らせる確率
augment1 = lambda x, y: tf.cond(tf.random.uniform([], 0, 1) < alpha, lambda: augment_time_warp(x, y), lambda: (x, y))
augment2 = lambda x, y: tf.cond(tf.random.uniform([], 0, 1) < alpha, lambda: augment_freq_mask(x, y), lambda: (x, y))
augment3 = lambda x, y: tf.cond(tf.random.uniform([], 0, 1) < alpha, lambda: augment_time_mask(x, y), lambda: (x, y))


# for MixUp
mixup_config = {
    "PROBABILITY": 0.5, # MixUpをかける確率
    "ALPHA": 0.5,
    "CLASSES": 2
}


def augment_mixup(melspec_tensors, label_tensors):
    return mixup_from_tensor(melspec_tensors,
                             label_tensors,
                             BATCH_SIZE=batch_size*extract_num,
                             ALPHA=mixup_config["ALPHA"],
                             PROBABILITY=mixup_config["PROBABILITY"],
                             CLASSES=mixup_config["CLASSES"])


def broadcasting_arr(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]


def broadcasting_tf(tensor, block_size=3, stride=2):
    return tf.squeeze(tf.compat.v1.extract_image_patches(tensor[None,...,None, None], ksizes=[1, block_size, 1, 1], strides=[1, stride, 1, 1], rates=[1, 1, 1, 1], padding='VALID'))


# audio preprocessing for tf2
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


def divide_waveform(waveform, sample_rate=sample_rate, window_sec=window_sec, stride_sec=stride_sec):
    ret = broadcasting_tf(waveform, sample_rate*window_sec, sample_rate*stride_sec)
    if tf.rank(ret) == 1:
        return tf.expand_dims(ret, axis=0)
    else:
        return ret


def get_logmelspectrogram_and_label_random(file_path, label, sample_rate, extract_sec, extract_num):
    waveform = get_waveform(file_path)
    divide_waveforms = wave_random_crop(waveform, sample_rate, extract_sec, extract_num)
    logmelspecs = log_mel_spectrogram(divide_waveforms)
    labels = tf.fill([tf.shape(logmelspecs)[0], 1], label)
    return logmelspecs, labels


def get_logmelspectrogram_and_label(file_path, label, sample_rate=sample_rate, window_sec=window_sec, stride_sec=stride_sec):
    waveform = get_waveform(file_path)
    divide_waveforms = divide_waveform(waveform, sample_rate=sample_rate, window_sec=window_sec, stride_sec=stride_sec)
    logmelspecs = log_mel_spectrogram(divide_waveforms)
    labels = tf.fill([tf.shape(logmelspecs)[0], 1], label)
    return logmelspecs, labels


def get_logmelspectrogram(waveform):
    if tf.rank(waveform) == 1:
        waveform = tf.expand_dims(waveform, axis=0)
        return tf.squeeze(log_mel_spectrogram(waveform), axis=0)
    return log_mel_spectrogram(waveform)


def wave_random_crop(waveform, sample_rate, extract_sec, extract_num):
    for i in range(extract_num):
        ret_tensors = tf.concat([ [tf.image.random_crop(waveform, [sample_rate*extract_sec])] for i in range(extract_num) ], axis=0)
    return tf.squeeze(ret_tensors)


def batch_flatten(melspecs, labels):
    shape = tf.shape(melspecs)
    return tf.reshape(melspecs, [shape[0]*shape[1], shape[2], shape[3]]), tf.reshape(labels, [shape[0]*shape[1], 1])


def for_flat_map(melspecs, labels):
    return tf.data.Dataset.zip(tf.data.Dataset.from_tensor_slices(melspecs), tf.data.Dataset.from_tensor_slices(labels))


def create_dataset(file_paths, labels, sample_rate, extract_sec, extract_num):
    output_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(file_paths), tf.data.Dataset.from_tensor_slices(labels)))
    output_ds = output_ds.map(lambda x, y: get_logmelspectrogram_and_label(x, y, sample_rate, extract_sec, extract_num), num_parallel_calls=AUTOTUNE)
    return output_ds


cv_df = get_cv_df_from_yaml(yaml_path, output_csv_path=f"{output_dir}/cvset.csv")
cv_df.loc[:, "_main_label_"] = cv_df.loc[:, "_main_label_"].astype("float32")
n_splits = len(dict.fromkeys(cv_df["cvset"].values)) - 1

result_df = pd.DataFrame({})

for cvset in range(1, n_splits+1):
    # train, val, testに分割
    info_df_dict = {}

    # fold != cvsetをトレーニング
    cvset_train_list = list(range(1, n_splits+1))
    cvset_train_list.remove(cvset)
    cvset_train_list = [f"cv{i}" for i in cvset_train_list]
    info_df_dict["train"] = cv_df[cv_df["cvset"].isin(cvset_train_list)]

    # fold == cvsetをバリデーションデータ
    info_df_dict["val"] = cv_df[cv_df["cvset"] == f"cv{cvset}"]

    # testをテストデータ
    info_df_dict["test"] = cv_df[cv_df["cvset"] == f"test"]

    # Dataset
    base_ds_dict = {}
    for phase in ["train", "val", "test"]:
        base_ds_dict[phase] = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(info_df_dict[phase]["filepath"]),
                                                   tf.data.Dataset.from_tensor_slices(info_df_dict[phase]["_main_label_"])))

    # 正規化用のds作成
    spectrogram_ds = tf.data.Dataset.from_tensor_slices(info_df_dict["train"]["filepath"].values) \
                       .map(get_waveform, num_parallel_calls=AUTOTUNE) \
                       .map(divide_waveform, num_parallel_calls=AUTOTUNE) \
                       .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) \
                       .map(get_logmelspectrogram, num_parallel_calls=AUTOTUNE)

    # 訓練用ds
    train_ds = base_ds_dict["train"]
    train_ds = train_ds.shuffle(buffer_size=len(base_ds_dict["train"]))
    train_ds = train_ds.map(lambda x, y: get_logmelspectrogram_and_label_random(x, y, sample_rate, window_sec, extract_num), num_parallel_calls=AUTOTUNE)
    if spec_augment_flag:
        train_ds = train_ds.flat_map(lambda x, y: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))))
        #train_ds = train_ds.map(augment1, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(augment2, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(augment3, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(batch_size * extract_num)
        if mixup_flag:
            train_ds = train_ds.map(augment_mixup, num_parallel_calls=AUTOTUNE)
        #train_ds = train_ds.cache().prefetch(AUTOTUNE)
        train_ds = train_ds.prefetch(AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.map(batch_flatten, num_parallel_calls=AUTOTUNE)
        if mixup_flag:
            train_ds = train_ds.map(augment_mixup, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.cache().prefetch(AUTOTUNE)

    # validation用ds
    val_ds = base_ds_dict["val"] \
             .map(get_logmelspectrogram_and_label) \
             .cache() \
             .prefetch(AUTOTUNE)

    # test用ds
    test_ds = base_ds_dict["test"] \
              .map(get_logmelspectrogram_and_label) \
              .cache() \
              .prefetch(AUTOTUNE)

    # モデル
    model = MinimumCNN(spectrogram_ds)

    checkpoint_path = f"{output_dir}/model/cv{cvset}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    cb_funcs = []
    # Checkpoint作成設定
    check_point = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    cb_funcs.append(check_point)

    # 上で設定したCheckpointToolsをCallbackに組み込む
    cb_cptools = CheckpointTools(checkpoint_path, save_best_only=True, num_saves=3)
    cb_funcs.append(cb_cptools)

    # 学習
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_funcs
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_model_path = glob.glob(f"{checkpoint_path}/epoch{str(best_epoch).zfill(3)}*")[0]
    # 最も良かったepochを読み込み
    model = models.load_model(best_model_path)

    # test
    test_labels = []
    test_sumup_labels = []
    y_pred = []
    y_scores = []
    y_pred_majority = []
    y_scores_majority = []
    y_pred_mean = []
    y_scores_mean = []
    y_pred_max = []
    y_scores_max = []

    for audio, label in test_ds:
        tmp_pred = tf.nn.softmax(model.predict(audio)).numpy()

        y_pred.extend(list(np.argmax(tmp_pred, axis=1)))
        y_scores.extend(list(tmp_pred[:, 1]))

        # 多数決
        y_pred_majority.append(sp.stats.mode(np.argmax(tmp_pred, axis=1))[0][0])
        y_scores_majority.append(tmp_pred[:, 1].mean())

        # 平均
        y_pred_mean.append(np.argmax(tmp_pred.mean(axis=0)))
        y_scores_mean.append(tmp_pred[:, 1].mean())

        # 最大(異常確率が最大の物を抽出)
        y_pred_max.append(np.argmax(tmp_pred[np.argmax(tmp_pred, axis=0)[1]]))
        y_scores_max.append(tmp_pred[:, 1].max())

        test_labels.extend(list(label.numpy().flatten()))
        test_sumup_labels.append(label.numpy().flatten()[0])

    test_labels = np.array(test_labels)
    test_sumup_labels = np.array(test_sumup_labels)

    y_pred = np.array(y_pred)
    y_pred_majority = np.array(y_pred_majority)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_max = np.array(y_pred_max)

    y_scores = np.array(y_scores)
    y_scores_majority = np.array(y_scores_majority)
    y_scores_mean = np.array(y_scores_mean)
    y_scores_max = np.array(y_scores_max)

    additional_condition = {
        "cvset": cvset,
        "spec_augment": spec_augment_flag,
        "mixup": mixup_flag,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
    }
    # 全ての結果
    y_true = test_labels
    additional_condition["sample_sumup"] = "all"
    result = common.evaluate(y_true, y_pred, y_scores, additional_condition=additional_condition)
    result_df = result_df.append(result, ignore_index=True)
    # 多数決
    y_true = test_sumup_labels
    additional_condition["sample_sumup"] = "majority"
    result = common.evaluate(y_true, y_pred_majority, y_scores_majority, additional_condition=additional_condition)
    result_df = result_df.append(result, ignore_index=True)
    # 平均
    additional_condition["sample_sumup"] = "mean"
    result = common.evaluate(y_true, y_pred_mean, y_scores_mean, additional_condition=additional_condition)
    result_df = result_df.append(result, ignore_index=True)
    # 最大
    additional_condition["sample_sumup"] = "max"
    result = common.evaluate(y_true, y_pred_max, y_scores_max, additional_condition=additional_condition)
    result_df = result_df.append(result, ignore_index=True)

result_df.to_csv(f"{output_dir}/result.csv")
print("fin")
