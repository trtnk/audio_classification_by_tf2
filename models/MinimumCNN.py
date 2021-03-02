import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models
from tensorflow.keras import layers


# モデルの定義(tensorflowのチュートリアルの物をそのまま使った場合)
def MinimumCNN(spectrogram_ds):
    for spec in spectrogram_ds.take(1):
        input_shape = spec.shape
    print('Input shape:', input_shape)
    num_labels = 2

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model
