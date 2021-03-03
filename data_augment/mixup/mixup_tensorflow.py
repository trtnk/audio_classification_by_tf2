"""
MixUP Implementation for Tensorflow2.x
"""
import tensorflow as tf


# Thanks to https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu#MixUp-Augmentation
# input images_tensor - is a batch of images of size [batch_size, width, height, ch_num] not a single image of [width, height, ch_num]
# input labels_tensor - is a batch of labels. Size is
#   [batch_size, m]: m is "number of classes" in the case of one-hot-vector
#   [batch_size]: When using the label value as it is instead of one-hot-vector
#   [batch_size, 1]: When using the probability value belonging to A when classifying two classes
# output - a batch of images with mixup applied
def mixup_from_tensor(images_tensor, labels_tensor, BATCH_SIZE=32, ALPHA=0.2, PROBABILITY=1.0, CLASSES=2):
    imgs = []
    labs = []

    for j in range(BATCH_SIZE):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32) # P: 0 or 1

        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)
        a = tf.random.uniform([], 0, ALPHA) * P # this is beta dist with alpha=1.0

        # MAKE MIXUP IMAGE
        img1 = images_tensor[j, ]
        img2 = images_tensor[k, ]
        imgs.append((1-a) * img1 + a * img2)

        # MAKE CUTMIX LABEL
        if len(labels_tensor.shape)==1:
            lab1 = tf.one_hot(labels_tensor[j], CLASSES)
            lab2 = tf.one_hot(labels_tensor[k], CLASSES)
        else:
            lab1 = labels_tensor[j, ]
            lab2 = labels_tensor[k, ]
        labs.append((1-a)*lab1 + a*lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    ret_images_tensor = tf.reshape(tf.stack(imgs), (BATCH_SIZE, images_tensor.shape[1], images_tensor.shape[2], images_tensor.shape[3]))

    if len(labels_tensor.shape) == 1:
        ret_labels_tensor = tf.reshape(tf.stack(labs), (BATCH_SIZE, CLASSES))
    elif labels_tensor.shape[1] != 1:
        ret_labels_tensor = tf.reshape(tf.stack(labs), (BATCH_SIZE, CLASSES))
    else:
        ret_labels_tensor = tf.reshape(tf.stack(labs), (BATCH_SIZE, 1))

    return ret_images_tensor, ret_labels_tensor
