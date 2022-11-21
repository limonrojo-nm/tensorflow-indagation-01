from pathlib import Path
import tensorflow as tf
batch_size = 32
img_height = 180
img_width = 180


def train_dataset(data_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                           validation_split=0.2,
                                                           subset="training",
                                                           seed=123,
                                                           image_size=(
                                                               img_height, img_width),
                                                           batch_size=batch_size)
    return train_ds


def validate_dataset(data_dir: Path):
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                         validation_split=0.2,
                                                         subset="validation",
                                                         seed=123,
                                                         image_size=(
                                                             img_height, img_width),
                                                         batch_size=batch_size)
    return val_ds
