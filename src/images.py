import tensorflow as tf
import pathlib


def import_images():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        'flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Image count:", image_count)

    return data_dir
