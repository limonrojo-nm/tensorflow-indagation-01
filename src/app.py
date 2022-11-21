# Based on https://www.tensorflow.org/tutorials/images/classification

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

from images import import_images
from dataset import train_dataset, validate_dataset
from visualization import visualize_dataset


def execute_app():
    print("Executing app.py")

    print("Import images...")
    data_dir = import_images()
    train_ds = train_dataset(data_dir)
    val_ds = validate_dataset(data_dir)
    class_names = train_ds.class_names  # type:  ignore
    print(class_names)

    visualize_dataset(train_ds, class_names)
