from tensorflow import keras
from lstm_model import get_lstm_model
from cnn_model import get_conv_model
from resnet_model import get_resnet_model
from dtgm_model import get_dtgm_model
from cnn_lstm import get_conv_lstm_model
import pandas as pd
import os


def make_or_restore_model(base_dir, ckpt_dir, dataset):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    # Select the model you want to train
    mod = 3
    if mod == 1:
        model = get_lstm_model(dataset)
    elif mod == 2:
        model = get_resnet_model(dataset)
    elif mod == 3:
        model = get_dtgm_model(dataset)
    elif mod == 4:
        model = get_conv_lstm_model(dataset)
    elif mod == 5:
        model = get_conv_model(dataset)

    last_epoch = 0
    if not os.listdir(ckpt_dir):
        print("Creating a new model")
    else:
        print("Restoring from", ckpt_dir)
        last_epoch = last_trained_epoch(base_dir)
        model = keras.models.load_model(ckpt_dir)
    return model, last_epoch


def last_trained_epoch(exp_dir):
    csv_path = os.path.join(exp_dir, 'training.csv')
    df = pd.read_csv(csv_path)

    return df.epoch.max() + 1
