import os
import numpy as np
import tensorflow as tf
import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from dataset_loader import build_tf_dataset
from model import make_or_restore_model
from datetime import datetime
import pandas as pd
import cremad
from dataset_loader import CLASSES
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('Check if TensorFlow is working with GPU: ', len(tf.config.list_physical_devices('GPU')) > 0)


def plot_confusion_matrix(y_true, y_pred, base_dir):
    confusion_mtx = confusion_matrix(y_true, y_pred)
    cmdisp = ConfusionMatrixDisplay(confusion_mtx, display_labels=CLASSES.keys())
    cmdisp.plot()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))

    return


def plot_classification_report(y_true, y_pred, base_dir):
    labels_dict = {str(v): k for k, v in CLASSES.items()}
    plt.figure()
    clf_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    clf_report_display = {labels_dict[key]: clf_report[key]
                          for key in sorted(clf_report.keys() & labels_dict.keys())}
    for key in clf_report.keys() - labels_dict.keys():
        clf_report_display[key] = clf_report[key]
    ax_clf = sns.heatmap(pd.DataFrame(clf_report_display).iloc[:-1, :].T, annot=True, cmap='crest')
    ax_clf.figure.tight_layout()
    ax_clf.figure.savefig(os.path.join(base_dir, 'classification_report.png'))

    return


def plot_history(base_dir, path):
    df = pd.read_csv(path)
    epochs = df['epoch'].tolist()
    # plot accuracy
    plt.plot(epochs, df[['categorical_accuracy', 'val_categorical_accuracy']],
             label=['train accuracy', 'validation accuracy'])
    # plt.xticks(epochs)
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.savefig(os.path.join(base_dir, 'accuracy.png'))

    # plot loss
    plt.figure()
    plt.plot(epochs, df[['loss', 'val_loss']], label=['train loss', 'validation loss'])
    # plt.xticks(epochs)
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'loss.png'))

    return


def check_resume(exp_name, exp_names):
    if exp_name in exp_names:
        print('Resume training with name {}?\n'.format(exp_name))
        flag = input('y/n\n')
        if flag in ('y', 'Y', 'yes'):
            return True

    return False


def run_experiment(dataset, exp_dir, exp_name, train_dir, val_dir, test_dir, epochs, resume=False):
    if not resume:
        current_ts = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        base_dir = os.path.join(exp_dir, current_ts + '_' + exp_name)
        os.makedirs(base_dir)
        ckpt_dir = os.path.join(base_dir, 'checkpoint')
        os.makedirs(ckpt_dir)
    else:
        exp_name_dir = [dir_name for dir_name in os.listdir(exp_dir) if dir_name.endswith(exp_name)][0]
        base_dir = os.path.join(exp_dir, exp_name_dir)
        ckpt_dir = os.path.join(base_dir, 'checkpoint')

    train_dataset = dataset.build_tf_dataset(train_dir)
    val_dataset = dataset.build_tf_dataset(val_dir)
    test_dataset = dataset.build_tf_dataset(test_dir)
    model, last_epoch = make_or_restore_model(base_dir, ckpt_dir, dataset)

    # save summary into text file
    with open(os.path.join(base_dir, 'summary.log'), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    if last_epoch > 0:
        print('Last running epoch = ', last_epoch)
        epochs = epochs + last_epoch

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        # We include the training loss in the saved model name.
        filepath=ckpt_dir,
        # filepath=ckpt_dir + "/ckpt-loss={loss:.2f}-epoch={epoch}",
        save_best_only=True,
        monitor='val_categorical_accuracy',
        mode='max'
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=30)
    csv_logger = keras.callbacks.CSVLogger(os.path.join(base_dir, 'training.csv'),
                                           append=True if resume else False)
    callbacks = [model_checkpoint, early_stopping, csv_logger]
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        initial_epoch=last_epoch,
                        use_multiprocessing=True,
                        workers=-1,
                        verbose=1,
                        callbacks=callbacks)

    predictions = model.predict(test_dataset)
    y_true = [np.argmax(y, axis=-1) for x, y in test_dataset.unbatch().as_numpy_iterator()]
    y_pred = [np.argmax(y, axis=-1) for y in predictions]
    csv_path = os.path.join(base_dir, 'training.csv')
    plot_history(base_dir, csv_path)
    plot_confusion_matrix(y_true, y_pred, base_dir)
    plot_classification_report(y_true, y_pred, base_dir)


if __name__ == '__main__':
    train_dir = 'D:\\app faccia emozioni\\augmented-cremad\\train\\'
    val_dir = 'D:\\app faccia emozioni\\augmented-cremad\\val'
    test_dir = 'D:\\app faccia emozioni\\augmented-cremad\\test'

    # root dir of all experiments
    exp_dir = 'D:\\app faccia emozioni\\cremad-experiments'

    exp_names = [filename.split('_')[-1] for filename in os.listdir(exp_dir)]
    exp_name = input('Insert exp name, available runs: {}\n'.format(exp_names))
    epochs = 1000
    resume = check_resume(exp_name, exp_names)
    run_experiment(cremad, exp_dir, exp_name, train_dir, val_dir, test_dir, epochs, resume=resume)
