from dataset.utils import get_X_y, load_matlab_acquisition
from preprocessing.augmentation import get_agumented_data
import numpy as np


def vanilla(registers, experimental_setup):
    raw_dir_path=experimental_setup["raw_dir_path"]
    channels_columns=experimental_setup["channels_columns"]
    segment_length=experimental_setup["segment_length"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    X, y = get_X_y(registers, 
                   raw_dir_path=raw_dir_path, 
                   channels_columns=channels_columns, 
                   segment_length=segment_length, 
                   load_acquisition_func=load_acquisition_func)
    return X, y


def augmented(list_of_registers, experimental_setup):
        X_aug, y_aug = get_agumented_data(list_of_registers, experimental_setup, repetitions=3)
        X_ori, y_ori = vanilla(list_of_registers, experimental_setup)
        X = np.concatenate([X_ori, X_aug], axis=0)
        y = np.concatenate([y_ori, y_aug], axis=0)
        return X, y

