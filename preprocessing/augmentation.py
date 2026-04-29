from dataset.utils import filter_registers_by_key_value_sequence, get_acquisition_data, get_values_by_key, load_matlab_acquisition, prepare_segments_and_targets
import numpy as np


def get_agumented_data(list_of_registers, experimental_setup, repetitions=1):
    X, y = [], []
    for _ in range(repetitions):
        X_aug, y_aug = augment_acquisition(list_of_registers, experimental_setup)
        X.append(X_aug)
        y.append(y_aug)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def augment_acquisition(list_of_registers, experimental_setup):
    conditions = get_values_by_key(list_of_registers, "condition")
    X, y, = [], []
    for condition in conditions:
        X_agregated, y_agregated = aggregate_load_acquistions(list_of_registers, condition, experimental_setup)
        if X_agregated is None or y_agregated is None:
            continue
        X.append(X_agregated)
        y.append(y_agregated)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def aggregate_load_acquistions(list_of_registers, condition, experimental_setup):
    X, y = [], []
    loads = (list(get_values_by_key(list_of_registers, "load")))
    for load in loads:
        condition_registers = filter_registers_by_key_value_sequence(list_of_registers, [("condition", [condition]), ("load", [load])])
        if len(condition_registers) <= 1:
            continue
        X_mixed, y_mixed = mix_severity_data(condition_registers, experimental_setup)
        X.append(X_mixed)
        y.append(y_mixed)
    if len(X) == 0 or len(y) == 0:
        return None, None
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


def mix_severity_data(condition_registers, experimental_setup):
    segment_length=experimental_setup["segment_length"]
    channels_columns=experimental_setup["channels_columns"]
    acquisitions = []
    for key_value in channels_columns.keys():
        key, value = key_value.split(":")
        value = eval(value)
        filtered_registers = filter_registers_by_key_value_sequence(condition_registers, [[key, value]])
        actual_channel_columns = channels_columns[key_value]
        for condition_register in filtered_registers:
            acquisition = load_original_acquisitions(condition_register, actual_channel_columns, experimental_setup)
            acquisitions.append(acquisition)
    X, y = mix_acquisitions(condition_registers, segment_length, acquisitions)
    return X, y
    

def mix_acquisitions(condition_registers, segment_length, acquisitions):
    X, y = [], []
    for i in range(len(acquisitions)-1):
        for j in range(i+1, len(acquisitions)):
            mixed_acquisition = mix_two_acquisitions(acquisitions[i], acquisitions[j])
            X_mix, y_mix = prepare_segments_and_targets(segment_length=segment_length, register=condition_registers[i], acquisition=mixed_acquisition)
            X.append(X_mix)
            y.append(y_mix)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


def load_original_acquisitions(condition_register, actual_channel_columns, experimental_setup):
    raw_dir_path=experimental_setup["raw_dir_path"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    acquisition = get_acquisition_data(raw_dir_path, actual_channel_columns, load_acquisition_func, condition_register)
    return acquisition


def mix_two_acquisitions(acq1, acq2):
    min_length = min(acq1.shape[0], acq2.shape[0])
    acq1, acq2 = acq1[:min_length], acq2[:min_length]
    xf1, xf2 = np.fft.rfft(acq1, axis=0), np.fft.rfft(acq2, axis=0)
    xf_mix = (xf1 + xf2)
    return np.fft.irfft(xf_mix, n=max(acq1.shape[0], acq2.shape[0]), axis=0)

