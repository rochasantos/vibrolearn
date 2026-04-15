from estimators.pipeline import Pipeline
from dataset.utils import filter_registers_by_key_value_sequence, get_acquisition_data, get_values_by_key, load_matlab_acquisition, prepare_segments_and_targets
import numpy as np


class AugmentedPipeline(Pipeline):
    def train(self, list_of_registers, experimental_setup):
        self.experimental_setup = experimental_setup
        X, y = augment_acquisition(list_of_registers, self.experimental_setup)
        self.pipe.fit(X, y)
        return self


def augment_acquisition(list_of_registers, experimental_setup):
    conditions = sorted(list(get_values_by_key(list_of_registers, "condition")))
    loads = sorted(list(get_values_by_key(list_of_registers, "load")))
    X, y, = [], [],
    for load in loads:
        X, y = aggregate_condition_data(list_of_registers, conditions, load, experimental_setup)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # print(f"Total augmented dataset size: {X.shape, y.shape, set(y)}")
    return X, y


def aggregate_condition_data(list_of_registers, conditions, load, experimental_setup):
    X_condition, y_condition = [], []
    for condition in conditions:
        condition_registers = filter_registers_by_key_value_sequence(list_of_registers, [("condition", [condition]), ("load", [load])])
        if len(condition_registers) == 0:
            continue
        X_severity, y_severity = mix_severity_data(condition_registers, experimental_setup)
        X_condition.append(X_severity)
        y_condition.append(y_severity)
    return X_condition,y_condition


def mix_severity_data(condition_registers, experimental_setup):
    severity_acquisitions, severity_registers = [], []
    severity_levels = get_values_by_key(condition_registers, "severity")
    severity_levels = list(set(severity_levels))

    X_severity, y_severity = prepare_severity_segments(condition_registers, severity_acquisitions, severity_registers, severity_levels, experimental_setup)

    generate_mixed_segments(condition_registers, severity_acquisitions, severity_levels, X_severity, y_severity, experimental_setup)

    X_severity = np.concatenate(X_severity, axis=0)
    y_severity = np.concatenate(y_severity, axis=0)
    return X_severity,y_severity


def prepare_severity_segments(condition_registers, severity_acquisitions, severity_registers, severity_levels, experimental_setup):
    raw_dir_path=experimental_setup["raw_dir_path"]
    channels_columns=experimental_setup["channels_columns"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    segment_length=experimental_setup["segment_length"]
    
    X_severity, y_severity = [], []
    for severity in severity_levels:
        severity_registers.append(filter_registers_by_key_value_sequence(condition_registers, [("severity", [severity])])[0])

        acquisition = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, severity_registers[-1])
        severity_acquisitions.append(acquisition)

        X, y = prepare_segments_and_targets(segment_length=segment_length, register=severity_registers[-1], acquisition=acquisition)
        X_severity.append(X)
        y_severity.append(y)
    return X_severity,y_severity


def generate_mixed_segments(condition_registers, severity_acquisitions, severity_levels, X_severity, y_severity, experimental_setup):
    segment_length=experimental_setup["segment_length"]
    for i in range(len(severity_levels)-1):
        for j in range(i+1, len(severity_levels)):
            X_mix = mix_two_acquisitions(severity_acquisitions[i], severity_acquisitions[j])
            X, y = prepare_segments_and_targets(segment_length=segment_length, register=condition_registers[i], acquisition=X_mix)
            X_severity.append(X)
            y_severity.append(y)


def mix_two_acquisitions(acq1, acq2):
    min_length = min(acq1.shape[0], acq2.shape[0])
    acq1, acq2 = acq1[:min_length], acq2[:min_length]
    xf1, xf2 = np.fft.rfft(acq1, axis=0), np.fft.rfft(acq2, axis=0)
    xf_mix = (xf1 + xf2)
    return np.fft.irfft(xf_mix, n=max(acq1.shape[0], acq2.shape[0]), axis=0)

