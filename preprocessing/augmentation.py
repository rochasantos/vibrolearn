from sklearn.pipeline import Pipeline as SklearnPipeline
import numpy as np

from dataset.utils import filter_registers_by_key_value_sequence, get_values_by_key


def mix_two_acquisitions(acq1, acq2):
    min_length = min(acq1.shape[0], acq2.shape[0])
    acq1, acq2 = acq1[:min_length], acq2[:min_length]
    xf1, xf2 = np.fft.rfft(acq1, axis=0), np.fft.rfft(acq2, axis=0)
    xf_mix = (xf1 + xf2) / 2
    return np.fft.irfft(xf_mix, n=max(acq1.shape[0], acq2.shape[0]), axis=0)


def augment_acquisition(list_of_registers, load_function):
    verbose = True
    conditions = get_values_by_key(list_of_registers, "condition")
    X_condition, y_condition = aggregate_condition_data(list_of_registers, load_function, conditions)
    X = np.concatenate(X_condition, axis=0)
    y = np.concatenate(y_condition, axis=0)
    if verbose:
        print(f"Total augmented dataset size: {X.shape, y.shape, set(y)}")
    return X, y


def aggregate_condition_data(list_of_registers, load_function, conditions):
    X_condition, y_condition = [], []
    for condition in conditions:
        condition_registers = filter_registers_by_key_value_sequence(list_of_registers, [("condition", [condition])])
        severity_levels = get_values_by_key(condition_registers, "severity")
        severity_levels = list(set(severity_levels))
        X_severity, y_severity = mix_severity_data(load_function, condition_registers, severity_levels)
        X_condition.append(X_severity)
        y_condition.append(y_severity)
    return X_condition,y_condition


def mix_severity_data(load_function, condition_registers, severity_levels):
    X_severity, y_severity = [], []
    for severity in severity_levels:
        severity_registers = filter_registers_by_key_value_sequence(condition_registers, [("severity", [severity])])
        X, y = load_function(severity_registers)
        X_severity.append(X)
        y_severity.append(y)
    for i in range(len(severity_levels)-1):
        for j in range(i+1, len(severity_levels)):
            X_mix = mix_two_acquisitions(X_severity[i], X_severity[j])
            y_mix = np.array([y[0] for _ in range(X_mix.shape[0])]) 
            X_severity.append(X_mix)
            y_severity.append(y_mix)
    X_severity = np.concatenate(X_severity, axis=0)
    y_severity = np.concatenate(y_severity, axis=0)
    return X_severity,y_severity


class AugmentedPipeline():
    def __init__(self, steps):
        self.pipe = SklearnPipeline(steps)
        self.load_function = None

    def set_load_function(self, load_function):
        self.load_function = load_function
  
    def train(self, list_of_registers):
        X, y = augment_acquisition(list_of_registers, self.load_function) if self.load_function else (np.array([[0, 0]]), None)
        self.pipe.fit(X, y)
        return self
    
    def evaluate(self, list_of_registers, list_of_metrics):
        X, y = self.load_function(list_of_registers) if self.load_function else (np.array([[0, 0]]), None)
        y_pred = self.pipe.predict(X)
        scores = {}
        for metric in list_of_metrics:
            scores[metric.__name__] = metric(y, y_pred)
        return scores
    