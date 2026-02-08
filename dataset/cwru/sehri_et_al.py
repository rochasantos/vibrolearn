<<<<<<< HEAD
from utils.assesment import cross_validation, holdout
from dataset.utils import filter_registers_by_key_value_sequence, get_list_of_X_y, load_matlab_acquisition, read_registers_from_config

segment_length = 1024
=======
from dataset.utils import filter_registers_by_key_value_sequence, get_list_of_X_y, load_matlab_acquisition, merge_X_y_from_lists,   read_registers_from_config
>>>>>>> 0b48f41d6b0eea3352b7411cdac0b62d995b3371


sehri_et_al_combination_rounds = [
    [(0, 0.007), (1, 0.014), (2, 0.021)],
    [(3, 0.007), (0, 0.014), (1, 0.021)],
    [(2, 0.007), (3, 0.014), (0, 0.021)],
    [(1, 0.007), (2, 0.014), (3, 0.021)],
    [(0, 0.014), (1, 0.007), (2, 0.021)],
    [(3, 0.014), (0, 0.007), (1, 0.021)],
    [(2, 0.014), (3, 0.007), (0, 0.021)],
    [(1, 0.014), (2, 0.007), (3, 0.021)],
]


def get_fold(normal_load, fault_bearing_severity, faulty_bearing, sample_rate):
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    faulty = filter_registers_by_key_value_sequence(
        registers, 
        [('sample_rate', [sample_rate]), 
         ('faulty_bearing', faulty_bearing), 
         ('severity', [f"{fault_bearing_severity:.3f}"]),
         ('prlz', ['None', '6'])])
    fold = []
    fold.extend(faulty)
    return fold


<<<<<<< HEAD
def run_papers_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_papers_splits()
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = holdout(model, list_of_X_y, test_fold_index=1, list_of_metrics=list_of_metrics) 
    return scores

papers_inspired_cross_validation_folds = ['0.007', '0.014', '0.021']

def get_papers_inspired_cross_validation_fold(severity):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ['None', '6']
    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [('sample_rate', [sample_rate]),
         ('severity', [severity]),
         ('prlz', prlzs)]
    )
    return filtered

def get_list_of_papers_inspired_cross_validation_folds():
    folds = []
    for severity in papers_inspired_cross_validation_folds:
        fold = get_papers_inspired_cross_validation_fold(severity)
        folds.append(fold)
    return folds

def run_papers_inspired_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_papers_inspired_cross_validation_folds()
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = cross_validation(model, list_of_X_y, list_of_metrics=list_of_metrics)
    return scores

proposed_cross_validation_combinations = [
    [
        [('Inner Race', '0.007'),('Outer Race', '0.007'),('Ball', '0.007')],
        [('Inner Race', '0.014'),('Outer Race', '0.014'),('Ball', '0.014')],
        [('Inner Race', '0.021'),('Outer Race', '0.021'),('Ball', '0.021')]
    ],
    [
        [('Inner Race', '0.007'),('Outer Race', '0.014'),('Ball', '0.021')],
        [('Inner Race', '0.014'),('Outer Race', '0.021'),('Ball', '0.007')],
        [('Inner Race', '0.021'),('Outer Race', '0.007'),('Ball', '0.014')]
    ],
    [
        [('Inner Race', '0.007'),('Outer Race', '0.021'),('Ball', '0.014')],
        [('Inner Race', '0.014'),('Outer Race', '0.007'),('Ball', '0.021')],
        [('Inner Race', '0.021'),('Outer Race', '0.014'),('Ball', '0.007')]
    ]
]

def get_fold(combination):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    prlzs = ['None', '6']
    fold = []
    for faulty_bearing, fault_bearing_severity in combination:
        filtered = filter_registers_by_key_value_sequence(
            registers, 
            [('sample_rate', [sample_rate]), 
            ('condition', [faulty_bearing]), 
            ('severity', [fault_bearing_severity]),
            ('prlz', prlzs)])
        fold.extend(filtered)
    return fold

def get_list_of_proposed_cross_validation_folds(comb_index=0):
    folds = []
    for combination in proposed_cross_validation_combinations[comb_index]:
        fold = get_fold(combination)
        folds.append(fold)
    return folds

def run_proposed_experiment(model, list_of_metrics):
    list_of_scores = []
    for comb_index in range(len(proposed_cross_validation_combinations)):
        scores = perform_cross_validation(model, list_of_metrics, segment_length, comb_index)
        list_of_scores.extend(scores)
    return list_of_scores

def perform_cross_validation(model, list_of_metrics, segment_length, comb_index):
    list_of_folds = get_list_of_proposed_cross_validation_folds(comb_index=comb_index)
    list_of_X_y = get_list_of_X_y(
            list_of_folds,
            raw_dir_path="raw_data/cwru",
            channels_columns=['DE'],
            segment_length=segment_length,
            load_acquisition_func=load_matlab_acquisition
        )
    scores = cross_validation(model, list_of_X_y, list_of_metrics=list_of_metrics)
    return scores
=======
def get_list_of_folds(faulty_bearing, sample_rate, combination):
    folds = []
    for normal_load, fault_bearing_severity in sehri_et_al_combination_rounds[combination%len(sehri_et_al_combination_rounds)]:
        fold = get_fold(normal_load=normal_load, fault_bearing_severity=fault_bearing_severity, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
        if len(fold) > 0:
            folds.append(fold)
    return folds


def single_channel_X_y(combination, segment_length, sample_rate, faulty_bearing, channel_column):
    list_of_folds = get_list_of_folds([faulty_bearing], sample_rate, combination)   
    return get_list_of_X_y(list_of_folds, raw_dir_path="raw_data/cwru", channels_columns=[channel_column], segment_length=segment_length, load_acquisition_func=load_matlab_acquisition)


def single_channel_X_y_DE_FE_48k(combination, segment_length):
    sample_rate = '48000'
    X_y_DE = single_channel_X_y(combination, segment_length, sample_rate, 'Drive End', 'DE')
    return X_y_DE
>>>>>>> 0b48f41d6b0eea3352b7411cdac0b62d995b3371
