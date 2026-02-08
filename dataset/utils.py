import os
import requests
import urllib.parse
import csv
import scipy.io
import numpy as np


def download_file_from_register(raw_dir_path, register):
    def download_from_url(url, file_path):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file from {url}: {e}")
            print(f"Trying again...")
    os.makedirs(raw_dir_path, exist_ok=True)
    file_url = urllib.parse.urljoin(register['base_url'], register['filename'])
    file_path = os.path.join(raw_dir_path, register['filename'])
    max_trials = 5
    while (not is_file_downloaded(file_url, raw_dir_path) or 
            not is_file_size_same(file_url, file_path)) and max_trials > 0:
        print(f"Downloading file {file_path}...")
        download_from_url(file_url, file_path)
        max_trials -= 1
    else:
        if max_trials == 0:
            raise Exception(f"Failed to download file {file_path} correctly after multiple attempts.")


def is_file_downloaded(url, folder_path):
    # Parse the URL to get the file name
    parsed_url = urllib.parse.urlparse(url)
    file_name = os.path.basename(parsed_url.path)    
    # Check if the file exists in the specified folder
    file_path = os.path.join(folder_path, file_name)
    return os.path.isfile(file_path)


def is_file_size_same(url, file_path):    
    # Check if the file exists
    if not os.path.isfile(file_path):
        return False    
    # Get the size of the local file
    local_file_size = os.path.getsize(file_path)    
    # Get the size of the file from the URL
    response = requests.head(url)
    if response.status_code != 200:
        return False
    url_file_size = int(response.headers.get('Content-Length', 0))    
    return local_file_size == url_file_size


def read_registers_from_config(config_path):
    registers = []
    with open(config_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row = {k.strip(): v.strip() if v is not None else v for k, v in row.items()}
            registers.append(row)
    return registers


def filter_registers_by_key_value_sequence(registers, key_value_sequence):
    return [reg for reg in registers if all(reg.get(k) in v for k, v in key_value_sequence)]


def filter_registers_by_key_value_absence(registers, key_value_sequence):
    return [reg for reg in registers if all(reg.get(k) not in v for k, v in key_value_sequence)]


def get_values_by_key(registers, key):
    return set([reg.get(key) for reg in registers if key in reg])


def get_all_keys_and_values(registers):
    for key in registers[0].keys():
        if key == 'filename':
            continue
        values = get_values_by_key(registers, key)
        print(f"{key}: {values}")


def load_matlab_file(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        return mat
    except Exception as e:
        print(f"Error loading MATLAB file {file_path}: {e}")
        raise e


def load_matlab_acquisition(file_path, channels):
    try:
        mat = load_matlab_file(file_path)
    except Exception as e:
        raise e
    acquisition = []
    for channel in channels:
        for key in mat.keys():
            if channel in key:
                acquisition.append(mat[key])
                break
    if acquisition:
        return np.concatenate(acquisition, axis=1)
    else:
        raise KeyError(f"Variables '{channels}' not found in the MATLAB file.")


def split_acquisition(acquisition, segment_length):
    num_segments = acquisition.shape[0] // segment_length
    segments = np.empty((num_segments, segment_length, acquisition.shape[1]), dtype=acquisition.dtype)
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments[i] = acquisition[start:end, :]
    return segments


def target_array(value, length):
    target_type = type(value)
    if target_type == str:
        target_type = np.dtype('U' + str(len(value)))
    return np.full(length, value, dtype=target_type)


def get_X_y(registers, raw_dir_path, channels_columns, segment_length, load_acquisition_func, train=False):
    X_list = []
    y_list = []
    if len(registers) == 0:
        return np.empty((0, segment_length, 1)), np.empty((0,), dtype='U10')
    for register in registers:
        segments, targets = extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, register)
        X_list.append(segments)
        y_list.append(targets)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

# from augmentation.frequency_domain import make_pairs_same_condition_diff_severity, augment_segments_mode3
# from augmentation.time_domain import make_pairs_same_condition_same_load, augment_time_mix_blocks, augment_segment_dam

# def get_X_y(registers, raw_dir_path, channels_columns, segment_length, load_acquisition_func, train=False):
#     X_list = []
#     y_list = []

#     if len(registers) == 0:
#         return np.empty((0, segment_length, 1)), np.empty((0,), dtype='U10')

#     for register in registers:
#         segments, targets = extract_segments_and_targets(
#             raw_dir_path, channels_columns, segment_length, load_acquisition_func, register
#         )

#         X_list.append(segments)
#         y_list.append(targets)

#         # DAM augmentation: add one augmented version for each segment
#         # if train and len(segments) > 0:
#         #     segments_dam = np.empty_like(segments, dtype=np.float32)
#         #     for i in range(len(segments)):
#         #         # Apply DAM to a single 1D segment
#         #         segments_dam[i, :, 0] = augment_segment_dam(segments[i, :, 0])
#         #     X_list.append(segments_dam)
#         #     y_list.append(targets)

#     if train:
#         pair_of_registers = make_pairs_same_condition_diff_severity(registers)
#         for reg_a, reg_b in pair_of_registers:
#             segments_a, targets_a = extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg_a)
#             segments_b, targets_b = extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg_b)

#             segments = augment_segments_mode3(segments_a, segments_b)
#             targets = targets_b if len(targets_a) > len(targets_b) else targets_a

#             X_list.append(segments)
#             y_list.append(targets)

#         pair_of_registers_td = make_pairs_same_condition_same_load(registers)
#         for reg_a, reg_b in pair_of_registers_td:
#             segments_a, targets_a = extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg_a)
#             segments_b, targets_b = extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg_b)

#             n = min(len(segments_a), len(segments_b))
#             if n == 0:
#                 continue

#             seg_len = segments_a.shape[1]

#             mixed1 = np.empty((n, seg_len, 1), dtype=np.float32)
#             mixed2 = np.empty((n, seg_len, 1), dtype=np.float32)

#             for i in range(n):
#                 x1 = segments_a[i, :, 0]
#                 x2 = segments_b[i, :, 0]

#                 y1, y2 = augment_time_mix_blocks(x1, x2)

#                 mixed1[i, :, 0] = y1[:seg_len]
#                 mixed2[i, :, 0] = y2[:seg_len]

#             targets = targets_b if len(targets_a) > len(targets_b) else targets_a
#             targets = targets[:n]

#             X_list.append(mixed1)
#             y_list.append(targets)

#             X_list.append(mixed2)
#             y_list.append(targets)

#     X = np.concatenate(X_list, axis=0)
#     y = np.concatenate(y_list, axis=0)
#     return X, y





# ========================================================================


def extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, register):
    acquisition = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, register)
    segments, targets = prepare_segments_and_targets(segment_length, register, acquisition)
    return segments,targets


def prepare_segments_and_targets(segment_length, register, acquisition):
    segments = split_acquisition(acquisition, segment_length=segment_length)
    targets = target_array(register['condition'], segments.shape[0])
    return segments,targets


def get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, register):
    file_path = os.path.join(raw_dir_path, register['filename'])
    channels = get_channels_from_register(channels_columns, register)
    try:
        acquisition = load_acquisition_func(file_path, channels=channels)
    except Exception as e:
        download_file_from_register(raw_dir_path, register)
        acquisition = load_acquisition_func(file_path, channels=channels)
    return acquisition


def get_channels_from_register(channels_columns, register):
    channels = []
    for channel_column in channels_columns:
        if channel_column not in register:
            raise KeyError(f"Channel column '{channel_column}' not found in register.")
        channels.append(register[channel_column])
    return channels


def concatenate_data(list_of_X_y):
    X_all = np.concatenate([X for X, _ in list_of_X_y], axis=0)
    y_all = np.concatenate([y for _, y in list_of_X_y], axis=0)
    return X_all, y_all


def get_list_of_X_y(list_of_folds, raw_dir_path, channels_columns, segment_length, load_acquisition_func):
    list_of_X_y = []
    for fold in list_of_folds:
        X, y = get_X_y(fold, raw_dir_path=raw_dir_path, channels_columns=channels_columns, 
                       segment_length=segment_length, load_acquisition_func=load_acquisition_func, train=train)
        list_of_X_y.append((X, y))
    return list_of_X_y

