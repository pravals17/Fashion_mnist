import numpy as np

def split_rows(data, proportion):
    size = data.shape[0]
    split_idx = int(size * proportion)
    np.random.shuffle(data)
    return data[:split_idx], data[split_idx:]

def split_columns(data, column_idx1, column_idx2):
    new_data = np.hsplit(data,[column_idx1, colum_idx2])
    return new_data[0], new_data[1]

def normalize_data(data):
    return (data/255)

def one_hot_encode(data, num_classes):
    data = np.array(data, dtype = np.int8).reshape(-1)
    return np.eye(num_classes)[data]

