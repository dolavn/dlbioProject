import numpy as np
import keras
max_size = 45
kernel_size = 12

reverse_compliment_base = {'A': 'U', 'C': 'G', 'T': 'A', 'G': 'C', 'N': 'N'}


def reverse_compliment(string):
    return string.replace('T', 'U')
    string = [reverse_compliment_base[base] for base in string[::-1]]
    return ''.join(string)


def pad(string, max_size):
    string += 'N' * (max_size-len(string))
    return string


def pad_conv(string, kernel_size):
    pad = 'N'*(kernel_size-1)
    string = pad + string + pad
    return string


def one_hot(string):
    dict = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]),
            'C': np.array([0, 0, 1, 0]), 'U': np.array([0, 0, 0, 1]),
            'N': np.array([0.25]*4)}
    vec_list = [dict[c].reshape(1, -1) for c in pad_conv(pad(string, max_size), kernel_size)]
    return np.concatenate(vec_list, axis=0).reshape(len(vec_list), 4, 1)


def one_hot_encode(line):
    return one_hot(reverse_compliment(line))


def read_file(path, ind, list):
    with open(path, 'r') as f:
        for line in f:
            seq = line.strip().split()[0]
            list.append((seq, ind))


class DataGenerator(keras.utils.Sequence):
    def __init__(self, rbns_files, batch_size=32, dim=(max_size+2*kernel_size-2, 4, 1), shuffle=False):
        self.rbns_files = rbns_files
        self.batch_size = batch_size
        self.lines = []
        for ind, file in enumerate(rbns_files):
            read_file(file, ind, self.lines)
        self.size = len(self.lines)
        self.shuffle = shuffle
        self.dim = dim
        self.indexes = np.arange(self.size)
        self.on_epoch_end()

    def __len__(self):
        return int(self.size/self.batch_size)

    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self._data_generation(indices)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        if self.shuffle:
            self.indexes = np.random.shuffle(self.indexes)

    def _data_generation(self, indices_list):
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, len(self.rbns_files)))
        for i, ind in enumerate(indices_list):
            curr_x = one_hot_encode(self.lines[ind][0])
            x[i, ] = curr_x
            y[i] = [1 if j == self.lines[ind][1] else 0 for j in range(len(self.rbns_files))]
        return x, y

