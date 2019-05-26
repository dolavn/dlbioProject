import numpy as np
import keras


reverse_compliment_base = {'A': 'U', 'C': 'G', 'T': 'A', 'G': 'C', 'N': 'N'}


class DataGenerator(keras.utils.Sequence):
    def __init__(self, rbns_files, kernel_size, max_sample_size, batch_size=16,
                 file_limit=None, shuffle=False, _validation=False):
        if _validation:
            return
        self.kernel_size = kernel_size
        self.max_sample_size = max_sample_size
        self.file_limit = file_limit
        self.dim = (max_sample_size + 2 * kernel_size - 2, 4, 1)
        self.rbns_files = rbns_files
        self.batch_size = batch_size
        self.lines = []

        for ind, file in enumerate(rbns_files):
            self.read_file(file, ind)
        self.size = len(self.lines)
        self.shuffle = shuffle
        self.indexes = np.arange(self.size)

        self.on_epoch_end()

    def read_file(self, path, ind):
        count = 0
        with open(path, 'r') as f:
            for line in f:
                seq = line.strip().split()[0]
                self.lines.append((seq, ind))
                count = count + 1
                if self.file_limit is not None and count > self.file_limit:
                    break

    def __len__(self):
        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self._data_generation(indices)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, indices_list):
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, len(self.rbns_files)))
        for i, ind in enumerate(indices_list):
            curr_x = self.one_hot_encode(self.lines[ind][0])
            x[i, ] = curr_x
            y[i] = [1 if j == self.lines[ind][1] else 0 for j in range(len(self.rbns_files))]
        return x, y

    @staticmethod
    def reverse_compliment(string):
        return string.replace('T', 'U')
        # string = [reverse_compliment_base[base] for base in string[::-1]]
        # return ''.join(string)

    def pad(self, string):
        string += 'N' * (self.max_sample_size - len(string))
        return string

    def pad_conv(self, string):
        pad = 'N' * (self.kernel_size - 1)
        string = pad + string + pad
        return string

    def one_hot(self, string):
        encoding = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]),
                    'C': np.array([0, 0, 1, 0]), 'U': np.array([0, 0, 0, 1]),
                    'N': np.array([0.25] * 4)}

        padded_string = self.pad(string)
        vec_list = [encoding[c].reshape(1, -1) for c in self.pad_conv(padded_string)]
        return np.concatenate(vec_list, axis=0).reshape(len(vec_list), 4, 1)

    def one_hot_encode(self, line):
        return self.one_hot(DataGenerator.reverse_compliment(line))

    def get_files_num(self):
        return len(self.rbns_files)

    def get_validation(self, p):
        v = DataGenerator([], 0, 0, _validation=True)
        v.kernel_size = self.kernel_size
        v.max_sample_size = self.max_sample_size
        v.file_limit = self.file_limit
        v.dim = self.dim
        v.rbns_files = self.rbns_files
        v.batch_size = self.batch_size
        v.lines = self.lines
        v.size = self.size
        v.shuffle = self.shuffle
        np.random.shuffle(self.indexes)
        valid_n = int(p*self.size)
        v.indexes = self.indexes[:valid_n]
        self.indexes = self.indexes[valid_n:]
        self.size = self.size-valid_n
        v.size = valid_n
        self.on_epoch_end()
        v.on_epoch_end()
        return v