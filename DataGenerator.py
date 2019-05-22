import numpy as np
import keras


# max_size = 45
# kernel_size = 12

# reverse_compliment_base = {'A': 'U', 'C': 'G', 'T': 'A', 'G': 'C', 'N': 'N'}

class DataGenerator(keras.utils.Sequence):
    def __init__(self, rbns_files, kernel_size, max_sample_size, batch_size=32, shuffle=False):
        self.kernel_size = kernel_size
        self.max_sample_size = max_sample_size

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
        with open(path, 'r') as f:
            for line in f:
                seq = line.strip().split()[0]
                self.lines.append((seq, ind))

    def __len__(self):
        return int(self.size / self.batch_size)

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
            x[i,] = curr_x
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
