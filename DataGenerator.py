import numpy as np
import keras


reverse_compliment_base = {'A': 'U', 'C': 'G', 'T': 'A', 'G': 'C', 'N': 'N'}


class DataGenerator(keras.utils.Sequence):

    def __init__(self, lines, num_of_classes, kernel_sizes, max_sample_size, batch_size=16, shuffle=False):

        self.kernel_sizes = kernel_sizes
        self.max_sample_size = max_sample_size
        self.dim = [(max_sample_size + 2 * kernel_size - 2, 4, 1) for kernel_size in self.kernel_sizes]
        self.batch_size = batch_size

        self.lines = lines
        self.num_of_classes = num_of_classes

        self.size = len(self.lines)

        self.shuffle = shuffle
        self.indexes = np.arange(self.size)

        self.valid_generator = None

        self.on_epoch_end()

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
        xs = []
        y = np.empty((self.batch_size, self.num_of_classes))
        for kernel_size, dim in zip(self.kernel_sizes, self.dim):
            x = np.empty((self.batch_size, *dim))
            for i, ind in enumerate(indices_list):
                curr_x = DataGenerator.one_hot(self.lines[ind][0], self.max_sample_size, kernel_size)
                x[i, ] = curr_x
            xs.append(x)
        for i, ind in enumerate(indices_list):
            if self.num_of_classes == 1:
                y[i, ] = 1 if self.lines[ind][1] > 0 else 0
            else:
                y[i] = [1 if file_ind == self.lines[ind][1] else 0 for file_ind in range(self.num_of_classes)]
        return xs, y

    @staticmethod
    def reverse_compliment(string):
        return string.replace('T', 'U')
        # string = [reverse_compliment_base[base] for base in string[::-1]]
        # return ''.join(string)

    @staticmethod
    def pad(string, max_size):
        string += 'N' * (max_size - len(string))
        return string

    @staticmethod
    def pad_conv(string, kernel_size):
        pad = 'N' * (kernel_size - 1)
        string = pad + string + pad
        return string

    @staticmethod
    def one_hot(string, max_size, kernel_size):
        encoding = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]),
                    'C': np.array([0, 0, 1, 0]), 'U': np.array([0, 0, 0, 1]),
                    'T': np.array([0, 0, 0, 1]),
                    'N': np.array([0.25] * 4)}

        padded_string = DataGenerator.pad(string, max_size)
        vec_list = [encoding[c].reshape(1, -1) for c in DataGenerator.pad_conv(padded_string, kernel_size)]
        return np.concatenate(vec_list, axis=0).reshape(len(vec_list), 4, 1)

    #def one_hot_encode(self, line):
    #    return self.one_hot(DataGenerator.reverse_compliment(line))
