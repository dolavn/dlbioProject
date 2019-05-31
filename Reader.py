import numpy as np
import pandas as pd
import os
import sys
import keras
from keras.models import Sequential
import time

from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN


from DataGenerator import DataGenerator
from sklearn.metrics import average_precision_score
from keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt

plt.switch_backend('agg')


PATH = 'RBNS/'
#PATH = './'

'''data parameters'''
valid_p = 0.25
max_size = 45
file_limit = 200000

'''model parameters'''
kernel_size = 14
num_of_kernels = 128
dense_layer_size = 32

'''fit parameters'''
batch_size = 264
epochs = 5
workers = 1


class RBNSreader():

    @staticmethod
    def read_files(rbns_files, file_limit=None):
        lines_from_all_files = []
        for ind, file in enumerate(rbns_files):
            lines_from_all_files += RBNSreader.read_file(file, ind+1, file_limit)

        return lines_from_all_files

    @staticmethod
    def read_file(path, ind, file_limit):
        lines = []
        count = 0
        with open(path, 'r') as f:
            for line in f:
                seq = line.strip().split()[0]
                lines.append((seq, ind))
                count += 1
                if file_limit and count >= file_limit:
                    break
        return lines

def get_files_list(rbp_ind):
    lst = []
    for file in os.listdir(PATH):
        if file.startswith('RBP' + str(rbp_ind) + '_'):
            suffix = file.split('.')[1]
            if suffix != 'seq':
                continue
            concentration = file.split('_')[1].split('nM')[0]
            concentration_val = 0
            if concentration != 'input.seq':
                concentration_val = int(concentration)

            lst.append((PATH + file, concentration_val))

    lst.sort(key=lambda x: x[1])
    return [file for file, cons in lst]


def read_file_rbns(path, max_len=-1):
    sequences = []

    with open(path, 'r') as f:
        for line in f:
            sequences.append(line.strip().split())
            if 0 < max_len <= len(sequences):
                break
    return sequences


def read_file_rncmpt(file_path):
    sequences = []
    with open(file_path) as f:
        for line in f:
            sequences.append(line.strip())
    return sequences


def sum_vec(arr):
    return np.sum(arr)


def create_model(dim, num_classes):

    model = Sequential()
    model.add(Conv2D(num_of_kernels, (kernel_size, 4), strides=(1, 1), padding='valid', input_shape=dim))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(dense_layer_size))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    return model


def load_files(args):
    if len(args) != 3:
        raise BaseException("Not enough arguments")
    else:
        cmpt_path = args[1]
        rbns_num = int(args[2])
        l = get_files_list(rbns_num)
        cmpt_file = read_file_rncmpt(cmpt_path)
        return l, cmpt_file


def calc_corr(seqs, y_test, y_pred, num_classes):
    #y_pred_scores = [np.dot(y, np.concatenate((np.ones(1)*-1, np.ones(num_classes-1)))) for y in y_pred]
    y_pred_scores = [y[0] for y in y_pred]

    #print(y_pred_scores)
    x_test_y_pred = list(zip(seqs, y_pred_scores, list(range(len(seqs)))))
    np.random.shuffle(x_test_y_pred)
    x_test_y_pred_sorted = sorted(x_test_y_pred, key=lambda x: x[1], reverse=True)
    #print(x_test_y_pred_sorted[:20])
    x_test_y_pred_tagged = [(seq, tag, int(ind)) for (seq, score, ind), tag in zip(x_test_y_pred_sorted, y_test)]
    x_test_y_pred_tagged = sorted(x_test_y_pred_tagged, key=lambda x: x[2])

    #print(x_test_y_pred_tagged[:20])
    positives = [tag for (seq, tag, ind) in x_test_y_pred_tagged[:1000]]

    print('positives', sum(positives))

    avg_precision = average_precision_score([tag for (seq, tag, ind) in x_test_y_pred_tagged], [int(x) for x in
                                                     np.append(np.ones(1000), np.zeros(len(x_test) - 1000),
                                                               axis=0)])
    print('avg_precision', avg_precision)

    plt.figure(figsize=(20, 10))
    plt.scatter(range(1, len(y_pred_scores)+1), y_pred_scores)
    plt.savefig('scores.png')

    with open('positives.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in range(len(positives)) if positives[i] == 1]))


def create_negative_seqs(lines_from_all_files):
    negative_seqs = []
    for seq, ind in lines_from_all_files:
        seq_list = [ch for ch in seq]
        np.random.shuffle(seq_list)
        negative_seqs.append((''.join(seq_list), 0))
    return negative_seqs



if __name__ == '__main__':

    start = time.time()

    print('Starting')
    files, cmpt_seqs = load_files(sys.argv)
    print(files)
    files = files[-2:]
    print(files)
    num_of_files = len(files)

    lines_from_all_files = RBNSreader.read_files(files, file_limit=file_limit)
    lines_from_all_files += create_negative_seqs(lines_from_all_files)
    np.random.shuffle(lines_from_all_files)

    valid_n = int(valid_p * len(lines_from_all_files))

    validation_data = lines_from_all_files[:valid_n]
    train_data = lines_from_all_files[valid_n:]

    print('validation size', len(validation_data))
    print('train size', len(train_data))

    train_gen = DataGenerator(train_data, num_of_files=num_of_files, kernel_size=kernel_size, max_sample_size=max_size,
                              batch_size=batch_size, shuffle=True)

    valid_gen = None
    if valid_n > 0:
        valid_gen = DataGenerator(validation_data, num_of_files=num_of_files, kernel_size=kernel_size, max_sample_size=max_size,
                              batch_size=batch_size, shuffle=True)

    model = create_model(train_gen.dim, num_of_files)
    opt = keras.optimizers.Adam()

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        use_multiprocessing=True,
                        epochs=epochs,
                        workers=workers)

    x_test = np.array([train_gen.one_hot(seq) for seq in cmpt_seqs])
    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
    y_pred = model.predict(x_test)
    calc_corr(cmpt_seqs, y_test, y_pred, num_of_files)

    end = time.time()
    print('took', (end - start)/60, 'minutes')
