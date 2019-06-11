import numpy as np
import pandas as pd
import os
import sys
import keras
from keras.models import Sequential
import time
import random
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN

from DataGenerator import DataGenerator
from sklearn.metrics import average_precision_score
from keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt

plt.switch_backend('agg')


PATH = 'RBNS/'
model_path = 'model'
# PATH = './'
PATH = './'

is_binary_model = True
use_shuffled_seqs = True

'''data parameters'''
valid_p = 0
max_size = 45
file_limit = 200000

'''model parameters'''
kernel_size = 8
num_of_kernels = 64
dense1_layer_size = 32
dense2_layer_size = 8

'''fit parameters'''
batch_size = 264
epochs = 2
workers = 1

model_path += '_samp{}_kernel{}_{}_dense1_{}_dense2_{}_batch{}_epoch{}_shuf_{}'.format(file_limit, kernel_size,
                                                                                      num_of_kernels,
                                                                                      dense1_layer_size,
                                                                                      dense2_layer_size,
                                                                                      batch_size, epochs,
                                                                                      use_shuffled_seqs)


class RBNSreader():

    @staticmethod
    def get_kmer_map(path, k):
        lines = []
        kmer_map = {}
        with open(path, 'r') as f:
            for line in f:
                seq = line.strip().split()[0]
                for i in range(len(seq)-k):
                    kmer = seq[i:i+k]
                    if 'N' in kmer:
                        continue
                    if kmer in kmer_map:
                        kmer_map[kmer] += 1
                    else:
                        kmer_map[kmer] = 1
        return kmer_map


    @staticmethod
    def read_files(rbns_files, file_limit=None, input_file=None, k=None):
        lines_from_all_files = []
        m_files = []
        medians = []
        if input_file and k:
            m_input = RBNSreader.get_kmer_map(input_file, k)
            m_files = [RBNSreader.get_kmer_map(file, k) for file in rbns_files]
            medians = [0 for file in rbns_files]
            for ind, m_file in enumerate(m_files):
                for key in m_file.keys():
                    if key not in m_input:
                        m_file[key] = 100
                    else:
                        m_file[key] = m_file[key]/m_input[key]
                l = [(key, m_file[key]) for key in m_file.keys()]
                l.sort(key=lambda a: a[1], reverse=True)
                medians[ind] = l[20][1]
                print(medians[ind])
        for ind, file in enumerate(rbns_files):
            m_file = None if input_file is None else m_files[ind]
            m_median = None if input_file is None else medians[ind]
            lines_from_all_files += RBNSreader.read_file(file, ind+1, file_limit, m_file, m_median, k)
        return lines_from_all_files

    @staticmethod
    def read_file(path, ind, file_limit, map=None, median=None, k=None):
        lines = []
        count = 0
        total = 0
        with open(path, 'r') as f:
            for line in f:
                total += 1
                seq = line.strip().split()[0]
                to_add = False if map else True
                if map:
                    for kmer in [seq[i:i+k] for i in range(len(seq)-k)]:
                        if kmer not in map or map[kmer] > median:
                            to_add = True
                            break
                if to_add:
                    lines.append((seq, ind))
                    count += 1
                if file_limit and count >= file_limit:
                    break
        print('total:{} count{}'.format(total, count))
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
    return lst


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

    model.add(Dense(dense1_layer_size))
    model.add(Activation('relu'))

    if dense2_layer_size > 0:
        model.add(Dense(dense2_layer_size))
        model.add(Activation('relu'))
        # model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.summary()

    return model


def load_files(args):
    if len(args) != 3:
        raise BaseException("Not enough arguments")
    else:
        cmpt_path = args[1]
        rbp_num = int(args[2])
        files = get_files_list(rbp_num)
        cmpt_file = read_file_rncmpt(cmpt_path)
        return files, cmpt_file, rbp_num


def calc_corr(seqs, y_test, y_pred, num_classes):
    if is_binary_model:
        y_pred_scores = [y[0] for y in y_pred]
    else:
        y_pred_scores = [np.dot(y, np.concatenate((np.ones(1) * -10, np.ones(num_classes - 1)))) for y in y_pred]

    # print(y_pred_scores)
    x_test_y_pred = list(zip(seqs, y_pred_scores, list(range(len(seqs)))))
    np.random.shuffle(x_test_y_pred)
    x_test_y_pred_sorted = sorted(x_test_y_pred, key=lambda x: x[1], reverse=True)
    # print(x_test_y_pred_sorted[:20])
    x_test_y_pred_tagged = [(seq, tag, int(ind)) for (seq, score, ind), tag in zip(x_test_y_pred_sorted, y_test)]
    x_test_y_pred_tagged = sorted(x_test_y_pred_tagged, key=lambda x: x[2])

    # print(x_test_y_pred_tagged[:20])
    positives = [tag for (seq, tag, ind) in x_test_y_pred_tagged[:1000]]

    print('positives', sum(positives))

    avg_precision = average_precision_score([tag for (seq, tag, ind) in x_test_y_pred_tagged], [int(x) for x in
                                                                                                np.append(np.ones(1000),
                                                                                                          np.zeros(len(
                                                                                                              x_test) - 1000),
                                                                                                          axis=0)])
    print('avg_precision', avg_precision)

    plt.figure(figsize=(20, 10))
    plt.scatter(range(1, len(y_pred_scores) + 1), y_pred_scores)
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


def create_train_valid_data(files):
    lines_from_all_files = RBNSreader.read_files(files[1:], file_limit=file_limit)
    if use_shuffled_seqs:
        lines_from_all_files += create_negative_seqs(lines_from_all_files)
    else:
        lines_from_all_files += RBNSreader.read_file(files[0], 0, file_limit * 2)

    np.random.shuffle(lines_from_all_files)

    print(lines_from_all_files[:10])

    valid_n = int(valid_p * len(lines_from_all_files))

    validation_data = lines_from_all_files[:valid_n]
    train_data = lines_from_all_files[valid_n:]

    print('validation size', len(validation_data))
    print('train size', len(train_data))

    train_gen = DataGenerator(train_data, num_of_classes=num_of_classes, kernel_size=kernel_size,
                              max_sample_size=max_size,
                              batch_size=batch_size, shuffle=True)

    valid_gen = None
    if valid_n > 0:
        valid_gen = DataGenerator(validation_data, num_of_classes=num_of_classes, kernel_size=kernel_size,
                                  max_sample_size=max_size,
                                  batch_size=batch_size, shuffle=True)

    return train_gen, valid_gen


def train_model(files, loss_func='categorical_crossentropy'):
    train_gen, valid_gen = create_train_valid_data(files)
    model = create_model(train_gen.dim, num_of_classes)
    opt = keras.optimizers.Adam()

    # Let's train the model using RMSprop
    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        use_multiprocessing=False,
                        epochs=epochs,
                        workers=workers)

    model.save(model_path)

    return model


if __name__ == '__main__':

    start = time.time()

    print('Starting')
    files, cmpt_seqs, rbp_num = load_files(sys.argv)

    print(files)
    files = [files[0]] + files[-1:]
    print(files)
    model_path += '_rbp' + str(rbp_num) + 'files_' + '_'.join([str(cons) for file, cons in files[1:]])
    print(model_path)

    files = [file for file, cons in files]
    num_of_classes = len(files) + 1
    loss_func = 'categorical_crossentropy'
    if is_binary_model:
        num_of_classes = 1
        loss_func = 'binary_crossentropy'

    exists = os.path.isfile(model_path)
    if exists:
        model = keras.models.load_model(model_path)
    else:
        model = train_model(files, loss_func)

    x_test = np.array([DataGenerator.one_hot(seq, max_size, kernel_size) for seq in cmpt_seqs])
    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
    y_pred = model.predict(x_test)
    calc_corr(cmpt_seqs, y_test, y_pred, num_of_classes)

    end = time.time()
    print('took', (end - start) / 60, 'minutes')
