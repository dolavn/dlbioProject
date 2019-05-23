import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
import time

from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN


from DataGenerator import DataGenerator
from sklearn.metrics import average_precision_score
from keras.layers import GlobalMaxPooling2D


PATH = 'RBNS_test/'
PATH = './'
kernel_size = 12
max_size = 45


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
    model.add(Conv2D(32, (kernel_size, 4), strides=(1, 1), padding='same', input_shape=dim))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    #model.add(MaxPooling2D(pool_size=(dim[0]-kernel_size+1, 4)))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    return model


def load_files(args):
    if len(args) != 3:
        raise BaseException("Not enough arguments")
    else:
        cmpt_path = args[1]
        rbns_num = int(args[2])
        l = get_files_list(rbns_num)
        print(l)
        cmpt_file = read_file_rncmpt(cmpt_path)
        return l, cmpt_file


def calc_corr(seqs, y_test, y_pred, num_classes):
    y_pred_scores = [np.dot(y, np.concatenate((np.ones(1)*-1, np.ones(num_classes-1)))) for y in y_pred]
    x_test_y_pred = list(zip(seqs, y_pred_scores, list(range(len(seqs)))))
    x_test_y_pred = np.random.permutation(x_test_y_pred)
    x_test_y_pred_sorted = sorted(x_test_y_pred, key=lambda x: x[1], reverse=True)
    #print(x_test_y_pred_sorted[:20])
    x_test_y_pred_tagged = [(seq, tag, int(ind)) for (seq, score, ind), tag in zip(x_test_y_pred_sorted, y_test)]
    x_test_y_pred_tagged = sorted(x_test_y_pred_tagged, key=lambda x: x[2])

    #print(x_test_y_pred_tagged[:20])
    positives = sum([tag for (seq, tag, ind) in x_test_y_pred_tagged[:1000]])

    print('positives', positives)

    avg_precision = average_precision_score([tag for (seq, tag, ind) in x_test_y_pred_tagged], [int(x) for x in
                                                     np.append(np.ones(1000), np.zeros(len(x_test) - 1000),
                                                               axis=0)])
    print('avg_precision', avg_precision)


if __name__ == '__main__':

    start = time.time()

    print('Starting')
    l, cmpt_seqs = load_files(sys.argv)
    d = DataGenerator(l, kernel_size, max_size, file_limit=200000, batch_size=264)
    print(d.get_files_num())
    model = create_model(d.dim, d.get_files_num())
    opt = keras.optimizers.Adam()

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit_generator(generator=d,
                        use_multiprocessing=True,
                        epochs=10,
                        workers=30)

    x_test = np.array([d.one_hot(seq) for seq in cmpt_seqs])
    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
    y_pred = model.predict(x_test)
    calc_corr(cmpt_seqs, y_test, y_pred, d.get_files_num())

    end = time.time()
    print('took', (end - start)/60, 'minutes')
