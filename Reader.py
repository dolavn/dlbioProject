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
model_path = 'model'
# PATH = './'

is_binary_model = True
use_shuffled_seqs = True

'''data parameters'''
valid_p = 0.2
max_size = 45
file_limit = 250000

'''model parameters'''
kernel_size = 9
num_of_kernels = 60
dense_layers = [40]
dense1_layer_size = 40
dense2_layer_size = 0
dense3_layer_size = 0

'''fit parameters'''
batch_size = 256
epochs = 4
workers = 1

dim = (max_size + 2 * kernel_size - 2, 4, 1)

model_path += '_samp{}_kernel{}_{}_dense1_{}_dense2_{}_batch{}_epoch{}_shuf_{}_binary{}'.format(file_limit, kernel_size,
                                                                                      num_of_kernels,
                                                                                      dense1_layer_size,
                                                                                      dense2_layer_size,
                                                                                      batch_size, epochs,
                                                                                      use_shuffled_seqs,
                                                                                        is_binary_model)


class RBNSreader():
    @staticmethod
    def read_files(rbns_files, file_limit=None):
        #file_limit2 = file_limit
        #if file_limit != None:
            #file_limit2 = file_limit * 4
        lines_from_all_files = []
        for ind, file in enumerate(rbns_files):
            lines = RBNSreader.read_file(file, ind + 1, file_limit)
            #np.random.shuffle(lines)
            #lines_from_all_files += lines[:file_limit]
            lines_from_all_files += lines[:file_limit]

        return lines_from_all_files

    @staticmethod
    def read_file(path, ind, file_limit):
        lines = []
        count = 0
        more_than_one_occ = 0
        with open(path, 'r') as f:
            for line in f:
                seq, val = line.strip().split()

                #if int(val) > 1:
                #    more_than_one_occ += 1
                #if count % 30 == 0:
                #    lines.append((seq, ind))
                #count += 1
                if file_limit and len(lines) >= file_limit:
                    break

        print('more than one occ', more_than_one_occ)
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

    for dense_size in dense_layers:
        model.add(Dense(dense_size))
        model.add(Activation('relu'))

    #model.add(Dropout(0.25))

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


def calc_auc(y_pred):

    if is_binary_model:
        y_pred_scores = [y[0] for y in y_pred]
    else:
        print(y_pred[0])
        print(np.concatenate((np.ones(1) * -10, np.array([10, 100]))))
        y_pred_scores = [np.dot(y, np.concatenate((np.ones(1) * -10, np.array([20, 10])))) for y in y_pred]

    true = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]

    avg_precision = average_precision_score(true, y_pred_scores)
    print('avg_precision', avg_precision)

    plt.figure(figsize=(20, 10))
    plt.scatter(range(1, len(y_pred_scores) + 1), y_pred_scores)
    plt.savefig('scores.png')

    '''
    with open('positives.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in range(len(positives)) if positives[i] == 1]))
    '''


def create_negative_seqs(lines_from_all_files):
    negative_seqs = []
    for seq, ind in lines_from_all_files:
        seq_list = [ch for ch in seq]
        np.random.shuffle(seq_list)
        negative_seqs.append((''.join(seq_list), 0))
    return negative_seqs


def create_train_valid_data(negative_file, positive_files):
    lines_from_all_files = RBNSreader.read_files(positive_files, file_limit=file_limit)
    if use_shuffled_seqs:
        lines_from_all_files += create_negative_seqs(lines_from_all_files)
    else:
        lines_from_all_files += RBNSreader.read_file(negative_file, 0, file_limit)

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

def create_and_compile_model(dim, num_of_classes, loss_func):

    model = create_model(dim, num_of_classes)
    opt = keras.optimizers.Adam(lr=0.01)

    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def train_model(model, train_gen, valid_gen):

    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        use_multiprocessing=True,
                        epochs=epochs,
                        workers=workers)

    return model


if __name__ == '__main__':

    start = time.time()

    print('Starting')
    files, cmpt_seqs, rbp_num = load_files(sys.argv)

    print(files)
    files = [files[0]] + files[-2:]
    print(files)
    model_path += '_rbp' + str(rbp_num) + 'files_' + '_'.join([str(cons) for file, cons in files[1:]])
    print(model_path)

    files = [file for file, cons in files]
    num_of_classes = len(files)
    print('classes', num_of_classes)
    loss_func = 'categorical_crossentropy'
    if is_binary_model:
        num_of_classes = 1
        loss_func = 'binary_crossentropy'

    exists = os.path.isfile(model_path)
    if exists:
        print('loading model')
        model = keras.models.load_model(model_path)
    else:

        model = create_and_compile_model(dim, num_of_classes, loss_func)

        print('training model')
        #for file in files[1:]:
        train_gen, valid_gen = create_train_valid_data(negative_file=files[0], positive_files=files[1:])
        model = train_model(model, train_gen, valid_gen)
        model.save(model_path)

    x_test = np.array([DataGenerator.one_hot(seq, max_size, kernel_size) for seq in cmpt_seqs])
    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
    y_pred = model.predict(x_test)
    calc_auc(y_pred)

    end = time.time()
    print('took', (end - start) / 60, 'minutes')
