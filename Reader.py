import numpy as np
import os
import sys
import keras
from keras.models import Sequential
import time
import json
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN

from DataGenerator import DataGenerator
from sklearn.metrics import average_precision_score
from keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt

plt.switch_backend('agg')

PATH = 'RBNS/'
model_path = 'model'
PATH = './'

is_binary_model = True
use_shuffled_seqs = True

'''data parameters'''
valid_p = 0.0
max_size = 45
file_limit = 2000

'''model parameters'''
kernel_size = 9
num_of_kernels = 60
DENSE_LAYERS = [40]

'''fit parameters'''
batch_size = 256
epochs = 2
workers = 1

dim = (max_size + 2 * kernel_size - 2, 4, 1)

layers_description = ''.join(['dense{}_{}'.format(i+1, dense) for i, dense in enumerate(DENSE_LAYERS)])
model_path += '_samp{}_kernel{}_{}_{}_batch{}_epoch{}_shuf_{}_binary{}'.format(file_limit, kernel_size,
                                                                                      num_of_kernels,
                                                                                      layers_description,
                                                                                      batch_size, epochs,
                                                                                      use_shuffled_seqs,
                                                                                        is_binary_model)


class RBNSreader():
    @staticmethod
    def get_kmer_map(path, k):
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
    def read_files(rbns_files, file_limit=None, input_file=None, k=None, threshold_val=None):
        lines_from_all_files = []
        m_files = []
        medians = []
        print(rbns_files)
        if input_file and k:
            file_name = input_file[input_file.find('./')+2:input_file.find('.seq')]
            path = "dict_input_{}_k{}".format(file_name, k)
            dict_exists = os.path.isfile(path)
            if dict_exists:
                with open(path) as dict_file:
                    m_files, _ = json.load(dict_file)
            else:
                m_input = RBNSreader.get_kmer_map(input_file, k)
                m_files = [RBNSreader.get_kmer_map(file, k) for file in rbns_files]
                medians = [0 for file in rbns_files]
                for ind, m_file in enumerate(m_files):
                    for key in m_file.keys():
                        if key not in m_input:
                            m_file[key] = 100
                        else:
                            m_file[key] = m_file[key] / m_input[key]
                with open(path, 'w') as dict_file:
                    json.dump((m_files, medians), dict_file)
            medians = [0 for _ in range(len(m_files))]
            for ind, m_file in enumerate(m_files):
                l = [(key, m_file[key]) for key in m_file.keys()]
                l.sort(key=lambda a: a[1], reverse=True)
                medians[ind] = threshold_val
                print(medians[ind])
                print(sum([a[1] for a in l])/len(l))
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


def create_model(dim, num_classes, dense_layers=None):
    model = Sequential()
    model.add(Conv2D(num_of_kernels, (kernel_size, 4), strides=(1, 1), padding='valid', input_shape=dim))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    if not dense_layers:
        dense_layers = DENSE_LAYERS
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


def calc_auc(y_pred, x_test):
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
    return avg_precision
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


def create_train_valid_data(negative_file, positive_files, num_of_classes, threshold_val=None):
    curr_k = 5
    inp = negative_file
    if not threshold_val:
        curr_k = None
        inp = None
    print(positive_files)
    lines_from_all_files = RBNSreader.read_files(positive_files, file_limit=file_limit, input_file=inp, k=curr_k,
                                                 threshold_val=threshold_val)
    if use_shuffled_seqs:
        lines_from_all_files += create_negative_seqs(lines_from_all_files)
    else:
        lines_from_all_files += RBNSreader.read_file(negative_file, 0, file_limit * 2)

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


def create_and_compile_model(dim, num_of_classes, loss_func, dense_layers=None):
    model = create_model(dim, num_of_classes, dense_layers=dense_layers)
    opt = keras.optimizers.Adam(lr=0.01)

    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def train_model(model, train_gen, valid_gen):

    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        use_multiprocessing=False,
                        epochs=epochs,
                        workers=workers)
    return model


def box_plot(data, name):
    fig = plt.figure(figsize=(9, 6))
    tags = [d[0] for d in data]
    tags = [d if d else 'None' for d in tags]
    values = [d[1] for d in data]
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.set_title(name)
    ax.set_xlabel('params')
    ax.set_ylabel('accuracy')
    # Create the boxplot
    bp = ax.boxplot(values)
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ax.set_xticklabels(tags, fontsize=10)
    # Save the figure
    fig.savefig('{}.png'.format(name), bbox_inches='tight')


def plot_data(results):
    for rbp in results.keys():
        data = results[rbp]
        box_plot(data, 'RBP{}'.format(rbp))
    exit()


def calc_statistics(rbp_list, dense_list, threshold_list, output_file):
    rncmpt_path = 'RNCMPT/RBP{}_RNCMPT.sorted'
    args = [(sys.argv[0], rncmpt_path.format(rbp), rbp) for rbp in rbp_list]
    args = [load_files(args) for args in args]
    results = {}
    start = time.time()
    for files, cmpt_seqs, rbp_num in args:
        files = [file for file, cons in files]
        precision_scores = []
        num_of_classes = len(files)
        loss_func = 'categorical_crossentropy'
        if is_binary_model:
            num_of_classes = 1
            loss_func = 'binary_crossentropy'
        for dense_layer in dense_list:
            for t in threshold_list:
                NUM_OF_TRIALS = 2
                for _ in range(NUM_OF_TRIALS):
                    model = create_and_compile_model(dim, num_of_classes, loss_func, dense_layers=dense_layer)
                    print('training model')
                    train_gen, valid_gen = create_train_valid_data(negative_file=files[0], positive_files=files[1:],
                                                                   threshold_val=t, num_of_classes=num_of_classes)
                    model = train_model(model, train_gen, valid_gen)
                    model.save(model_path)

                    x_test = np.array([DataGenerator.one_hot(seq, max_size, kernel_size) for seq in cmpt_seqs])
                    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
                    y_pred = model.predict(x_test)
                    precision = calc_auc(y_pred, x_test)
                    precision_scores.append(((dense_layer, t), precision))
        results[rbp_num] = precision_scores
        with open(output_file, 'w') as data_file:
            json.dump(results, data_file)
    end = time.time()
    print('took', (end - start) / 60, 'minutes')
    print(results)


def load_data(f_name):
    with open(f_name) as data_file:
        results = json.load(data_file)
        plot_data(results)


if __name__ == '__main__':
    if sys.argv[1] == '-stats':
        f_name = sys.argv[2]
        calc_statistics([2, 7], [[], [16], [32]], [None], f_name)
        exit()
    if sys.argv[1] == '-plot':
        f_name = sys.argv[2]
        load_data(f_name)
        exit()
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
        train_gen, valid_gen = create_train_valid_data(negative_file=files[0], positive_files=files[1:],
                                                       num_of_classes=num_of_classes)
        model = train_model(model, train_gen, valid_gen)
        model.save(model_path)

    x_test = np.array([DataGenerator.one_hot(seq, max_size, kernel_size) for seq in cmpt_seqs])
    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
    y_pred = model.predict(x_test)
    calc_auc(y_pred, x_test)

    end = time.time()
    print('took', (end - start) / 60, 'minutes')