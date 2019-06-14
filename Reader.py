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
valid_p = 0
max_size = 45
file_limit = 200000

'''model parameters'''
kernel_size = 8
num_of_kernels = 64
dense1_layer_size = 64
dense2_layer_size = 0

'''fit parameters'''
batch_size = 264
epochs = 2
workers = 16

model_path += '_samp{}_kernel{}_{}_dense1_{}_dense2_{}_batch{}_epoch{}_shuf_{}'.format(file_limit, kernel_size,
                                                                                      num_of_kernels,
                                                                                      dense1_layer_size,
                                                                                      dense2_layer_size,
                                                                                      batch_size, epochs,
                                                                                      use_shuffled_seqs)


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
                seq, occ = line.strip().split()
                if int(occ) <= 1 and False:
                    continue
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


def create_model(dim, num_classes, dense_layer1, dense_layer2):
    model = Sequential()
    model.add(Conv2D(num_of_kernels, (kernel_size, 4), strides=(1, 1), padding='valid', input_shape=dim))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(dense_layer1))
    model.add(Activation('relu'))

    if dense_layer2 > 0:
        model.add(Dense(dense_layer2))
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
    pos_num = sum(positives)
    print('positives', pos_num)

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

    return pos_num


def create_negative_seqs(lines_from_all_files):
    negative_seqs = []
    for seq, ind in lines_from_all_files:
        seq_list = [ch for ch in seq]
        np.random.shuffle(seq_list)
        negative_seqs.append((''.join(seq_list), 0))
    return negative_seqs


def create_train_valid_data(files, threshold_val=None):
    curr_k = 5
    inp = files[0]
    if not threshold_val:
        curr_k = None
        inp = None
    lines_from_all_files = RBNSreader.read_files(files[1:], file_limit=file_limit, input_file=inp, k=curr_k,
                                                 threshold_val=threshold_val)
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


def train_model(files, loss_func='categorical_crossentropy', threshold_val=None, dense_layer1=dense1_layer_size,
                dense_layer2=dense2_layer_size):
    train_gen, valid_gen = create_train_valid_data(files, threshold_val=threshold_val)
    model = create_model(train_gen.dim, num_of_classes, dense_layer1, dense_layer2)
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


def box_plot(data, name):
    fig = plt.figure(figsize=(9, 6))
    tags = [d[0] for d in data]
    tags = [d if d else 'None' for d in tags]
    values = [d[1] for d in data]
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.set_title(name)
    ax.set_xlabel('$r$ threshold value')
    ax.set_ylabel('Num of positives')
    # Create the boxplot
    bp = ax.boxplot(values)
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ax.set_xticklabels(tags, fontsize=10)
    # Save the figure
    fig.savefig('{}_fourth.png'.format(name), bbox_inches='tight')


def plot_data(file_name):
    with open(file_name) as data_file:
        results = json.load(data_file)
    for rbp in results.keys():
        data = results[rbp]
        box_plot(data, 'RBP{}'.format(rbp))
    exit()


if __name__ == '__main__':
    if 1 == 0:
        with open('data_output4') as data:
            results = json.load(data)
            for rbp in results.keys():
                box_plot(results[rbp], rbp)
        exit()
    start = time.time()
    print('Starting')
    RBPS = [2, 7]
    general_path_cmpt = 'RNCMPT/RBP{}_RNCMPT.sorted'
    args = [[sys.argv[0], general_path_cmpt.format(rbp), rbp] for rbp in RBPS]
    params = [load_files(arg) for arg in args]
    results = {}
    for files, cmpt_seqs, rbp_num in params:
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
        data = []
        dense_layers = [(32, 0), (64, 0), (32, 32), (64, 64)]
        threshold_vals = [None]
        for threshold_val in threshold_vals:
            print('threshold_val:{}'.format(threshold_val))
            for dense1, dense2 in dense_layers:
                print('dense1:{}, dense2:{}'.format(dense1, dense2))
                positives = []
                NUM_TRIALS = 4
                for _ in range(NUM_TRIALS):
                    exists = os.path.isfile(model_path)
                    exists = False
                    if exists:
                        model = keras.models.load_model(model_path)
                    else:
                        model = train_model(files, loss_func, threshold_val=threshold_val, dense_layer1=dense1,
                                            dense_layer2=dense2)

                    x_test = np.array([DataGenerator.one_hot(seq, max_size, kernel_size) for seq in cmpt_seqs])
                    y_test = [int(x) for x in np.append(np.ones(1000), np.zeros(len(x_test) - 1000), axis=0)]
                    y_pred = model.predict(x_test)
                    pos_num = calc_corr(cmpt_seqs, y_test, y_pred, num_of_classes)
                    positives.append(pos_num)
                data.append(((threshold_val, dense1, dense2), positives))
        print(data)
        results[rbp_num] = data
    end = time.time()
    print('took', (end - start) / 60, 'minutes')
    print(results)
    with open('data_output5', 'w') as data_file:
        json.dump(results, data_file)
