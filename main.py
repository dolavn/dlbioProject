import numpy as np
import os
import sys
import keras
from keras.models import Sequential, Model
import time
import json
from itertools import product
from keras.layers import Dense, Dropout, Activation, concatenate, Input, Conv2D
from DataGenerator import DataGenerator
from sklearn.metrics import average_precision_score
from keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt

plt.switch_backend('agg')

PATH = 'RBNS/'
# PATH = './'

IS_BINARY_MODEL = True
USE_SHUFFLED_SEQS = True

'''data parameters'''
valid_p = 0
MAX_SAMPLE_SIZE = 45
FILE_LIMIT = 5000
POSITIVE_FILES_INDEXES = [-1]

'''model parameters'''
KERNEL_SIZES = [6, 8, 10]
NUM_OF_KERNELS = [32, 32, 32]
DENSE_LAYERS = []

'''fit parameters'''
BATCH_SIZE = 264
EPOCHS = 4
workers = 1

OUTPUT = True

dim_func = lambda k_size: (MAX_SAMPLE_SIZE + 2 * k_size - 2, 4, 1)

layers_description = ''.join(['dense{}_{}'.format(i + 1, dense) for i, dense in enumerate(DENSE_LAYERS)])
kernels_description = ''.join(['kernel{}_{}'.format(size, num) for size, num in zip(KERNEL_SIZES, NUM_OF_KERNELS)])
model_path = 'model_samp{}_{}_{}_batch{}_epoch{}_shuf_{}_binary{}'.format(FILE_LIMIT, kernels_description,
                                                                          layers_description,
                                                                          BATCH_SIZE, EPOCHS,
                                                                          USE_SHUFFLED_SEQS,
                                                                          IS_BINARY_MODEL)


''' READ FILES '''


class RBNSreader():
    @staticmethod
    def read_files(rbns_files, file_limit=None):
        lines_from_all_files = []
        print(rbns_files)
        for ind, file in enumerate(rbns_files):
            lines_from_all_files += RBNSreader.read_file(file, ind + 1, file_limit)
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


def read_file_rncmpt(file_path):
    sequences = []
    with open(file_path) as f:
        for line in f:
            sequences.append(line.strip())
    return sequences


def get_sorted_files(file_paths):

    lst = []
    for file in file_paths:

        concentration = file.split('_')[-1].split('nM')[0]
        concentration_val = 0
        if concentration != 'input.seq':
            concentration_val = int(concentration)

        lst.append((file, concentration_val))

    lst.sort(key=lambda x: x[1])
    return lst


def get_files_list(rbp_ind):

    substr = 'RBP' + str(rbp_ind) + '_'
    files = [PATH+file for file in os.listdir(PATH) if substr in file]
    lst = get_sorted_files(files)

    return lst


def load_files(rbp_num):

    cmpt_path = 'RNCMPT/RBP{}_RNCMPT.sorted'.format(rbp_num)
    files = get_files_list(rbp_num)
    cmpt_file = read_file_rncmpt(cmpt_path)
    return files, cmpt_file, rbp_num


def load_files_from_args(args):
    if len(args) < 3:
        raise BaseException("Not enough arguments")
    else:
        cmpt_path = args[1]
        cmpt_file = read_file_rncmpt(cmpt_path)

        files = args[2:]

        rbp_num = int(cmpt_path.split('RBP')[1].split('_')[0])

        return files, cmpt_file, rbp_num


'''DATA PREPROCESS'''


def create_negative_seqs(lines_from_all_files):
    """For each seq create a shuffled version"""
    negative_seqs = []
    for seq, ind in lines_from_all_files:
        seq_list = [ch for ch in seq]
        np.random.shuffle(seq_list)
        negative_seqs.append((''.join(seq_list), 0))
    return negative_seqs


def create_train_valid_data(negative_file, positive_files, num_of_classes, kernel_size=None,
                            custom_file_limit=None):
    """Create generators for train an validation"""

    lines_from_all_files = RBNSreader.read_files(positive_files, file_limit=custom_file_limit)
    if USE_SHUFFLED_SEQS:
        lines_from_all_files += create_negative_seqs(lines_from_all_files)
    else:
        lines_from_all_files += RBNSreader.read_file(negative_file, 0, custom_file_limit * 2)

    np.random.shuffle(lines_from_all_files)

    valid_n = int(valid_p * len(lines_from_all_files))

    validation_data = lines_from_all_files[:valid_n]
    train_data = lines_from_all_files[valid_n:]
    print('validation size', len(validation_data))
    print('train size', len(train_data))
    print('kernel size', kernel_size)
    train_gen = DataGenerator(train_data, num_of_classes=num_of_classes, kernel_sizes=kernel_size,
                              max_sample_size=MAX_SAMPLE_SIZE,
                              batch_size=BATCH_SIZE, shuffle=True)

    valid_gen = None
    if valid_n > 0:
        valid_gen = DataGenerator(validation_data, num_of_classes=num_of_classes, kernel_sizes=kernel_size,
                                  max_sample_size=MAX_SAMPLE_SIZE,
                                  batch_size=BATCH_SIZE, shuffle=True)

    return train_gen, valid_gen


'''MODEL'''


def create_model(num_classes, dense_layers=None, kernel_sizes=None, num_of_kernels=None):
    """Model with different kernel sizes"""
    inputs = [Input(shape=dim_func(kernel_size)) for kernel_size in kernel_sizes]
    kernels = [Conv2D(num_of_kernel, (k_size, 4), strides=(1, 1),
                      padding='valid')(inp) for k_size, num_of_kernel, inp in zip(kernel_sizes, num_of_kernels, inputs)]
    kernels = [Activation('relu')(kernel) for kernel in kernels]
    kernels = [GlobalMaxPooling2D()(kernel) for kernel in kernels]
    func = concatenate(kernels, axis=1)
    for dense_size in dense_layers:
        func = Dense(dense_size, activation='relu', name='dense')(func)
    func = Dense(num_classes, activation='sigmoid', name='output')(func)
    model = Model(inputs, func)
    model.summary()

    return model


def create_and_compile_model(num_of_classes, loss_func, dense_layers=None, kernel_size=None,
                             num_of_kernels=None):
    model = create_model(num_of_classes, dense_layers=dense_layers, kernel_sizes=kernel_size,
                         num_of_kernels=num_of_kernels)
    opt = keras.optimizers.Adam(lr=0.01)

    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


'''TRAIN AND PREDICT'''


def train_model(model, train_gen, valid_gen, epochs):
    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        use_multiprocessing=False,
                        epochs=epochs,
                        workers=workers)
    return model


def train_pipeline(negative_file, positive_files, dense_layer, kernel_size, num_of_kernel, c_file_limit, epochs):
    """create model, datasets, train model. Return the trained model"""
    num_of_classes = len(files)
    loss_func = 'categorical_crossentropy'
    if IS_BINARY_MODEL:
        num_of_classes = 1
        loss_func = 'binary_crossentropy'

    model = create_and_compile_model(num_of_classes, loss_func, dense_layers=dense_layer,
                                     kernel_size=kernel_size, num_of_kernels=num_of_kernel)
    print('training model')
    train_gen, valid_gen = create_train_valid_data(negative_file=negative_file,
                                                   positive_files=positive_files,
                                                   num_of_classes=num_of_classes,
                                                   kernel_size=kernel_size,
                                                   custom_file_limit=c_file_limit)

    model = train_model(model, train_gen, valid_gen, epochs)

    return model


def predict(model, kernel_size, cmpt_seqs):
    """Use model to predict AUPR on the test dataset, and create a sorted RNCMPT file"""

    x_test = []
    for kernel_size_i in kernel_size:
        x_curr = np.array([DataGenerator.one_hot(seq, MAX_SAMPLE_SIZE, kernel_size_i) for seq in cmpt_seqs])
        x_test.append(x_curr)

    y_pred = model.predict(x_test)
    precision, scores = calc_auc(y_pred)

    return precision, scores


'''PLOTS'''


def box_plot(data, name):
    fig = plt.figure(figsize=(9, 6))
    curr_tag = None
    tags = []
    values = []
    data = sorted(data, key=lambda a: a[0])
    for i in range(len(data)):
        if not curr_tag or curr_tag != data[i][0]:
            tags.append(data[i][0])
            values.append([data[i][1]])
            curr_tag = data[i][0]
        else:
            values[-1].append(data[i][1])
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.set_title(name)
    ax.set_xlabel('params')
    ax.set_ylabel('accuracy')
    # Create the boxplot
    print(values)
    print(tags)
    print(data)
    bp = ax.boxplot(values)
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ax.set_xticklabels(tags, fontsize=10, rotation='45')
    # Save the figure
    fig.savefig('{}.png'.format(name), bbox_inches='tight')


def plot_data(results, file_name):
    for rbp in results.keys():
        data = results[rbp]
        box_plot(data, '{}_RBP{}'.format(file_name, rbp))


def plot_scores(scores, rbp_num):
    if not OUTPUT:
        return
    plt.figure(figsize=(20, 10))
    plt.scatter(range(1, len(scores) + 1), scores)
    plt.title('RBP_{}'.format(rbp_num))
    plt.savefig('scores_RBP_{}.png'.format(rbp_num))


def calc_auc(y_pred):
    if IS_BINARY_MODEL:
        y_pred_scores = [y[0] for y in y_pred]
    else:
        y_pred_scores = [np.dot(y, np.concatenate((np.ones(1) * -10, np.array([20, 10])))) for y in y_pred]

    true = [int(x) for x in np.append(np.ones(1000), np.zeros(len(y_pred) - 1000), axis=0)]

    avg_precision = average_precision_score(true, y_pred_scores)
    print('avg_precision', avg_precision)

    return avg_precision, y_pred_scores


def calc_statistics(rbp_list, dense_list, num_of_kernels, kernel_sizes, file_limits, epochs_list, output_file):
    """Try different hyper-parameters"""

    args = [load_files(rbp) for rbp in rbp_list]
    results = {}
    start = time.time()
    NUM_OF_TRIALS = 10
    for _ in range(NUM_OF_TRIALS):
        for files, cmpt_seqs, rbp_num in args:
            files = [file for file, cons in files]
            if rbp_num not in results:
                precision_scores = []
            else:
                precision_scores = results[rbp_num]
            for c_file_limit, dense_layer, num_of_kernel, kernel_size, epochs in product(file_limits,
                                                                                         dense_list,
                                                                                         num_of_kernels, kernel_sizes,
                                                                                         epochs_list):
                print('limit:{} dense:{} num_of_kernel:{} kernel_size:{} epochs:{}'.format(c_file_limit,
                                                                                           dense_layer,
                                                                                           num_of_kernel,
                                                                                           kernel_size,
                                                                                           epochs))
                negative_file = files[0]
                positive_files = files[-2:]
                model = train_pipeline(negative_file, positive_files, dense_layer, kernel_size, num_of_kernel,
                                       c_file_limit, epochs)
                precision = predict(model, kernel_size, cmpt_seqs)
                precision_scores.append(((dense_layer, kernel_size, num_of_kernel, epochs),
                                         precision))
                results[rbp_num] = precision_scores
                with open(output_file, 'w') as data_file:
                    json.dump(results, data_file)
    end = time.time()
    print('took', (end - start) / 60, 'minutes')
    print(results)


def write_aupr(precision):
    if not OUTPUT:
        return
    with open('AUC/RBP_{}.txt'.format(rbp_num), 'w') as f:
        f.write(str(precision))


def load_data(f_names):
    for f_name in f_names:
        with open(f_name) as data_file:
            results = json.load(data_file)
            print(results)
            plot_data(results, f_name)


def output_sorted_rncmpt(cmpt_seqs, scores, rbp_num):
    seqs_with_scores = list(zip(cmpt_seqs, scores))
    seqs_with_scores.sort(key=lambda x: x[1], reverse=True)

    with open('RBP{}_RNCMPT.sorted'.format(rbp_num), 'w') as f:
        for seq, score in seqs_with_scores:
            f.write(seq + '\n')


if __name__ == '__main__':

    first_arg = sys.argv[1]
    if first_arg == '-stats':
        f_name = sys.argv[2]
        calc_statistics(rbp_list=[2, 7], dense_list=[[32]], kernel_sizes=[[32, 32, 32]],
                        num_of_kernels=[[8, 8, 8], [6, 8, 8], [8, 8, 10], [6, 8, 10]], file_limits=[FILE_LIMIT],
                        epochs_list=[2], output_file=f_name)
        exit()
    if first_arg == '-plot':
        f_names = sys.argv[2:]
        load_data(f_names)
        exit()

    start = time.time()

    print('Starting')

    files, cmpt_seqs, rbp_num = load_files_from_args(sys.argv)

    sorted_files = get_sorted_files(files)

    print(sorted_files)

    positive_file_names = [sorted_files[i][0] for i in POSITIVE_FILES_INDEXES]

    negative_file_name = sorted_files[0][0]

    print('RBP', rbp_num)
    new_model_path = model_path + '_rbp' + str(rbp_num) + 'files_' + '_'.join([str(i) for i in POSITIVE_FILES_INDEXES])
    print(new_model_path)

    exists = os.path.isfile(new_model_path)
    if exists:
        print('loading model')
        model = keras.models.load_model(new_model_path)
    else:
        print('training model')
        model = train_pipeline(negative_file_name, positive_file_names, DENSE_LAYERS, KERNEL_SIZES, NUM_OF_KERNELS,
                               FILE_LIMIT, EPOCHS)
        if OUTPUT:
            model.save(new_model_path)

    precision, scores = predict(model, KERNEL_SIZES, cmpt_seqs)

    plot_scores(scores, rbp_num)

    output_sorted_rncmpt(cmpt_seqs, scores, rbp_num)

    write_aupr(precision)

    end = time.time()
    print('took', (end - start) / 60, 'minutes')
