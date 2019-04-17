import numpy as np
import pandas as pd
import os
PATH = 'RBP1_example1000/'


def one_hot(string):
    dict = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]),
            'C': np.array([0, 0, 1, 0]), 'T': np.array([0, 0, 0, 1]),
            'N': np.array([0.25]*4)}
    vec_list = [dict[c].reshape(1, -1) for c in string]
    return np.concatenate(vec_list, axis=0)


def get_files_list(rbp_ind):
    lst = []
    for file in os.listdir(PATH):
        if file.startswith('RBP' + str(rbp_ind)):
            lst.append(file)
    return lst


def get_x(str_dict):
    x = [entry['x'] for entry in str_dict]
    return x


def get_str_dict(dfs):
    str_dict = {}
    for df_ind, df in enumerate(dfs):
        for seq in df[0]:
            if seq in str_dict:
                str_dict[seq]['y'][df_ind] = 1
            else:
                str_dict[seq] = {'x': one_hot(seq), 'y': [1 if i == df_ind else 0 for i in range(len(dfs))]}
    return list(str_dict.values())


def get_y(str_dict):
    y = [entry['y'] for entry in str_dict]
    return y


def read_file(path):
    return pd.read_csv(path, sep="\t", header=-1)


def sum_vec(arr):
    return np.sum(arr)


l = get_files_list(1)
print(l)
dfs = [read_file(PATH + file) for file in l]
str_dict = get_str_dict(dfs)
x = get_x(str_dict)
y = get_y(str_dict)