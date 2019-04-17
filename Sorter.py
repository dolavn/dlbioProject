import random

path = 'RNCMPT_training/'


def read_file(file_path):
    sequences = []
    f = open(file_path)
    for line in f:
        sequences.append(line)
    return sequences


def sort_sequences(sequences, func):
    sequences = sorted(sequences, key=lambda x: func(x))
    return sequences


seq = read_file(path + 'RBP1_RNCMPT.sorted')
print(seq[0])
seq = sort_sequences(seq, lambda a: random.randint(1, 21)*5)
print(seq[0])
