# -*- coding: utf-8 -*-
import pickle
import csv
from random import shuffle
import numpy as np

def csv2dicts(csvfile):
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            keys = row
            print(row)
            continue
        if row_index % 5345512 == 0:
            break
        data.append({key: value for key, value in zip(keys, row)})
    return data

def set_nan_as_string(data, replace_str='0'):
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x


train_data = "/mnt/train_new2.csv"
test_data = "/mnt/test_new2.csv"

with open(train_data) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    with open('/mnt/train_data.pickle', 'wb') as f:
        data = csv2dicts(data)
        data = data[::-1]
        pickle.dump(data, f, -1)
        print(data[:3])

with open(test_data) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    with open('/mnt/test_data.pickle', 'wb') as f:
        data = csv2dicts(data)
        pickle.dump(data, f, -1)
        print(data[0])