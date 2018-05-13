import numpy as np
import re
import itertools
from collections import Counter
import json
import jieba

q_field = "question"

def segment_chinese_text(json_list,field):
    q_list = []
    for item in json_list:
        q = item[field]
        if q:
            q_seg = jieba.cut(q, cut_all=True)
            q_seg_list = " ".join(q_seg)
            q_list.append(q_seg_list)
    return q_list

def generate_label(json_list, field, ignore_field,label_json=None):
    l_list = []
    labels = {}
    if label_json:
        labels = label_json
    else:
        idx = 0
        for item in json_list:
            title = item[field]
            if title not in labels:
                labels[title] = idx
                idx+=1
    l_len = len(labels)
    for item in json_list:
        q = item[ignore_field]
        if q:
            title = item[field]
            l_idx = labels[title]
            l_tmp = [0 for i in range(l_len)]
            l_tmp[l_idx] = 1
            l_list.append(l_tmp)

    return l_list, labels

def load_data_and_labels(JDQnA_path,label_field,label_json=None):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open(JDQnA_path) as data_file:
        JDQnA_json = json.load(data_file)
    # Split by words
    x_text = segment_chinese_text(JDQnA_json,q_field)
    # Generate labels
    y, labels = generate_label(JDQnA_json,label_field,q_field,label_json)
    y = np.array(y)
    return [x_text, y, labels]

def segment_answer_chinese_text(json_list,field):
    a_list = []
    for item in json_list:
        q = item["question"]
        if q:
            a = item[field]
            a_list_tmp = " ".join(a)
            a_seg = jieba.cut(a_list_tmp, cut_all=True)
            a_seg_list = " ".join(a_seg)
            a_list.append(a_seg_list)
    return a_list

def load_answer_data_and_labels(JDQnA_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open(JDQnA_path) as data_file:
        JDQnA_json = json.load(data_file)
    # Split by words
    x_text = segment_answer_chinese_text(JDQnA_json,"answer")

    return x_text

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
