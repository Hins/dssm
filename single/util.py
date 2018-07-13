# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 8:56
# @Author      : panxiaotong
# @Description : 通用函数库

import random
import numpy as np
from config import cfg

def load_samples(file_path):
    placeholder = "none_xtpan"
    separator = "###"

    input_file = open(file_path, 'r')

    bigram_dict = {}
    bigram_count = {}

    # calculate bigram count
    for line in input_file:    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
        line = line.replace('\n', '').replace('\r', '')
        elements = line.split('\001')
        if len(elements) < 2:
            continue
        user_query_list = elements[0].split(",")
        user_query_len = len(user_query_list)
        for index,word in enumerate(user_query_list):
            if index + 1 < user_query_len:
                key = word + separator + user_query_list[index + 1]
                if key not in bigram_count:
                    bigram_count[key] = 1
                else:
                    bigram_count[key] += 1
            else:
                key = word + separator + placeholder
                if key not in bigram_count:
                    bigram_count[key] = 1
                else:
                    bigram_count[key] += 1

        documents = elements[1].split('\002')
        for document in documents:
            sub_elements = document.split('\t')
            document = sub_elements[0].split(",")
            document_len = len(document)

            for index, word in enumerate(document):
                if index + 1 < document_len:
                    key = word + separator + document[index + 1]
                    if key not in bigram_count:
                        bigram_count[key] = 1
                    else:
                        bigram_count[key] += 1
                else:
                    key = word + separator + placeholder
                    if key not in bigram_count:
                        bigram_count[key] = 1
                    else:
                        bigram_count[key] += 1
    input_file.seek(0)

    print("calculate bigram count complete")

    user_indices = []
    user_values = []
    doc_indices = []
    doc_values = []
    for line_index, line in enumerate(input_file):    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
        line = line.replace('\n', '').replace('\r', '')
        elements = line.split('\001')
        if len(elements) < 2:
            continue
        user_query_list = elements[0].split(",")
        user_query_len = len(user_query_list)
        query_indice_list = []
        query_value_list = []
        for index,word in enumerate(user_query_list):
            if index + 1 < user_query_len:
                key = word + separator + user_query_list[index + 1]
                #if bigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                query_indice_list.append([line_index, bigram_dict[key]])
                query_value_list.append(1.0)
            else:
                key = word + separator + placeholder
                #if trigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                query_indice_list.append([line_index, bigram_dict[key]])
                query_value_list.append(1.0)
        if len(query_indice_list) == 0:
            continue

        documents = elements[1].split('\002')
        flag = True
        doc_indice_list = []
        doc_value_list = []
        for document in documents:
            sub_elements = document.split('\t')
            document = sub_elements[0].split(",")
            document_len = len(document)

            prev_size = len(doc_indice_list)
            for index, word in enumerate(document):
                if index + 2 < document_len:
                    key = word + separator + document[index + 1] + separator + document[index + 2]
                    #if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    doc_indice_list.append([line_index, index, bigram_dict[key]])
                    doc_value_list.append(1.0)
                else:
                    key = word + separator + placeholder
                    #if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    doc_indice_list.append([line_index, index, bigram_dict[key]])
                    doc_value_list.append(1.0)
            if prev_size == len(doc_indice_list):
                flag = False
                break
        if flag == True:
            user_indices.extend(query_indice_list)
            user_values.extend(query_value_list)
            doc_indices.extend(doc_indice_list)
            doc_values.extend(doc_value_list)

    input_file.close()

    bigram_dict_size = len(bigram_dict) + 1
    print("bigram_dict_size is %d" % bigram_dict_size)

    sample_size = (line_index + 1) / cfg.batch_size
    print("sample_size is %d" % sample_size)
    train_index = random.sample(range(sample_size), int(sample_size * cfg.train_set_ratio))
    test_index = np.setdiff1d(range(sample_size), train_index)

    return (user_indices, user_values, doc_indices, doc_values, train_index, test_index, bigram_dict_size, sample_size)