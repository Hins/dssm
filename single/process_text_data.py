# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 8:56
# @Author      : panxiaotong
# @Description : 通用函数库

import sys
import random
import numpy as np
from config import cfg

if __name__ == "__main__":
    bigram_dict = {}
    bigram_count = {}
    input_file = open(cfg.wb_file_path, 'r')
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
                key = word + cfg.separator + user_query_list[index + 1]
                if key not in bigram_count:
                    bigram_count[key] = 1
                else:
                    bigram_count[key] += 1
            else:
                key = word + cfg.separator + cfg.placeholder
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
                    key = word + cfg.separator + document[index + 1]
                    if key not in bigram_count:
                        bigram_count[key] = 1
                    else:
                        bigram_count[key] += 1
                else:
                    key = word + cfg.separator + cfg.placeholder
                    if key not in bigram_count:
                        bigram_count[key] = 1
                    else:
                        bigram_count[key] += 1
    input_file.seek(0)

    print("calculate bigram count complete")

    user_indices = []
    doc_indices = []
    for line_index, line in enumerate(input_file):    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
        line = line.replace('\n', '').replace('\r', '')
        elements = line.split('\001')
        if len(elements) < 2:
            continue
        user_query_list = elements[0].split(",")
        user_query_len = len(user_query_list)
        query_indice_list = []
        for index,word in enumerate(user_query_list):
            if index + 1 < user_query_len:
                key = word + cfg.separator + user_query_list[index + 1]
                #if bigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                query_indice_list.append([line_index, bigram_dict[key]])
            else:
                key = word + cfg.separator + cfg.placeholder
                #if trigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                query_indice_list.append([line_index, bigram_dict[key]])
        if len(query_indice_list) == 0:
            continue

        documents = elements[1].split('\002')
        flag = True
        doc_indice_list = []
        for document in documents:
            sub_elements = document.split('\t')
            document = sub_elements[0].split(",")
            document_len = len(document)

            tmp_doc_indice_list = []
            for index, word in enumerate(document):
                if index + 1 < document_len:
                    key = word + cfg.separator + document[index + 1]
                    #if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    tmp_doc_indice_list.append([line_index, index, bigram_dict[key]])
                else:
                    key = word + cfg.separator + cfg.placeholder
                    #if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    tmp_doc_indice_list.append([line_index, index, bigram_dict[key]])
            if len(tmp_doc_indice_list) == 0:
                flag = False
                break
            doc_indice_list.append(tmp_doc_indice_list)
        if flag == True:
            user_indices.append(query_indice_list)
            doc_indices.append(doc_indice_list)
    input_file.close()

    output_file = open(cfg.dict_file_path, 'w')
    for k,v in bigram_dict.items():
        try:
            output_file.write(k.decode('utf-8') + "\t" + str(v) + "\n")
        except:
            output_file.write(k + "\t" + str(v) + "\n")
    output_file.close()

    bigram_dict_size = len(bigram_dict) + 1
    print("bigram_dict_size is %d" % bigram_dict_size)

    user_indices_output_file = open(cfg.query_indices_path, 'w')
    for item in user_indices:
        user_indices_output_file.write("\001".join(str(sub_item[0]) + "\002" + str(sub_item[1]) for sub_item in item) + "\n")
    user_indices_output_file.close()

    doc_indices_output_file = open(cfg.doc_indices_path, 'w')
    for docs in doc_indices:   # item meant 20 docs
        doc_list = []
        for doc in docs:
            doc_list.append("\002".join(str(indice[0]) + "\003" + str(indice[1]) for indice in doc))
        doc_indices_output_file.write("\001".join(doc_list) + "\n")
    doc_indices_output_file.close()

    sample_size = (line_index + 1) / cfg.batch_size
    print("sample_size is %d" % sample_size)
    train_index = random.sample(range(sample_size), int(sample_size * cfg.train_set_ratio)).astype(np.int32)
    np.savetxt(cfg.train_index_path, train_index, delimiter=",")