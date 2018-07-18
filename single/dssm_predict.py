# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 14:23
# @Author      : panxiaotong
# @Description : DSSM model prediction

import jieba
import os
import re
import sys
import tensorflow as tf
import numpy as np
from config import cfg

float_digit_pattern = re.compile(r"-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$")
integ_digit_pattern = re.compile(r"-?[1-9]\d*")

def get_sparse_input(user_query, doc_list, bigram_dict, query_in_shape, doc_in_shape):
    user_word_list = jieba.cut(user_query, cut_all=False)
    user_word_list = [item.encode('utf-8') for item in user_word_list if
                 item.encode('utf-8') not in stopword_dict and
                 item.encode('utf-8').find(" ") == -1 and
                 float_digit_pattern.match(item.encode("utf-8")) == None and
                 integ_digit_pattern.match(item.encode("utf-8")) == None]
    user_word_len = len(user_word_list)
    if user_word_len == 0:
        return -1
    query_indice_list = []    # [[0,1], [0,3], [0,6]]
    query_value_list = []
    for index, word in enumerate(user_word_list):
        if index + 1 < user_word_len:
            key = word + cfg.separator + user_word_list[index + 1]
            if key not in bigram_dict:
                bigram_dict[key] = len(bigram_dict) + 1
            query_indice_list.append([0, bigram_dict[key]])
        else:
            key = word + cfg.separator + cfg.placeholder
            if key not in bigram_dict:
                bigram_dict[key] = len(bigram_dict) + 1
            query_indice_list.append([0, bigram_dict[key]])
        query_value_list.append(1)
    if len(query_indice_list) == 0:
        return -1
    query_in = tf.SparseTensorValue(np.array(query_indice_list, dtype=np.int64),
                                    np.array(query_value_list, dtype=np.float32), query_in_shape)


    section_size = len(doc_list) / cfg.negative_size + 1
    for i in range(section_size):
        if (i + 1) * cfg.negative_size > len(doc_list):
            doc_sec_list = doc_list[i*cfg.negative_size:(i*cfg.negative_size+len(doc_list)%cfg.negative_size)]
        else:
            doc_sec_list = doc_list[i*cfg.negative_size:(i+1)*cfg.negative_size]
        doc_indice_list = []  # [[[0,1], [0,3], [0,6]], [[1,1], [1,2], [1,5]], ...]
        doc_value_list = 1
        for doc_index, doc in enumerate(doc_sec_list):
            doc_word_list = jieba.cut(doc, cut_all=False)
            doc_word_list = [item.encode('utf-8') for item in doc_word_list if
                              item.encode('utf-8') not in stopword_dict and
                              item.encode('utf-8').find(" ") == -1 and
                              float_digit_pattern.match(item.encode("utf-8")) == None and
                              integ_digit_pattern.match(item.encode("utf-8")) == None]
            for index, word in enumerate(doc_word_list):
                if index + 1 < user_word_len:
                    key = word + cfg.separator + doc_word_list[index + 1]
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    doc_indice_list.append([0, doc_index, bigram_dict[key]])
                else:
                    key = word + cfg.separator + cfg.placeholder
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    doc_indice_list.append([0, doc_index, bigram_dict[key]])
                doc_value_list.append(1)
        doc_in = tf.SparseTensorValue(np.array(doc_indice_list, dtype=np.int64),
                                  np.array(doc_value_list, dtype=np.float32), doc_in_shape)

    return query_in, doc_in

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("dssm_predict <model_file> <self-defined dictionary> <stopwords dictionary> <bigram dict file> <test file>")
        sys.exit()

    jieba.load_userdict(sys.argv[2])
    stopword_dict = {}
    with open(sys.argv[3], 'r') as input_file:
        for line in input_file:
            line = line.replace('\r', '').replace('\n', '').strip()
            if line not in stopword_dict:
                stopword_dict[line] = 1
        input_file.close()
    print(len(stopword_dict))

    bigram_dict = {}
    with open(sys.argv[4], 'r') as input_file:
        for line in input_file:
            line = line.replace('\r','').replace('\n','').strip()
            elements = line.split("\t")
            if len(elements) < 2:
                continue
            bigram_dict[elements[0]] = int(elements[1])
        input_file.close()

    query_in_shape = np.array([cfg.batch_size, len(bigram_dict)], np.int64)
    doc_in_shape = np.array([cfg.batch_size, cfg.negative_size, len(bigram_dict)], np.int64)

    user_query_list = []
    docs_list = []

    with open(sys.argv[5], 'r') as input_file:
        for line in input_file:    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
            line = line.replace('\r', '').replace('\n', '').strip()
            elements = line.split("\001")
            user_query_list.append(elements[0])
            doc_list = elements[1].split("\002")
            real_doc_list = []
            for doc in doc_list:
                real_doc_list.append(doc.split("\t")[0])
            docs_list.append(real_doc_list)
        input_file.close()

    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if os.path.exists(sys.argv[1] + ".meta") == True:
            dssm_model = tf.train.import_meta_graph(sys.argv[1] + '.meta')
            dssm_model.restore(sess, sys.argv[1])
            graph = tf.get_default_graph()
            prob = graph.get_tensor_by_name("prob:0")
            query_batch = graph.get_tensor_by_name("QueryBatch:0")
            doc_batch = graph.get_tensor_by_name("DocBatch:0")
            for index,query in enumerate(user_query_list):
                query_in, doc_in = get_sparse_input(query, docs_list[index], bigram_dict, query_in_shape, doc_in_shape)
                real_prob = sess.run(prob, feed_dict={query_batch: query_in, doc_batch: doc_in})
            print(real_prob.shape)
            sys.exit()