# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 14:23
# @Author      : panxiaotong
# @Description : DSSM model prediction

import os
import sys
import tensorflow as tf
import numpy as np
from config import cfg

def pull_batch(self, batch_idx):
    lower_bound = batch_idx * cfg.batch_size
    upper_bound = (batch_idx + 1) * cfg.batch_size
    batch_indice_list = []
    batch_value_list = []
    for index, item in enumerate(user_indices):
        if item[0] >= lower_bound and item[0] < upper_bound:
            offset_item = item[:]
            offset_item[0] %= cfg.batch_size
            batch_indice_list.append(offset_item)
            batch_value_list.append(user_values[index])
    query_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                    np.array(batch_value_list, dtype=np.float32), self.query_in_shape)

    batch_indice_list = []
    batch_value_list = []
    for index, item in enumerate(doc_indices):
        if item[0] >= lower_bound and item[0] < upper_bound:
            offset_item = item[:]
            offset_item[0] %= cfg.batch_size
            batch_indice_list.append(offset_item)
            batch_value_list.append(doc_values[index])
    doc_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                  np.array(batch_value_list, dtype=np.float32), self.doc_in_shape)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("dssm_predict <model_file> <dict file> <test file>")
        sys.exit()

    bigram_dict = {}
    with open(sys.argv[2], 'r') as input_file:
        for line in input_file:
            line = line.replace('\r','').replace('\n','').strip()
            elements = line.split("\t")
            if len(elements) < 2:
                continue
            bigram_dict[elements[0]] = int(elements[1])
        input_file.close()

    query_in_shape = np.array([cfg.batch_size, len(bigram_dict)], np.int64)
    doc_in_shape = np.array([cfg.batch_size, cfg.negative_size, len(bigram_dict)], np.int64)

    user_query_indices = []
    user_query_values = []
    doc_indices = []
    doc_values = []

    with open(sys.argv[3], 'r') as input_file:
        for line in input_file:    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
            line = line.replace('\r', '').replace('\n', '').strip()
            elements = line.split("\001")
            user_query =
        input_file.close()


    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if os.path.exists(sys.argv[1] + ".meta") == True:
            dssm_model = tf.train.import_meta_graph(sys.argv[1] + '.meta')
            dssm_model.restore(sess, sys.argv[1])
            graph = tf.get_default_graph()
            prob = graph.get_tensor_by_name("prob:0")
            query_batch = graph.get_tensor_by_name("query_batch:0")
            doc_batch = graph.get_tensor_by_name("doc_batch:0")
            real_prob = sess.run(prob, feed_dict={query_batch: , doc_batch: })
            print(real_prob.shape)
            sys.exit()