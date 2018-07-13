# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 14:23
# @Author      : panxiaotong
# @Description : DSSM model prediction

import os
import sys
import tensorflow as tf

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("dssm_predict <model_file> <dict file> <test file>")
        sys.exit()

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