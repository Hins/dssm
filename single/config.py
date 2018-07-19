# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 8:59
# @Author      : panxiaotong
# @Description : configuration setting

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 256, 'train/test batch size')
flags.DEFINE_float('train_set_ratio', 0.9, 'train set ratio')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('negative_size', 61, 'negative size')
flags.DEFINE_integer('epoch_size', 20, "Number of training epoch.")
flags.DEFINE_integer('iteration', 10, "Number of training iteration.")
flags.DEFINE_integer('l1_norm', 400, 'l1 normalization')
flags.DEFINE_integer('l2_norm', 120, 'l2 normalization')
flags.DEFINE_string('separator', '###', 'separator')
flags.DEFINE_string('placeholder', 'none_xtpan', 'placeholder')
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

flags.DEFINE_string('summaries_dir', '../data/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_string('raw_text_file', '../data/dssm.dat', 'raw text file, un-segmented')
flags.DEFINE_string('wb_file_path', '../data/wb.dat', 'word break file path')
flags.DEFINE_string('dict_file_path', '../data/dict.dat', 'dictionary file')
flags.DEFINE_string('query_indices_path', '../data/query_indices.dat', 'query indices path')
flags.DEFINE_string('doc_indices_path', '../data/doc_indices.dat', 'doc indices path')
flags.DEFINE_string('train_index_path', '../data/train_index.dat', 'train set index path')
flags.DEFINE_string('dssm_model_path', '../model/dssm', 'dssm model path')
flags.DEFINE_string('train_summary_writer_path', '/dssm/data/train', 'train summary writer path')
flags.DEFINE_string('test_summary_writer_path', '/dssm/data/test', 'test summary writer path')

cfg = tf.app.flags.FLAGS