# -*- coding: utf-8 -*-
# @Time        : 2018/7/13 8:59
# @Author      : panxiaotong
# @Description : configuration setting

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', '/root/dssm/data/wb.dat', 'sample files')
flags.DEFINE_integer('batch_size', 100, 'train/test batch size')
flags.DEFINE_float('train_set_ratio', 0.7, 'train set ratio')
flags.DEFINE_string('summaries_dir', '/root/dssm/data/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('negative_size', 60, 'negative size')
flags.DEFINE_integer('epoch_size', 20, "Number of training epoch.")
flags.DEFINE_integer('iteration', 10, "Number of training iteration.")
flags.DEFINE_integer('l1_norm', 400, 'l1 normalization')
flags.DEFINE_integer('l2_norm', 120, 'l2 normalization')
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

cfg = tf.app.flags.FLAGS