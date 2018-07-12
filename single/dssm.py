import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', '/root/dssm/data/wb.dat', 'sample files')
flags.DEFINE_integer('batch_size', 10, 'train/test batch size')
flags.DEFINE_float('train_set_ratio', 0.7, 'train set ratio')
flags.DEFINE_string('summaries_dir', '/root/dssm/data/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('negative_size', 20, 'negative size')
flags.DEFINE_integer('epoch_size', 5, "Number of training epoch.")
flags.DEFINE_integer('iteration', 10, "Number of training iteration.")
flags.DEFINE_integer('l1_norm', 400, 'l1 normalization')
flags.DEFINE_integer('l2_norm', 120, 'l2 normalization')
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()

placeholder = "none_xtpan"
separator = "###"
bigram_dict = {}
bigram_count = {}

BIGRAM_D = 49284

def load_samples(file_path):
    global BIGRAM_D

    input_file = open(file_path, 'r')

    # calculate trigram count
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

    BIGRAM_D = len(bigram_dict) + 1
    print("BIGRAM_D is %d" % BIGRAM_D)

    sample_size = (line_index + 1) / FLAGS.batch_size
    train_index = random.sample(range(sample_size), int(sample_size * FLAGS.train_set_ratio))
    test_index = np.setdiff1d(range(sample_size), train_index)

    return (user_indices, user_values, doc_indices, doc_values, train_index, test_index)

(user_indices, user_values, doc_indices, doc_values, train_index_list, test_index_list) = load_samples(FLAGS.file_path)

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

query_in_shape = np.array([FLAGS.batch_size, BIGRAM_D], np.int64)
doc_in_shape = np.array([FLAGS.batch_size, FLAGS.negative_size, BIGRAM_D], np.int64)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

with tf.device('/gpu:0'):
    with tf.name_scope('input'):
        # Shape [FLAGS.batch_size, TRIGRAM_D].
        query_batch = tf.sparse_placeholder(tf.float32, shape=[None, BIGRAM_D], name='QueryBatch')
        print("query_batch shape is %s" % query_batch.get_shape())    # [1000, BIGRAM_D]
        # Shape [FLAGS.batch_size, TRIGRAM_D]
        doc_batch = tf.sparse_placeholder(tf.float32, shape=[None, FLAGS.negative_size, BIGRAM_D], name='DocBatch')
        print("doc_batch shape is %s" % doc_batch.get_shape())    # [1000, 20, BIGRAM_D]

    with tf.name_scope('L1'):
        l1_par_range = np.sqrt(6.0 / (BIGRAM_D + FLAGS.l1_norm))
        weight1 = tf.Variable(tf.random_uniform([BIGRAM_D, FLAGS.l1_norm], -l1_par_range, l1_par_range))
        bias1 = tf.Variable(tf.random_uniform([FLAGS.l1_norm], -l1_par_range, l1_par_range))
        variable_summaries(weight1, 'L1_weights')
        variable_summaries(bias1, 'L1_biases')

        # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
        query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
        # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
        doc_batches = tf.sparse_split(sp_input=doc_batch, num_split=FLAGS.negative_size, axis=1)
        doc_l1_batch = []
        for doc in doc_batches:
            doc_l1_batch.append(tf.sparse_tensor_dense_matmul(tf.sparse_reshape(doc, shape=[FLAGS.batch_size, BIGRAM_D]), weight1) + bias1)
        doc_l1 = tf.reshape(tf.convert_to_tensor(doc_l1_batch), shape=[FLAGS.batch_size, FLAGS.negative_size, -1])
        print("doc_l1 shape is %s" % doc_l1.get_shape())
        # tf.convert_to_tensor_or_sparse_tensor(tf.squeeze(doc_l1_batch, axis=0))

        query_l1_out = tf.nn.relu(query_l1)
        print("query_l1_out shape is %s" % query_l1_out.get_shape())    # [1000, 400]
        doc_l1_out = tf.nn.relu(doc_l1)
        print("doc_l1_out shape is %s" % doc_l1_out.get_shape())    # [1000, 20, 400]

    with tf.name_scope('L2'):
        l2_par_range = np.sqrt(6.0 / (FLAGS.l1_norm + FLAGS.l2_norm))
        weight2 = tf.Variable(tf.random_uniform([FLAGS.l1_norm, FLAGS.l2_norm], -l2_par_range, l2_par_range))
        bias2 = tf.Variable(tf.random_uniform([FLAGS.l2_norm], -l2_par_range, l2_par_range))
        variable_summaries(weight2, 'L2_weights')
        variable_summaries(bias2, 'L2_biases')

        query_l2 = tf.matmul(query_l1_out, weight2) + bias2
        print("query_l2 shape is %s" % query_l2.get_shape())    # [1000, 120]

        doc_batches = tf.split(value=doc_l1_out, num_or_size_splits=FLAGS.negative_size, axis=1)
        doc_l2_batch = []
        for doc in doc_batches:
            doc_l2_batch.append(tf.matmul(tf.squeeze(doc), weight2) + bias2)
        doc_l2 = tf.reshape(tf.convert_to_tensor(doc_l2_batch), shape=[FLAGS.batch_size, FLAGS.negative_size, -1])
        print("doc_l2 shape is %s" % doc_l2.get_shape()[2])    # [1000, 20, 120]
        query_y = tf.nn.relu(query_l2)
        print("query_y shape is %s" % query_y.get_shape())    # [1000, 120]
        doc_y = tf.nn.relu(doc_l2)
        print("doc_y shape is %s" % doc_y.get_shape())    # [1000, 20, 120]

    with tf.name_scope('Cosine_Similarity'):
        # Cosine similarity
        query_y_tile = tf.tile(query_y, [1, FLAGS.negative_size])    # [1000, 2400], 2400 = 20 * 120
        print("query_y_tile shape is %s" % query_y_tile.get_shape())
        doc_y_concat = tf.reshape(doc_y, shape=[FLAGS.batch_size, -1])    # [1000, 2400]
        print("doc_y_concat shape is %s" % doc_y_concat.get_shape())
        query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [1, FLAGS.negative_size])    # [1000, 20]
        print("query_norm shape is %s" % query_norm.get_shape())
        doc_norm = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(doc_y), 2, True)))    # [1000, 20]
        print("doc_norm shape is %s" % doc_norm.get_shape())
        print("tf.multiply(query_y_tile, doc_y_concat) shape is %s" % tf.multiply(query_y_tile, doc_y_concat).get_shape())
        prod = tf.reduce_sum(tf.reshape(tf.multiply(query_y_tile, doc_y_concat), shape=[FLAGS.batch_size, FLAGS.negative_size, -1]), 2)    # [1000, 20]
        print("prod shape is %s" % prod.get_shape())
        norm_prod = tf.multiply(query_norm, doc_norm)    # [1000, 20]
        print("norm_prod shape is %s" % norm_prod.get_shape())

        cos_sim_raw = tf.truediv(prod, norm_prod)    # [1000, 20]
        print("cos_sim_raw shape is %s" % cos_sim_raw.get_shape())
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.negative_size, FLAGS.batch_size])) * 20    # 20 is \gamma, [1000, 20]
        print("cos_sim shape is %s" % cos_sim.get_shape())

    with tf.name_scope('Loss'):
        # Train Loss
        prob = tf.nn.softmax((cos_sim))    # [1000, 20]
        print("prob shape is %s" % prob.get_shape())
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])    # [1000, 1]
        print("hit_prob shape is %s" % hit_prob.get_shape())
        loss = -tf.reduce_sum(tf.log(hit_prob)) / FLAGS.batch_size
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Training'):
        # Optimizer
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

def pull_batch(batch_idx):
    # start = time.time()
    lower_bound = batch_idx * FLAGS.batch_size
    upper_bound = (batch_idx + 1) * FLAGS.batch_size
    batch_indice_list = []
    batch_value_list = []
    for index, item in enumerate(user_indices):
        if item[0] >= lower_bound and item[0] < upper_bound:
            offset_item = item[:]
            offset_item[0] %= FLAGS.batch_size
            batch_indice_list.append(offset_item)
            batch_value_list.append(user_values[index])
    query_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                    np.array(batch_value_list, dtype=np.float32), query_in_shape)

    batch_indice_list = []
    batch_value_list = []
    for index, item in enumerate(doc_indices):
        if item[0] >= lower_bound and item[0] < upper_bound:
            offset_item = item[:]
            offset_item[0] %= FLAGS.batch_size
            batch_indice_list.append(offset_item)
            batch_value_list.append(doc_values[index])
    doc_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                  np.array(batch_value_list, dtype=np.float32), doc_in_shape)

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    query_in, doc_in = pull_batch(batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}

config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '~/dssm/data/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '~/dssm/data/test', sess.graph)

    for epoch_step in range(FLAGS.epoch_size):
        epoch_loss = 0.0
        for iter in range(FLAGS.iteration):
            train_idx = iter % (len(train_index_list) + len(test_index_list))
            if train_idx in train_index_list:
                _, iter_loss = sess.run([train_step, loss],
                         feed_dict=feed_dict(train_idx))
                print("epoch %d : iteration %d, loss is %f" % (epoch_step, iter, iter_loss))
                epoch_loss += iter_loss
        train_writer.add_summary(epoch_loss / len(train_index_list), epoch_step + 1)

        epoch_accuracy = 0.0
        for iter in range(FLAGS.iteration):
            test_idx = iter % (len(train_index_list) + len(test_index_list))
            if test_idx in test_index_list:
                iter_accuracy = sess.run(accuracy, feed_dict=feed_dict(test_idx))
                print("epoch %d : iteration %d, accuracy is %f" % (epoch_step, iter, iter_accuracy))
                epoch_accuracy += iter_accuracy
        test_writer.add_summary(epoch_accuracy / len(test_index_list), epoch_step + 1)