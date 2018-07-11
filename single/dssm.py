import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', '/root/dssm/data/wb.dat.10', 'sample files')
flags.DEFINE_float('train_set_ratio', 0.7, 'train set ratio')
flags.DEFINE_string('summaries_dir', '/root/dssm/data/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('negative_size', 20, 'negative size')
flags.DEFINE_integer('max_steps', 900000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 18000, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()

'''
doc_train_data = None
query_train_data = None

# load test data for now
query_test_data = pickle.load(open('../data/query.test.1.pickle', 'rb')).tocsr()
doc_test_data = pickle.load(open('../data/doc.test.1.pickle', 'rb')).tocsr()
'''

placeholder = "none_xtpan"
separator = "###"
bigram_dict = {}
bigram_count = {}

BIGRAM_D = 49284

def load_samples(file_path):
    global BIGRAM_D

    source_samples = []
    target_samples = []
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

    counter = 0
    for line_count, line in enumerate(input_file):    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
        counter += 1
        line = line.replace('\n', '').replace('\r', '')
        elements = line.split('\001')
        if len(elements) < 2:
            continue
        user_query_list = elements[0].split(",")
        user_query_len = len(user_query_list)
        word_index_list = []
        for index,word in enumerate(user_query_list):
            if index + 1 < user_query_len:
                key = word + separator + user_query_list[index + 1]
                #if bigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                word_index_list.append(bigram_dict[key])
            else:
                key = word + separator + placeholder
                #if trigram_count[key] > 5:
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict) + 1
                word_index_list.append(bigram_dict[key])
        if len(word_index_list) == 0:
            continue
        source_samples.append(word_index_list)

        documents = elements[1].split('\002')
        for document in documents:
            sub_elements = document.split('\t')
            document = sub_elements[0].split(",")
            document_len = len(document)
            label = sub_elements[1]

            total_list = []
            word_index_list = []
            for index, word in enumerate(document):
                if index + 2 < document_len:
                    key = word + separator + document[index + 1] + separator + document[index + 2]
                    # if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    word_index_list.append(bigram_dict[key])
                else:
                    key = word + separator + placeholder
                    #if trigram_count[key] > 5:
                    if key not in bigram_dict:
                        bigram_dict[key] = len(bigram_dict) + 1
                    word_index_list.append(bigram_dict[key])
            if len(word_index_list) == 0:
                continue
            if label == "1":
                total_list = [word_index_list] + total_list
            else:
                total_list.append(word_index_list)
        target_samples.append(total_list)
    input_file.close()

    print("bigram_dict length is %d" % len(bigram_dict))
    BIGRAM_D = len(bigram_dict) + 1

    print("line_count is %d, counter is %d" % (line_count, counter))
    user_query_dat = np.zeros(shape=[line_count+1, BIGRAM_D])
    document_dat = np.zeros(shape=[line_count+1, FLAGS.negative_size * BIGRAM_D])    # flat document one-hot data

    for i in xrange(len(source_samples)):
        for item in source_samples[i]:
            if item > user_query_dat.shape[1]:
                print(item)
            user_query_dat[i][item] = 1
    print('source_samples load complete')
    for i in xrange(len(target_samples)):
        for j in xrange(len(target_samples[i])):
            for k in target_samples[i][j]:
                document_dat[i][j*BIGRAM_D+k] = 1
    print('target_samples load complete')

    return (user_query_dat, document_dat, BIGRAM_D)

'''
def load_train_data(path):
    # return load_samples(path)
    global doc_train_data, query_train_data
    doc_train_data = None
    query_train_data = None
    start = time.time()
    doc_train_data = pickle.load(open('../data/doc.train.' + str(pack_idx)+ '.pickle', 'rb')).tocsr()
    query_train_data = pickle.load(open('../data/query.train.'+ str(pack_idx)+ '.pickle', 'rb')).tocsr()
    end = time.time()
    print ("\nTrain data %d/9 is loaded in %.2fs" % (pack_idx, end - start))
'''

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

# NEG = 50
BS = 1000

L1_N = 400
L2_N = 120

query_in_shape = np.array([BS, BIGRAM_D], np.int64)
doc_in_shape = np.array([BS, FLAGS.negative_size, BIGRAM_D], np.int64)

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


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    print("query_batch shape is %s" % query_batch.get_shape())    # [1000, BIGRAM_D]
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')
    print("doc_batch shape is %s" % doc_batch.get_shape())    # [1000, 20, BIGRAM_D]

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (BIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([BIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_batches = tf.sparse_split(sp_input=doc_batch, num_split=FLAGS.negative_size, axis=1)
    doc_l1_batch = []
    for doc in doc_batches:
        doc_l1_batch.append(tf.sparse_tensor_dense_matmul(tf.sparse_reshape(doc, shape=[BS, BIGRAM_D]), weight1) + bias1)
    doc_l1 = tf.reshape(tf.convert_to_tensor(doc_l1_batch), shape=[BS, FLAGS.negative_size, -1])
    print("doc_l1 shape is %s" % doc_l1.get_shape())
    # tf.convert_to_tensor_or_sparse_tensor(tf.squeeze(doc_l1_batch, axis=0))

    query_l1_out = tf.nn.relu(query_l1)
    print("query_l1_out shape is %s" % query_l1_out.get_shape())    # [1000, 400]
    doc_l1_out = tf.nn.relu(doc_l1)
    print("doc_l1_out shape is %s" % doc_l1_out.get_shape())    # [1000, 20, 400]

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    print("query_l2 shape is %s" % query_l2.get_shape())    # [1000, 120]

    doc_batches = tf.split(value=doc_l1_out, num_or_size_splits=FLAGS.negative_size, axis=1)
    doc_l2_batch = []
    for doc in doc_batches:
        doc_l2_batch.append(tf.matmul(tf.squeeze(doc), weight2) + bias2)
    doc_l2 = tf.reshape(tf.convert_to_tensor(doc_l2_batch), shape=[BS, FLAGS.negative_size, -1])
    print("doc_l2 shape is %s" % doc_l2.get_shape()[2])    # [1000, 20, 120]
    query_y = tf.nn.relu(query_l2)
    print("query_y shape is %s" % query_y.get_shape())    # [1000, 120]
    doc_y = tf.nn.relu(doc_l2)
    print("doc_y shape is %s" % doc_y.get_shape())    # [1000, 20, 120]

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_y_tile = tf.tile(query_y, [1, FLAGS.negative_size])    # [1000, 2400], 2400 = 20 * 120
    print("query_y_tile shape is %s" % query_y_tile.get_shape())
    doc_y_concat = tf.reshape(doc_y, shape=[BS, -1])    # [1000, 2400]
    print("doc_y_concat shape is %s" % doc_y_concat.get_shape())
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [1, FLAGS.negative_size])    # [1000, 20]
    print("query_norm shape is %s" % query_norm.get_shape())
    doc_norm = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(doc_y), 2, True)))    # [1000, 20]
    print("doc_norm shape is %s" % doc_norm.get_shape())
    print("tf.multiply(query_y_tile, doc_y_concat) shape is %s" % tf.multiply(query_y_tile, doc_y_concat).get_shape())
    prod = tf.reduce_sum(tf.reshape(tf.multiply(query_y_tile, doc_y_concat), shape=[BS, FLAGS.negative_size, -1]), 2)    # [1000, 20]
    print("prod shape is %s" % prod.get_shape())
    norm_prod = tf.multiply(query_norm, doc_norm)    # [1000, 20]
    print("norm_prod shape is %s" % norm_prod.get_shape())

    cos_sim_raw = tf.truediv(prod, norm_prod)    # [1000, 20]
    print("cos_sim_raw shape is %s" % cos_sim_raw.get_shape())
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.negative_size, BS])) * 20    # 20 is \gamma, [1000, 20]
    print("cos_sim shape is %s" % cos_sim.get_shape())

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))    # [1000, 20]
    print("prob shape is %s" % prob.get_shape())
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])    # [1000, 1]
    print("hit_prob shape is %s" % hit_prob.get_shape())
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
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

def pull_batch(query_data, doc_data, batch_idx):
    # start = time.time()
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(Train, query_data, doc_data, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_data, doc_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_data, doc_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}


config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '~/dssm/data/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '~/dssm/data/test', sess.graph)

    # Actual execution
    start = time.time()
    # fp_time = 0
    # fbp_time = 0

    (query_samples, doc_samples, trigram_dict_size) = load_samples(FLAGS.file_path)
    sample_size = query_samples.shape[0]
    print("sample_size is %d" % sample_size)
    print("int((sample_size / BS * FLAGS.train_set_ratio) * BS) is %d" % int((sample_size / BS * FLAGS.train_set_ratio) * BS))
    r = random.sample(range(sample_size), int((sample_size / BS * FLAGS.train_set_ratio) * BS))
    query_train = query_samples[r]
    print(query_train.shape)
    train_set_size = query_train.shape[0]
    print(train_set_size)
    doc_train = doc_samples[r]
    query_test = query_samples[~r]
    print(query_test.shape)
    doc_test = doc_samples[~r]
    print(doc_test.shape)

    '''
    for step in range(FLAGS.max_steps):
        batch_idx = step % FLAGS.epoch_steps
        #if batch_idx % FLAGS.pack_size == 0:
        #    load_train_data(batch_idx / FLAGS.pack_size + 1)

            # # setup toolbar
            # sys.stdout.write("[%s]" % (" " * toolbar_width))
            # #sys.stdout.flush()
            # sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['


        if batch_idx % (FLAGS.pack_size / 64) == 0:
            progress = 100.0 * batch_idx / FLAGS.epoch_steps
            sys.stdout.write("\r%.2f%% Epoch" % progress)
            sys.stdout.flush()

        # t1 = time.time()
        # sess.run(loss, feed_dict = feed_dict(True, batch_idx))
        # t2 = time.time()
        # fp_time += t2 - t1
        # #print(t2-t1)
        # t1 = time.time()
        sess.run(train_step, feed_dict=feed_dict(True, query_train, doc_train, batch_idx % FLAGS.pack_size))
        # t2 = time.time()
        # fbp_time += t2 - t1
        # #print(t2 - t1)
        # if batch_idx % 2000 == 1999:
        #     print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
        #        (fp_time / step, fbp_time / step))


        if batch_idx == FLAGS.epoch_steps - 1:
            end = time.time()
            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(True, query_train, doc_train, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            # print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
            #        (fp_time / step, fbp_time / step))
            #
            print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                    (step / FLAGS.epoch_steps, epoch_loss, end - start))

            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, query_test, doc_test, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size

            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, step + 1)

            start = time.time()
            print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                   (step / FLAGS.epoch_steps, epoch_loss, start - end))
    '''