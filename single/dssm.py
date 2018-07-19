import random
import time
import os
import sys
import numpy as np
import tensorflow as tf

from config import cfg

class DSSM:
    def __init__(self, sess, dict_size, output_file):
        self.sess = sess
        self.dict_size = dict_size
        self.output_file = output_file
        self.query_in_shape = np.array([cfg.batch_size, dict_size], np.int64)
        self.doc_in_shape = np.array([cfg.batch_size, cfg.negative_size, dict_size], np.int64)

        with tf.device('/gpu:0'):
            with tf.name_scope('input'):
                # Shape [cfg.batch_size, TRIGRAM_D].
                self.query_batch = tf.sparse_placeholder(tf.float32, shape=[None, self.dict_size], name='QueryBatch')
                print("query_batch shape is %s" % self.query_batch.get_shape())    # [1000, BIGRAM_D]
                # Shape [cfg.batch_size, TRIGRAM_D]
                self.doc_batch = tf.sparse_placeholder(tf.float32, shape=[None, cfg.negative_size, self.dict_size], name='DocBatch')
                print("doc_batch shape is %s" % self.doc_batch.get_shape())    # [1000, 20, BIGRAM_D]

        with tf.name_scope('L1'):
            l1_par_range = np.sqrt(6.0 / (self.dict_size + cfg.l1_norm))
            weight1 = tf.Variable(tf.random_uniform([self.dict_size, cfg.l1_norm], -l1_par_range, l1_par_range))
            bias1 = tf.Variable(tf.random_uniform([cfg.l1_norm], -l1_par_range, l1_par_range))
            self.variable_summaries(weight1, 'L1_weights')
            self.variable_summaries(bias1, 'L1_biases')

            # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
            query_l1 = tf.sparse_tensor_dense_matmul(self.query_batch, weight1) + bias1
            # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
            doc_batches = tf.sparse_split(sp_input=self.doc_batch, num_split=cfg.negative_size, axis=1)
            doc_l1_batch = []
            for doc in doc_batches:
                doc_l1_batch.append(tf.sparse_tensor_dense_matmul(tf.sparse_reshape(doc, shape=[cfg.batch_size, self.dict_size]), weight1) + bias1)
            doc_l1 = tf.reshape(tf.convert_to_tensor(doc_l1_batch), shape=[cfg.batch_size, cfg.negative_size, -1])
            print("doc_l1 shape is %s" % doc_l1.get_shape())
            # tf.convert_to_tensor_or_sparse_tensor(tf.squeeze(doc_l1_batch, axis=0))

            query_l1_out = tf.nn.relu(query_l1)
            print("query_l1_out shape is %s" % query_l1_out.get_shape())    # [1000, 400]
            doc_l1_out = tf.nn.relu(doc_l1)
            print("doc_l1_out shape is %s" % doc_l1_out.get_shape())    # [1000, 20, 400]

        with tf.name_scope('L2'):
            l2_par_range = np.sqrt(6.0 / (cfg.l1_norm + cfg.l2_norm))
            weight2 = tf.Variable(tf.random_uniform([cfg.l1_norm, cfg.l2_norm], -l2_par_range, l2_par_range))
            bias2 = tf.Variable(tf.random_uniform([cfg.l2_norm], -l2_par_range, l2_par_range))
            self.variable_summaries(weight2, 'L2_weights')
            self.variable_summaries(bias2, 'L2_biases')

            query_l2 = tf.matmul(query_l1_out, weight2) + bias2
            print("query_l2 shape is %s" % query_l2.get_shape())    # [1000, 120]

            doc_batches = tf.split(value=doc_l1_out, num_or_size_splits=cfg.negative_size, axis=1)
            doc_l2_batch = []
            for doc in doc_batches:
                doc_l2_batch.append(tf.matmul(tf.squeeze(doc), weight2) + bias2)
            doc_l2 = tf.reshape(tf.convert_to_tensor(doc_l2_batch), shape=[cfg.batch_size, cfg.negative_size, -1])
            print("doc_l2 shape is %s" % doc_l2.get_shape()[2])    # [1000, 20, 120]
            query_y = tf.nn.relu(query_l2)
            print("query_y shape is %s" % query_y.get_shape())    # [1000, 120]
            doc_y = tf.nn.relu(doc_l2)
            print("doc_y shape is %s" % doc_y.get_shape())    # [1000, 20, 120]

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            query_y_tile = tf.tile(query_y, [1, cfg.negative_size])    # [1000, 2400], 2400 = 20 * 120
            print("query_y_tile shape is %s" % query_y_tile.get_shape())
            doc_y_concat = tf.reshape(doc_y, shape=[cfg.batch_size, -1])    # [1000, 2400]
            print("doc_y_concat shape is %s" % doc_y_concat.get_shape())
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [1, cfg.negative_size])    # [1000, 20]
            print("query_norm shape is %s" % query_norm.get_shape())
            doc_norm = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(doc_y), 2, True)))    # [1000, 20]
            print("doc_norm shape is %s" % doc_norm.get_shape())
            print("tf.multiply(query_y_tile, doc_y_concat) shape is %s" % tf.multiply(query_y_tile, doc_y_concat).get_shape())
            prod = tf.reduce_sum(tf.reshape(tf.multiply(query_y_tile, doc_y_concat), shape=[cfg.batch_size, cfg.negative_size, -1]), 2)    # [1000, 20]
            print("prod shape is %s" % prod.get_shape())
            norm_prod = tf.multiply(query_norm, doc_norm)    # [1000, 20]
            print("norm_prod shape is %s" % norm_prod.get_shape())

            cos_sim_raw = tf.truediv(prod, norm_prod)    # [1000, 20]
            print("cos_sim_raw shape is %s" % cos_sim_raw.get_shape())
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [cfg.negative_size, cfg.batch_size])) * 20    # 20 is \gamma, [1000, 20]
            print("cos_sim shape is %s" % cos_sim.get_shape())

        with tf.name_scope('Loss'):
            # Train Loss
            self.prob = tf.nn.softmax((cos_sim), name="prob")    # [1000, 20]
            print("prob shape is %s" % self.prob.get_shape())
            hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])    # [1000, 1]
            print("hit_prob shape is %s" % hit_prob.get_shape())
            self.loss = -tf.reduce_sum(tf.log(hit_prob)) / cfg.batch_size
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('Training'):
            # Optimizer
            self.train_step = tf.train.GradientDescentOptimizer(cfg.learning_rate).minimize(self.loss)

        self.merged = tf.summary.merge_all()
        self.model = tf.train.Saver()

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.prob, 1), 0)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('Test'):
            self.average_accuracy = tf.placeholder(tf.float32)
            self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

        with tf.name_scope('Train'):
            self.average_loss = tf.placeholder(tf.float32)
            self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def variable_summaries(self, var, name):
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

    def pull_batch(self, batch_idx):
        lower_bound = batch_idx * cfg.batch_size
        upper_bound = (batch_idx + 1) * cfg.batch_size
        batch_indice_list = []
        batch_value_list = []
        for item in user_indices:
            if item[0][0] >= lower_bound and item[0][0] < upper_bound:
                for sub_item in item:
                    offset_item = sub_item[:]
                    offset_item[0] %= cfg.batch_size
                    batch_indice_list.append(offset_item)
                    batch_value_list.append(1)
        query_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                        np.array(batch_value_list, dtype=np.float32), self.query_in_shape)

        batch_indice_list = []
        batch_value_list = []
        for doc in doc_indices:
            if doc[0][0][0] >= lower_bound and doc[0][0][0] < upper_bound:
                for indices in doc:
                    for indice in indices:
                        offset_item = indice[:]
                        offset_item[0] %= cfg.batch_size
                        batch_indice_list.append(offset_item)
                        batch_value_list.append(1)
        doc_in = tf.SparseTensorValue(np.array(batch_indice_list, dtype=np.int64),
                                      np.array(batch_value_list, dtype=np.float32), self.doc_in_shape)

        return query_in, doc_in

    def feed_dict(self, batch_idx):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        query_in, doc_in = self.pull_batch(batch_idx)
        return {self.query_batch: query_in, self.doc_batch: doc_in}

    def train(self, train_idx, print_var=False):
        if print_var == True:
            return self.sess.run([self.train_step, self.merged, self.loss], feed_dict=self.feed_dict(train_idx))
        else:
            return self.sess.run([self.train_step, self.loss], feed_dict=self.feed_dict(train_idx))[1]

    def validate(self, test_idx):
        return self.sess.run(self.accuracy, feed_dict=self.feed_dict(test_idx))

    def predict(self, idx):
        return self.sess.run(self.prob, feed_dict=self.feed_dict(idx))

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

    def save(self):
        self.model.save(sess, self.output_file)

if __name__ == "__main__":
    start = time.time()
    with open(cfg.wb_file_path, 'r') as input_file:
        for index, line in enumerate(input_file):
            pass
        sample_size = index + 1
        input_file.close()
    print("batch: %d" % (sample_size / cfg.batch_size))

    words_dict = {}
    with open(cfg.dict_file_path, 'r') as input_file:
        for line in input_file:
            line = line.replace('\r','').replace('\n','').strip()
            elements = line.split("\t")
            if len(elements) < 2:
                continue
            words_dict[elements[0]] = int(elements[1])
        input_file.close()
    bigram_dict_size = len(words_dict)
    print("bigram_dict_size length is %d" % bigram_dict_size)

    user_indices = []
    with open(cfg.query_indices_path, 'r') as input_file:
        for line in input_file:
            line = line.replace('\r', '').replace('\n', '').strip()
            elements = line.split("\001")
            indices = []
            for element in elements:
                indices.append([int(indice) for indice in element.split("\002")])
            user_indices.append(indices)
        input_file.close()

    doc_indices = []
    with open(cfg.doc_indices_path, 'r') as input_file:
        for line in input_file:
            line = line.replace('\r', '').replace('\n', '').strip()
            elements = line.split("\001")
            docs = []
            for doc in elements:
                indices_array = doc.split("\002")
                indices_list = []
                for indice in indices_array:
                    indices_list.append([int(sub_indice) for sub_indice in indice.split("\003")])
                docs.append(indices_list)
            doc_indices.append(docs)
        input_file.close()

    train_index_list = np.loadtxt(cfg.train_index_path, dtype=float).astype(int)
    end = time.time()
    print("Loading data from HDD to memory: %.2fs" % (end - start))

    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    #if not cfg.gpu:
    #config = tf.ConfigProto(device_count= {'GPU' : 0})

    with tf.Session(config=config) as sess:
        #sess.run(tf.global_variables_initializer())
        dssm_obj = DSSM(sess, bigram_dict_size, cfg.dssm_model_path)
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.test_summary_writer_path, sess.graph)

        # load previous model to predict
        if os.path.exists(cfg.dssm_model_path + ".meta") == True:
            dssm_model = tf.train.import_meta_graph(cfg.dssm_model_path + '.meta')
            dssm_model.restore(sess, cfg.dssm_model_path)
            for epoch_step in range(cfg.epoch_size):
                epoch_accuracy = 0.0
                for iter in range(cfg.iteration):
                    test_idx = iter % (sample_size / cfg.batch_size)
                    if np.isin(test_idx, train_index_list) == False:
                        real_prob = dssm_model.predict(test_idx)
                        print(real_prob.shape)
            sys.exit()

        iteration = (sample_size / cfg.batch_size) if cfg.epoch_size < sample_size / cfg.batch_size else cfg.iteration
        for epoch_step in range(cfg.epoch_size):
            epoch_loss = 0.0
            for iter in range(iteration):
                train_idx = iter % (sample_size / cfg.batch_size)
                if np.isin(train_idx, train_index_list) == True:
                    if iter % 100 == 0:
                        _, merged, iter_loss = dssm_obj.train(train_idx, True)
                        train_writer.add_summary(merged, iter * epoch_step)
                    else:
                        iter_loss = dssm_obj.train(train_idx)
                    print("epoch %d : iteration %d, loss is %f" % (epoch_step, iter, iter_loss))
                    epoch_loss += iter_loss
            epoch_loss /= len(train_index_list)
            train_loss = dssm_obj.get_loss_summary(epoch_loss)
            train_writer.add_summary(train_loss, epoch_step + 1)

            epoch_accuracy = 0.0
            for iter in range(iteration):
                test_idx = iter % (sample_size / cfg.batch_size)
                if np.isin(test_idx, train_index_list) == False:
                    iter_accuracy = dssm_obj.validate(test_idx)
                    print("epoch %d : iteration %d, accuracy is %f" % (epoch_step, iter, iter_accuracy))
                    epoch_accuracy += iter_accuracy
            epoch_accuracy /= (sample_size / cfg.batch_size - len(train_index_list))
            test_accuracy = dssm_obj.get_accuracy_summary(epoch_accuracy)
            test_writer.add_summary(test_accuracy, epoch_step + 1)
        dssm_obj.save()
        sess.close()