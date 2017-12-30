import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def weight_var(shape):
    init = tf.truncated_normal(shape=shape, mean=0.001, stddev=0.01, dtype=tf.float32)
    return tf.Variable(init)


def bias_var(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=0.001, stddev=0.01))


def conv_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Net:
    def __init__(self, batch_size, alpha, max_iter):
        self.init_var()
        self.img_size = 28
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.anc = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.pos = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.neg = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.anc_out = self.network(self.anc)
        self.pos_out = self.network(self.pos)
        self.neg_out = self.network(self.neg)

        self.test_input = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.test_out = self.network(self.test_input)

    def init_var(self):
        self.w1 = weight_var([5, 5, 1, 32])
        self.b1 = bias_var([32])
        self.w2 = weight_var([3, 3, 32, 64])
        self.b2 = bias_var([64])
        self.w3 = weight_var([3, 3, 64, 128])
        self.b3 = bias_var([128])

        self.w4 = weight_var([4 * 4 * 128, 512])
        self.b4 = bias_var([512])
        self.w5 = weight_var([512, 2])
        self.b5 = bias_var([2])

    def network(self, x):
        # part 1:
        conv_1 = tf.nn.relu(conv_2d(x, self.w1) + self.b1)
        pool_1 = max_pool2x2(conv_1)
        conv_2 = tf.nn.relu(conv_2d(pool_1, self.w2) + self.b2)
        pool_2 = max_pool2x2(conv_2)
        conv_3 = tf.nn.relu(conv_2d(pool_2, self.w3) + self.b3)
        pool_3 = max_pool2x2(conv_3)
        a = tf.reshape(pool_3, [-1, 4 * 4 * 128])
        ip1 = tf.nn.relu(tf.matmul(a, self.w4) + self.b4)
        return tf.matmul(ip1, self.w5) + self.b5

    def loss(self):
        pos = tf.reduce_sum(tf.pow(self.anc_out-self.pos_out, 2), 1)
        neg = tf.reduce_sum(tf.pow(self.anc_out-self.neg_out, 2), 1)
        return tf.reduce_mean(tf.maximum(pos-neg+self.alpha, 0))

    @staticmethod
    def get_pair(image, label):
        s = np.array(range(np.size(label)))
        ind = np.random.randint(0, len(label), [1])
        lb = label[ind]
        pick_same = s[label == lb]
        pick_diff = s[lb != label]
        anc = image[ind, :, :]
        pos_label = -1
        neg_label = lb
        while pos_label != lb:
            x = np.random.randint(0, len(pick_same), [1])
            pos_label = label[x]
            pos = image[x, :, :]
        while neg_label == lb:
            x = np.random.randint(0, len(pick_diff), [1])
            neg_label = label[x]
            neg = image[x, :, :]
        return np.reshape(anc, [1, 28, 28, 1]), np.reshape(pos, [1, 28, 28, 1]), np.reshape(neg, [1, 28, 28, 1])

    def train(self, image, label):
        np.random.seed(123456)
        plt.figure(0)
        plt.ion()

        loss = self.loss()
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for i in range(self.max_iter):
            anc = np.zeros([self.batch_size, 28, 28, 1])
            pos = np.zeros([self.batch_size, 28, 28, 1])
            neg = np.zeros([self.batch_size, 28, 28, 1])
            for j in range(self.batch_size):
                anc[j, :, :, :], pos[j, :, :, :], neg[j, :, :, :] = self.get_pair(image, label)

            sess.run([train_step], feed_dict={self.anc: anc, self.pos: pos, self.neg: neg})
            if i % 20 == 0:
                ll = sess.run([loss], feed_dict={self.anc: anc, self.pos: pos, self.neg: neg})
                print 'iter:', i, ' loss:', ll
                plt.plot(i, ll, 'rx')
                plt.pause(0.001)

        plt.show()
        saver = tf.train.Saver()
        saver.save(sess, 'triplet.ckpt')
        sess.close()

    def test(self, img, label):
        plt.figure(1)
        plt.ion()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, 'triplet.ckpt')
        test_num = 1000
        for i in range(test_num):
            out = sess.run(self.test_out, feed_dict={self.test_input: np.reshape(img[i, :, :], [1, 28, 28, 1])})
            out = out[0]
            color = '#%06x'%(label[i]*1396745)
            plt.plot(out[0], out[1], 'x', color=color)
            #plt.pause(0.00001)

        plt.ioff()
        plt.show()
