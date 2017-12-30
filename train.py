import numpy as np
import Net


mnist = np.load('mnist_data_label.npz')
image = (mnist['data']/255.0-0.5)*2
label = mnist['label']

test = np.load('test_data_label.npz')
test_img = (test['data']/255.0-0.5)*2
test_label = test['label']

max_iter = 10000
batch_size = 16
alpha = 100

net = Net.Net(batch_size, alpha, max_iter)
net.train(image, label)
net.test(test_img, test_label)
